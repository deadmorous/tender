#include <tender/egraph.hpp>

#include <mpk/mix/util/overloads.hpp>
#include <tender/context.hpp>
#include <tender/derivation.hpp> // steps::canonicalize, structural_eq, is_component_valued
#include <tender/identity.hpp> // match_into, bind_pattern_index

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mpk::mix;

namespace tender
{

namespace
{

// The operator tag of an e-node.  Controls reconstruction; kept independent of
// the Expr::Node variant ordering so the two can evolve separately.
enum class EKind : std::uint8_t
{
    TensorObject,
    ScalarLiteral,
    Negate,
    Sum,
    Difference,
    TensorProduct,
    ScalarDiv,
    Dot,
    DDot,
    DDotAlt,
    Cross,
    ExplicitSum,
    NoSum,
};

// One e-node: an operator applied to child e-classes.
//
// Leaves (TensorObject, ScalarLiteral) carry their whole Expr in `leaf` and
// have no children; their identity is structural.  Internal nodes carry their
// child class ids; ExplicitSum/NoSum additionally carry their (already
// α-canonical) binder id.  Children are kept canonical (find()-ed) by the
// EGraph.
struct ENode final
{
    EKind kind;
    Expr const* leaf = nullptr;
    int binder = 0;
    std::vector<EClassId> children;
};

void hash_combine(std::size_t& seed, std::size_t v)
{
    seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

// Flatten a left/right pattern tree of node kind NodeT into its leaf operands
// (mirrors the canonicalizer's flatten; used to AC-match a commutative
// pattern).
template <typename NodeT>
void flatten_pat(Expr const* e, std::vector<Expr const*>& out)
{
    if (auto const* n = std::get_if<NodeT>(&e->node))
    {
        flatten_pat<NodeT>(n->left, out);
        flatten_pat<NodeT>(n->right, out);
    }
    else
        out.push_back(e);
}

// Structural hash of a leaf Expr, consistent with structural_eq (which compares
// leaves field by field).  Space pointers are deliberately omitted — equal
// leaves share the same fields, and omitting a field only ever adds collisions,
// which structural equality then resolves.
auto leaf_hash(Expr const* e) -> std::size_t
{
    std::size_t seed = 0;
    visit(
        Overloads{
            [&](ScalarLiteral const& s)
            {
                hash_combine(seed, std::hash<std::int64_t>{}(s.value.num()));
                hash_combine(seed, std::hash<std::int64_t>{}(s.value.den()));
            },
            [&](TensorObject const& t)
            {
                hash_combine(
                    seed, std::hash<std::string_view>{}(t.name.v.view()));
                hash_combine(seed, t.rank ? std::size_t(*t.rank) + 1 : 0);
                hash_combine(seed, t.slots.size());
                for (auto const& sb: t.slots)
                {
                    hash_combine(seed, std::size_t(sb.slot.level));
                    hash_combine(seed, std::size_t(sb.slot.realm));
                    if (!sb.index)
                    {
                        hash_combine(seed, 0);
                        continue;
                    }
                    visit(
                        Overloads{
                            [&](CountableIndex const& c)
                            { hash_combine(seed, std::size_t(c.id) * 3 + 1); },
                            [&](ConcreteIndex const& c) {
                                hash_combine(seed, std::size_t(c.value) * 3 + 2);
                            },
                            [&](LabelIndex const& c) {
                                hash_combine(
                                    seed,
                                    std::hash<std::string_view>{}(
                                        c.name.v.view())
                                            * 3
                                        + 3);
                            },
                        },
                        *sb.index);
                }
            },
            [&](auto const&) {}, // GCOV_EXCL_LINE  (leaf_hash sees only leaves)
        },
        *e);
    return seed;
}

// Per-node extraction weight.  Cost is lexicographic: minimize the number of
// Levi-Civita symbols first (the object the index identities exist to contract
// away — its δ-expansion is often *larger*, so a plain node count would keep
// the ε form), then total node count.  The weight dominates any realistic node
// count, giving the (eps-count, node-count) ordering.
constexpr std::size_t kLeviCivitaWeight = 1'000'000;

auto node_cost(ENode const& n) -> std::size_t
{
    if (n.leaf)
        if (auto const* t = std::get_if<TensorObject>(&n.leaf->node))
            if (t->traits && t->traits->well_known == WellKnownKind::LeviCivita)
                return kLeviCivitaWeight;
    return 1;
}

struct ENodeHash final
{
    auto operator()(ENode const& n) const -> std::size_t
    {
        std::size_t seed = std::hash<std::uint8_t>{}(std::uint8_t(n.kind));
        hash_combine(seed, std::size_t(n.binder));
        for (EClassId c: n.children)
            hash_combine(seed, std::size_t(c));
        if (n.leaf)
            hash_combine(seed, leaf_hash(n.leaf));
        return seed;
    }
};

struct ENodeEq final
{
    auto operator()(ENode const& a, ENode const& b) const -> bool
    {
        if (a.kind != b.kind || a.binder != b.binder
            || a.children != b.children)
            return false; // GCOV_EXCL_LINE  (only on a hash-bucket collision)
        if (a.leaf == b.leaf)
            return true;
        return a.leaf && b.leaf && structural_eq(a.leaf, b.leaf);
    }
};

} // namespace

struct EGraph::Impl final
{
    explicit Impl(Context& ctx) : ctx_{&ctx}
    {
    }

    Context* ctx_;

    // Union-find over class ids.
    std::vector<EClassId> parent_;

    struct EClass final
    {
        std::vector<ENode> nodes;
        // Parents: (e-node that has this class as a child, owning class id).
        std::vector<std::pair<ENode, EClassId>> parents;
    };

    // Only canonical (root) ids have an entry.
    std::unordered_map<EClassId, EClass> classes_;
    // Hash-cons: canonical e-node -> class id.
    std::unordered_map<ENode, EClassId, ENodeHash, ENodeEq> memo_;
    std::vector<EClassId> worklist_;

    auto find(EClassId x) -> EClassId
    {
        while (parent_[x] != x)
        {
            parent_[x] = parent_[parent_[x]]; // path halving
            x = parent_[x];
        }
        return x;
    }

    auto canon_node(ENode n) -> ENode
    {
        for (auto& c: n.children)
            c = find(c);
        return n;
    }

    auto add_node(ENode n) -> EClassId
    {
        n = canon_node(std::move(n));
        if (auto it = memo_.find(n); it != memo_.end())
            return find(it->second);

        EClassId const id = static_cast<EClassId>(parent_.size());
        parent_.push_back(id);
        auto& ec = classes_[id];
        ec.nodes.push_back(n);
        for (EClassId c: n.children)
            classes_[c].parents.emplace_back(n, id);
        memo_.emplace(std::move(n), id);
        return id;
    }

    auto add_canon(Expr const* e) -> EClassId
    {
        auto bin = [&](EKind k, Expr const* l, Expr const* r) {
            return add_node(ENode{k, nullptr, 0, {add_canon(l), add_canon(r)}});
        };
        return visit(
            Overloads{
                [&](TensorObject const&)
                { return add_node(ENode{EKind::TensorObject, e, 0, {}}); },
                [&](ScalarLiteral const&)
                { return add_node(ENode{EKind::ScalarLiteral, e, 0, {}}); },
                [&](Negate const& n) {
                    return add_node(ENode{
                        EKind::Negate, nullptr, 0, {add_canon(n.operand)}});
                },
                [&](Sum const& s) { return bin(EKind::Sum, s.left, s.right); },
                // GCOV_EXCL_START  canonicalize carries signs as coefficients,
                // so a canonical form (all that add() inserts) never holds a
                // Difference node; kept for variant totality.
                [&](Difference const& d)
                { return bin(EKind::Difference, d.left, d.right); },
                // GCOV_EXCL_STOP
                [&](TensorProduct const& p)
                { return bin(EKind::TensorProduct, p.left, p.right); },
                [&](ScalarDiv const& d)
                { return bin(EKind::ScalarDiv, d.left, d.right); },
                [&](Dot const& d) { return bin(EKind::Dot, d.left, d.right); },
                [&](DDot const& d)
                { return bin(EKind::DDot, d.left, d.right); },
                [&](DDotAlt const& d)
                { return bin(EKind::DDotAlt, d.left, d.right); },
                [&](Cross const& c)
                { return bin(EKind::Cross, c.left, c.right); },
                [&](ExplicitSum const& s)
                {
                    std::vector<EClassId> ch{add_canon(s.body)};
                    if (s.bound)
                        ch.push_back(add_canon(s.bound));
                    return add_node(ENode{
                        EKind::ExplicitSum, nullptr, s.index.id, std::move(ch)});
                },
                [&](NoSum const& s)
                {
                    return add_node(ENode{
                        EKind::NoSum, nullptr, s.index.id, {add_canon(s.body)}});
                },
            },
            *e);
    }

    auto merge(EClassId a, EClassId b) -> EClassId
    {
        a = find(a);
        b = find(b);
        if (a == b)
            return a;
        parent_[b] = a;
        auto& ca = classes_[a];
        auto& cb = classes_[b];
        ca.nodes.insert(ca.nodes.end(), cb.nodes.begin(), cb.nodes.end());
        ca.parents.insert(
            ca.parents.end(), cb.parents.begin(), cb.parents.end());
        classes_.erase(b);
        worklist_.push_back(a);
        return a;
    }

    void repair(EClassId cls)
    {
        // cls stays a representative throughout: parents are structurally
        // distinct from cls, so no merge here can absorb cls itself.
        auto parents = classes_[cls].parents; // copy: merges mutate classes_

        for (auto const& [pnode, pclass]: parents)
        {
            memo_.erase(pnode);
            memo_[canon_node(pnode)] = find(pclass);
        }

        std::unordered_map<ENode, EClassId, ENodeHash, ENodeEq> deduped;
        for (auto const& [pnode, pclass]: parents)
        {
            ENode c = canon_node(pnode);
            if (auto it = deduped.find(c); it != deduped.end())
                merge(pclass, it->second);
            deduped[c] = find(pclass);
        }

        auto& ec = classes_[cls];
        ec.parents.clear();
        for (auto& [pnode, pclass]: deduped)
            ec.parents.emplace_back(pnode, find(pclass));
    }

    void rebuild()
    {
        while (!worklist_.empty())
        {
            std::vector<EClassId> todo;
            todo.swap(worklist_);
            std::unordered_set<EClassId> seen;
            for (EClassId c: todo)
            {
                c = find(c);
                if (seen.insert(c).second)
                    repair(c);
            }
        }
    }

    // ---- extraction ------------------------------------------------------

    auto reconstruct(
        EClassId cls,
        std::unordered_map<EClassId, ENode> const& best,
        std::unordered_map<EClassId, Expr const*>& memo) -> Expr const*
    {
        cls = find(cls);
        if (auto it = memo.find(cls); it != memo.end())
            return it->second;
        ENode const& n = best.at(cls);
        auto child = [&](std::size_t i)
        { return reconstruct(n.children[i], best, memo); };
        Context& ctx = *ctx_;
        Expr const* r = nullptr;
        switch (n.kind)
        {
            case EKind::TensorObject:
            case EKind::ScalarLiteral: r = n.leaf; break;
            case EKind::Negate: r = make_negate(ctx, child(0)); break;
            case EKind::Sum:
                r = make_sum(ctx, child(0), child(1));
                break;
                // GCOV_EXCL_START  (Difference never inserted; see add_canon)
            case EKind::Difference:
                r = make_difference(ctx, child(0), child(1));
                break;
                // GCOV_EXCL_STOP
            case EKind::TensorProduct:
                r = make_tensor_product(ctx, child(0), child(1));
                break;
            case EKind::ScalarDiv:
                r = make_scalar_div(ctx, child(0), child(1));
                break;
            case EKind::Dot: r = make_dot(ctx, child(0), child(1)); break;
            case EKind::DDot: r = make_ddot(ctx, child(0), child(1)); break;
            case EKind::DDotAlt:
                r = make_ddot_alt(ctx, child(0), child(1));
                break;
            case EKind::Cross: r = make_cross(ctx, child(0), child(1)); break;
            case EKind::ExplicitSum:
                r = make_explicit_sum(
                    ctx,
                    CountableIndex{n.binder},
                    child(0),
                    n.children.size() > 1 ? child(1) : nullptr);
                break;
            case EKind::NoSum:
                r = make_no_sum(ctx, CountableIndex{n.binder}, child(0));
                break;
        }
        memo.emplace(cls, r);
        return r;
    }

    auto extract(EClassId root) -> Expr const*
    {
        std::unordered_map<EClassId, std::size_t> cost;
        std::unordered_map<EClassId, ENode> best;

        bool changed = true;
        while (changed)
        {
            changed = false;
            for (auto const& [id, ec]: classes_)
            {
                for (ENode const& n: ec.nodes)
                {
                    std::size_t c = node_cost(n);
                    bool ok = true;
                    for (EClassId ch: n.children)
                    {
                        auto it = cost.find(find(ch));
                        if (it == cost.end())
                        {
                            ok = false;
                            break;
                        }
                        c += it->second;
                    }
                    if (!ok)
                        continue;
                    auto it = cost.find(id);
                    if (it == cost.end() || c < it->second)
                    {
                        cost[id] = c;
                        best[id] = n;
                        changed = true;
                    }
                }
            }
        }

        std::unordered_map<EClassId, Expr const*> memo;
        return reconstruct(root, best, memo);
    }

    // ---- e-matching ------------------------------------------------------
    //
    // A canonical pattern is matched against the e-graph by descending through
    // e-classes: each pattern node is matched against every e-node of the
    // candidate class, recursing into child classes.  Leaves and binder indices
    // reuse the Expr-level matcher (match_into / bind_pattern_index), so the
    // index-binding semantics live in one place.  Commutative nodes (Sum, and
    // component TensorProduct) flatten the pattern and the target and
    // AC-backtrack.
    //
    // The target flattening follows one same-kind e-node per class (single
    // flattening); enumerating alternative flattenings of multi-form classes is
    // a follow-up.  All other matching is complete over the class's e-nodes.

    using Bindings = std::vector<MatchBinding>;

    // Collect addend/factor classes of a commutative `k`-node tree rooted at
    // cls.
    void flatten_target(EKind k, EClassId cls, std::vector<EClassId>& out)
    {
        cls = find(cls);
        for (ENode const& n: classes_[cls].nodes)
            if (n.kind == k)
            {
                flatten_target(k, n.children[0], out);
                flatten_target(k, n.children[1], out);
                return;
            }
        out.push_back(cls);
    }

    auto match_class(Expr const* pat, EClassId cls, MatchBinding const& bnd)
        -> Bindings;

    auto match_binary(
        Expr const* pl,
        Expr const* pr,
        ENode const& n,
        MatchBinding const& bnd) -> Bindings
    {
        Bindings out;
        for (auto const& bl: match_class(pl, n.children[0], bnd))
        {
            auto br = match_class(pr, n.children[1], bl);
            out.insert(out.end(), br.begin(), br.end());
        }
        return out;
    } // GCOV_EXCL_LINE  (-O0 exception-cleanup landing pad)

    auto match_commutative(
        Expr const* pat,
        ENode const& n,
        EKind k,
        MatchBinding const& bnd) -> Bindings
    {
        std::vector<Expr const*> pats;
        if (k == EKind::Sum)
            flatten_pat<Sum>(pat, pats);
        else
            flatten_pat<TensorProduct>(pat, pats);

        std::vector<EClassId> tgts;
        flatten_target(k, n.children[0], tgts);
        flatten_target(k, n.children[1], tgts);
        if (pats.size() != tgts.size())
            return {};

        Bindings out;
        std::vector<bool> used(tgts.size(), false);
        auto rec = [&](auto&& self, std::size_t i, MatchBinding cur) -> void
        {
            if (i == pats.size())
            {
                out.push_back(std::move(cur));
                return;
            }
            for (std::size_t j = 0; j < tgts.size(); ++j)
            {
                if (used[j])
                    continue;
                used[j] = true;
                for (auto& b: match_class(pats[i], tgts[j], cur))
                    self(self, i + 1, b);
                used[j] = false;
            }
        };
        rec(rec, 0, bnd);
        return out;
    }

    auto match_node(Expr const* pat, ENode const& n, MatchBinding const& bnd)
        -> Bindings
    {
        auto leaf = [&](EKind k) -> Bindings
        {
            if (n.kind != k || !n.leaf)
                return {};
            MatchBinding b = bnd;
            if (match_into(pat, n.leaf, b))
                return {std::move(b)};
            return {};
        };
        auto bin = [&](EKind k, Expr const* l, Expr const* r) -> Bindings
        { return n.kind == k ? match_binary(l, r, n, bnd) : Bindings{}; };

        return visit(
            Overloads{
                [&](TensorObject const&) { return leaf(EKind::TensorObject); },
                [&](ScalarLiteral const&)
                { return leaf(EKind::ScalarLiteral); },
                [&](Negate const& p) -> Bindings
                {
                    return n.kind == EKind::Negate ?
                               match_class(p.operand, n.children[0], bnd) :
                               Bindings{};
                },
                [&](Sum const&) -> Bindings
                {
                    return n.kind == EKind::Sum ?
                               match_commutative(pat, n, EKind::Sum, bnd) :
                               Bindings{};
                },
                // GCOV_EXCL_START  the pattern is canonicalized before
                // matching, and canonical forms hold no Difference node (see
                // add_canon).
                [&](Difference const& p)
                { return bin(EKind::Difference, p.left, p.right); },
                // GCOV_EXCL_STOP
                [&](TensorProduct const& p) -> Bindings
                {
                    if (n.kind != EKind::TensorProduct)
                        return {};
                    if (is_component_valued(pat))
                        return match_commutative(
                            pat, n, EKind::TensorProduct, bnd);
                    return match_binary(p.left, p.right, n, bnd);
                },
                [&](ScalarDiv const& p)
                { return bin(EKind::ScalarDiv, p.left, p.right); },
                [&](Dot const& p) { return bin(EKind::Dot, p.left, p.right); },
                [&](DDot const& p)
                { return bin(EKind::DDot, p.left, p.right); },
                [&](DDotAlt const& p)
                { return bin(EKind::DDotAlt, p.left, p.right); },
                [&](Cross const& p)
                { return bin(EKind::Cross, p.left, p.right); },
                [&](ExplicitSum const& p) -> Bindings
                {
                    if (n.kind != EKind::ExplicitSum)
                        return {};
                    MatchBinding b = bnd;
                    // GCOV_EXCL_START  canonicalize gives every binder a
                    // distinct (depth-based) id, so this consistency check
                    // never fails on a canonical pattern; kept for robustness.
                    if (!bind_pattern_index(
                            b, p.index.id, CountableIndex{n.binder}))
                        return {};
                    // GCOV_EXCL_STOP
                    bool const pbound = p.bound != nullptr;
                    if (pbound != (n.children.size() > 1))
                        return {};
                    auto bodies = match_class(p.body, n.children[0], b);
                    if (!pbound)
                        return bodies;
                    Bindings out;
                    for (auto const& bb: bodies)
                    {
                        auto r = match_class(p.bound, n.children[1], bb);
                        out.insert(out.end(), r.begin(), r.end());
                    }
                    return out;
                },
                [&](NoSum const& p) -> Bindings
                {
                    if (n.kind != EKind::NoSum)
                        return {};
                    MatchBinding b = bnd;
                    // GCOV_EXCL_START  (distinct canonical binder ids; see
                    // above)
                    if (!bind_pattern_index(
                            b, p.index.id, CountableIndex{n.binder}))
                        return {};
                    // GCOV_EXCL_STOP
                    return match_class(p.body, n.children[0], b);
                },
            },
            *pat);
    }

    auto ematch(Expr const* pat)
        -> std::vector<std::pair<EClassId, MatchBinding>>
    {
        std::vector<std::pair<EClassId, MatchBinding>> out;
        for (auto const& [id, ec]: classes_)
            for (auto& b: match_class(pat, id, MatchBinding{}))
                out.emplace_back(id, std::move(b));
        return out;
    } // GCOV_EXCL_LINE  (-O0 exception-cleanup landing pad)

    // ---- saturation ------------------------------------------------------
    //
    // Equality saturation: apply every rule everywhere, to a fixed point.  Each
    // pass has a read phase (collect every (matched class, instantiated RHS) on
    // a stable graph) and a write phase (add each RHS and merge it into its
    // matched class), then one rebuild.  Splitting the phases keeps matching
    // off a half-mutated graph.
    //
    // A pass that merges nothing new is the fixed point.  An iteration cap
    // bounds size-increasing rule sets that never converge (vibe 000034); for
    // tender's size-reducing identities the loop settles in a few passes well
    // under the cap.

    auto saturate(std::vector<Identity> const& rules, int max_iterations) -> int
    {
        // Compile once: rules' LHS canonicalized for matching, RHS kept raw for
        // instantiation under each binding.
        std::vector<std::pair<Expr const*, Expr const*>> compiled;
        compiled.reserve(rules.size());
        for (auto const& r: rules)
            compiled.emplace_back(steps::canonicalize(*ctx_, r.lhs), r.rhs);

        int passes = 0;
        while (passes < max_iterations)
        {
            ++passes;

            // Read phase: gather rewrites without mutating the graph.
            std::vector<std::pair<EClassId, Expr const*>> rewrites;
            for (auto const& [lhs, rhs]: compiled)
                for (auto& [cls, bnd]: ematch(lhs))
                    rewrites.emplace_back(cls, instantiate(*ctx_, rhs, bnd));

            // Write phase: insert each RHS and merge it into its matched class.
            bool changed = false;
            for (auto const& [cls, rhs]: rewrites)
            {
                EClassId const rcls =
                    add_canon(steps::canonicalize(*ctx_, rhs));
                if (find(cls) != find(rcls))
                {
                    merge(cls, rcls);
                    changed = true;
                }
            }
            rebuild();
            if (!changed)
                break;
        }
        return passes;
    }
};

auto EGraph::Impl::match_class(
    Expr const* pat, EClassId cls, MatchBinding const& bnd) -> Bindings
{
    cls = find(cls);
    Bindings out;
    // Copy the node list: match_class is re-entrant but does not mutate
    // classes_; the copy guards against iterator surprises if that ever
    // changes.
    auto nodes = classes_[cls].nodes;
    for (ENode const& n: nodes)
    {
        auto r = match_node(pat, n, bnd);
        out.insert(out.end(), r.begin(), r.end());
    }
    return out;
}

EGraph::EGraph(Context& ctx) : impl_{std::make_unique<Impl>(ctx)}
{
}
EGraph::~EGraph() = default;
EGraph::EGraph(EGraph&&) noexcept = default;
auto EGraph::operator=(EGraph&&) noexcept -> EGraph& = default;

auto EGraph::add(Expr const* e) -> EClassId
{
    return impl_->add_canon(steps::canonicalize(*impl_->ctx_, e));
}
auto EGraph::merge(EClassId a, EClassId b) -> EClassId
{
    return impl_->merge(a, b);
}
void EGraph::rebuild()
{
    impl_->rebuild();
}
auto EGraph::find(EClassId id) -> EClassId
{
    return impl_->find(id);
}
auto EGraph::extract(EClassId id) -> Expr const*
{
    return impl_->extract(id);
}
auto EGraph::ematch(Expr const* pattern)
    -> std::vector<std::pair<EClassId, MatchBinding>>
{
    return impl_->ematch(steps::canonicalize(*impl_->ctx_, pattern));
}
auto EGraph::saturate(std::vector<Identity> const& rules, int max_iterations)
    -> int
{
    return impl_->saturate(rules, max_iterations);
}
auto EGraph::class_count() -> std::size_t
{
    return impl_->classes_.size();
}
auto EGraph::node_count() const -> std::size_t
{
    return impl_->memo_.size();
}

} // namespace tender
