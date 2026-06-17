#include <tender/egraph.hpp>

#include <mpk/mix/util/overloads.hpp>
#include <tender/context.hpp>
#include <tender/derivation.hpp> // steps::canonicalize, structural_eq

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
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
                    std::size_t c = 1;
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
};

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
auto EGraph::class_count() -> std::size_t
{
    return impl_->classes_.size();
}
auto EGraph::node_count() const -> std::size_t
{
    return impl_->memo_.size();
}

} // namespace tender
