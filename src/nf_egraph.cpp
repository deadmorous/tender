#include <tender/nf_egraph.hpp>

#include <mpk/mix/util/overloads.hpp>
#include <tender/context.hpp>
#include <tender/derivation.hpp> // steps::canonicalize
#include <tender/nf_lower.hpp>   // canonicalize_nf, raise
#include <tender/nf_match.hpp>   // fire_identity_on_term

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mpk::mix;

namespace tender::nf
{

namespace
{

// The operator tag of an e-node, mirroring the `Nf` structure: the `Factor`
// variants, the multiplicative `Term`, and the additive `Sum`.
enum class NfEKind : std::uint8_t
{
    Atom,
    Contraction,
    Cross,
    Paren,
    Unary,
    Div,
    Term,
    Sum,
};

// One e-node: an `Nf` constructor applied to child e-classes.
//
//   Atom        : leaf — its whole `Factor` in `atom`, no children.
//   Contraction : `children` factor classes, `ops` the join operators.
//   Cross       : `children` factor classes.
//   Paren       : one child — the body `Sum` class.
//   Unary       : one child — the operand factor class; `uop` the operator.
//   Div         : two children — the num and den `Sum` classes.
//   Term        : `scalar_count` scalar classes then the tensor classes;
//                 `coeff` and `bound` carried inline.
//   Sum         : `children` term classes (the additive layer).
struct NfENode final
{
    NfEKind kind;
    Factor const* atom = nullptr;
    std::vector<EClassId> children;
    std::vector<COp> ops = {};
    UnaryOp uop = UnaryOp::Trace;
    Rational coeff = Rational{1};
    std::vector<BoundIndex> bound = {};
    std::size_t scalar_count = 0;
};

void hash_combine(std::size_t& seed, std::size_t v)
{
    seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

// Per-node extraction weight, lexicographic: minimize the number of Levi-Civita
// symbols first (the object the index identities exist to contract away — its
// δ-expansion is often *larger*, so a plain node count would keep the ε form),
// then total node count.  The weight dominates any realistic count, giving the
// (eps-count, node-count) ordering.  Mirrors the Expr e-graph's `node_cost`.
constexpr std::size_t kLeviCivitaWeight = 1'000'000;

auto node_cost(NfENode const& n) -> std::size_t
{
    if (n.kind == NfEKind::Atom)
        if (auto const* a = std::get_if<Atom>(&n.atom->node))
            if (a->obj.traits
                && a->obj.traits->well_known == WellKnownKind::LeviCivita)
                return kLeviCivitaWeight;
    return 1;
}

auto node_hash(NfENode const& n) -> std::size_t
{
    std::size_t seed =
        std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(n.kind));
    for (EClassId c: n.children)
        hash_combine(seed, std::hash<EClassId>{}(c));
    switch (n.kind)
    {
        case NfEKind::Atom: hash_combine(seed, hash(*n.atom)); break;
        case NfEKind::Contraction:
            for (COp o: n.ops)
                hash_combine(seed, std::size_t(o));
            break;
        case NfEKind::Unary: hash_combine(seed, std::size_t(n.uop)); break;
        case NfEKind::Term:
            hash_combine(seed, std::hash<std::int64_t>{}(n.coeff.num()));
            hash_combine(seed, std::hash<std::int64_t>{}(n.coeff.den()));
            hash_combine(seed, n.scalar_count);
            for (auto const& b: n.bound)
            {
                hash_combine(
                    seed, std::size_t(b.index.id) * 3 + std::size_t(b.mode));
                hash_combine(seed, b.range ? hash(*b.range) + 1 : 0);
            }
            break;
        default: break;
    }
    return seed;
}

auto bound_eq(
    std::vector<BoundIndex> const& a, std::vector<BoundIndex> const& b) -> bool
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].index.id != b[i].index.id || a[i].mode != b[i].mode)
            return false;
        if (!equal(a[i].range, b[i].range))
            return false;
    }
    return true;
}

auto node_eq(NfENode const& a, NfENode const& b) -> bool
{
    if (a.kind != b.kind || a.children != b.children)
        return false;
    switch (a.kind)
    {
        case NfEKind::Atom: return equal(a.atom, b.atom);
        case NfEKind::Contraction: return a.ops == b.ops;
        case NfEKind::Unary: return a.uop == b.uop;
        case NfEKind::Term:
            return a.coeff == b.coeff && a.scalar_count == b.scalar_count
                   && bound_eq(a.bound, b.bound);
        default: return true;
    }
}

struct NodeHash final
{
    auto operator()(NfENode const& n) const -> std::size_t
    {
        return node_hash(n);
    }
};
struct NodeEq final
{
    auto operator()(NfENode const& a, NfENode const& b) const -> bool
    {
        return node_eq(a, b);
    }
};

} // namespace

struct NfEGraph::Impl final
{
    Context* ctx_;

    struct EClass final
    {
        std::vector<NfENode> nodes;
        std::vector<std::pair<NfENode, EClassId>> parents;
    };

    std::vector<EClassId> parent_;
    std::unordered_map<EClassId, EClass> classes_;
    std::unordered_map<NfENode, EClassId, NodeHash, NodeEq> memo_;
    std::vector<EClassId> worklist_;

    explicit Impl(Context& ctx) : ctx_(&ctx)
    {
    }

    auto find(EClassId x) -> EClassId
    {
        while (parent_[static_cast<std::size_t>(x)] != x)
        {
            parent_[static_cast<std::size_t>(x)] =
                parent_[static_cast<std::size_t>(
                    parent_[static_cast<std::size_t>(x)])];
            x = parent_[static_cast<std::size_t>(x)];
        }
        return x;
    }

    auto canon_node(NfENode n) -> NfENode
    {
        for (auto& c: n.children)
            c = find(c);
        return n;
    }

    auto add_node(NfENode n) -> EClassId
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

    auto add_factor(Factor const* f) -> EClassId
    {
        return visit(
            Overloads{
                [&](Atom const&) -> EClassId
                { return add_node(NfENode{NfEKind::Atom, f, {}}); },
                [&](Contraction const& c) -> EClassId
                {
                    std::vector<EClassId> ch;
                    ch.reserve(c.factors.size());
                    for (auto const* x: c.factors)
                        ch.push_back(add_factor(x));
                    NfENode n{NfEKind::Contraction, nullptr, std::move(ch)};
                    n.ops = c.ops;
                    return add_node(std::move(n));
                },
                [&](Cross const& c) -> EClassId
                {
                    std::vector<EClassId> ch;
                    ch.reserve(c.factors.size());
                    for (auto const* x: c.factors)
                        ch.push_back(add_factor(x));
                    return add_node(
                        NfENode{NfEKind::Cross, nullptr, std::move(ch)});
                },
                [&](Paren const& p) -> EClassId {
                    return add_node(
                        NfENode{NfEKind::Paren, nullptr, {add_nf(p.body)}});
                },
                [&](Unary const& u) -> EClassId
                {
                    NfENode n{NfEKind::Unary, nullptr, {add_factor(u.operand)}};
                    n.uop = u.op;
                    return add_node(std::move(n));
                },
                [&](Div const& d) -> EClassId
                {
                    return add_node(NfENode{
                        NfEKind::Div, nullptr, {add_nf(d.num), add_nf(d.den)}});
                },
                // Scalar fields (vibe 000069): opaque leaves — stored whole via
                // the generic Atom node path, so the e-graph carries them
                // through without rewriting their interior (the targeted scalar
                // simplifier handles those separately).
                [&](ScalarFn const&) -> EClassId
                { return add_node(NfENode{NfEKind::Atom, f, {}}); },
                [&](Pow const&) -> EClassId
                { return add_node(NfENode{NfEKind::Atom, f, {}}); },
            },
            *f);
    }

    auto add_term(Term const& t) -> EClassId
    {
        std::vector<EClassId> ch;
        ch.reserve(t.scalars.size() + t.tensors.size());
        for (auto const* s: t.scalars)
            ch.push_back(add_factor(s));
        for (auto const* tn: t.tensors)
            ch.push_back(add_factor(tn));
        NfENode n{NfEKind::Term, nullptr, std::move(ch)};
        n.coeff = t.coeff;
        n.bound = t.bound;
        n.scalar_count = t.scalars.size();
        return add_node(std::move(n));
    }

    auto add_nf(Nf const* nf) -> EClassId
    {
        std::vector<EClassId> ch;
        if (nf)
        {
            ch.reserve(nf->terms.size());
            for (auto const& t: nf->terms)
                ch.push_back(add_term(t));
        }
        return add_node(NfENode{NfEKind::Sum, nullptr, std::move(ch)});
    }

    auto merge(EClassId a, EClassId b) -> EClassId
    {
        a = find(a);
        b = find(b);
        if (a == b)
            return a;
        parent_[static_cast<std::size_t>(b)] = a;
        auto& ca = classes_[a];
        auto& cb = classes_[b];
        ca.nodes.insert(ca.nodes.end(), cb.nodes.begin(), cb.nodes.end());
        ca.parents.insert(
            ca.parents.end(), cb.parents.begin(), cb.parents.end());
        classes_.erase(b);
        worklist_.push_back(a);
        return a;
    }

    // ---- congruence ------------------------------------------------------

    void repair(EClassId cls)
    {
        auto parents = classes_[cls].parents; // copy: merges mutate classes_

        for (auto const& [pnode, pclass]: parents)
        {
            memo_.erase(pnode);
            memo_[canon_node(pnode)] = find(pclass);
        }

        std::unordered_map<NfENode, EClassId, NodeHash, NodeEq> deduped;
        for (auto const& [pnode, pclass]: parents)
        {
            NfENode c = canon_node(pnode);
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

    // The cheapest e-node per class by total node count (a fixpoint over the
    // child costs, so cyclic classes converge).
    auto compute_best() -> std::unordered_map<EClassId, NfENode>
    {
        std::unordered_map<EClassId, std::size_t> cost;
        std::unordered_map<EClassId, NfENode> best;
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (auto const& [id, ec]: classes_)
                for (NfENode const& n: ec.nodes)
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
        return best;
    }

    auto reconstruct_nf(
        EClassId cls,
        std::unordered_map<EClassId, NfENode> const& best,
        std::unordered_map<EClassId, Nf const*>& nfmemo,
        std::unordered_map<EClassId, Factor const*>& fmemo) -> Nf const*;

    auto reconstruct_factor(
        EClassId cls,
        std::unordered_map<EClassId, NfENode> const& best,
        std::unordered_map<EClassId, Nf const*>& nfmemo,
        std::unordered_map<EClassId, Factor const*>& fmemo) -> Factor const*
    {
        cls = find(cls);
        if (auto it = fmemo.find(cls); it != fmemo.end())
            return it->second;
        NfENode const& n = best.at(cls);
        Context& ctx = *ctx_;
        auto fac = [&](std::size_t i)
        { return reconstruct_factor(n.children[i], best, nfmemo, fmemo); };
        Factor const* r = nullptr;
        switch (n.kind)
        {
            case NfEKind::Atom: r = n.atom; break;
            case NfEKind::Contraction:
            {
                std::vector<Factor const*> fs;
                for (std::size_t i = 0; i < n.children.size(); ++i)
                    fs.push_back(fac(i));
                r = make_contraction(ctx, std::move(fs), n.ops);
                break;
            }
            case NfEKind::Cross:
            {
                std::vector<Factor const*> fs;
                for (std::size_t i = 0; i < n.children.size(); ++i)
                    fs.push_back(fac(i));
                r = make_cross(ctx, std::move(fs));
                break;
            }
            case NfEKind::Paren:
                r = make_paren(
                    ctx, reconstruct_nf(n.children[0], best, nfmemo, fmemo));
                break;
            case NfEKind::Unary: r = make_unary(ctx, n.uop, fac(0)); break;
            case NfEKind::Div:
                r = make_div(
                    ctx,
                    reconstruct_nf(n.children[0], best, nfmemo, fmemo),
                    reconstruct_nf(n.children[1], best, nfmemo, fmemo));
                break;
            case NfEKind::Term:
            case NfEKind::Sum: break; // not a factor sort
        }
        fmemo.emplace(cls, r);
        return r;
    }

    auto reconstruct_term(
        EClassId cls,
        std::unordered_map<EClassId, NfENode> const& best,
        std::unordered_map<EClassId, Nf const*>& nfmemo,
        std::unordered_map<EClassId, Factor const*>& fmemo) -> Term
    {
        NfENode const& n = best.at(find(cls));
        Term t;
        t.coeff = n.coeff;
        t.bound = n.bound;
        for (std::size_t i = 0; i < n.children.size(); ++i)
        {
            Factor const* f =
                reconstruct_factor(n.children[i], best, nfmemo, fmemo);
            if (i < n.scalar_count)
                t.scalars.push_back(f);
            else
                t.tensors.push_back(f);
        }
        return t;
    }

    auto node_count() const -> std::size_t
    {
        return memo_.size();
    }

    // ---- saturation ------------------------------------------------------
    //
    // E-class matching is the `nf_match` matcher run over the graph: every
    // additive (`Sum`) e-node is a candidate rewrite site, its term children
    // reconstructed at their cheapest form.  A single-term rule fires on a term
    // (sub-product or sub-chain) exactly as in `apply_identity`; the resulting
    // term(s) replace that one term, leaving the rest of the sum intact, to
    // form the rewritten `Nf`.  Splitting read (collect) and write (insert +
    // merge) phases keeps matching off a half-mutated graph, mirroring the Expr
    // `EGraph::saturate`.

    // Read phase: every (Sum class, rewritten Nf) a compiled rule produces over
    // a stable graph.  The rewritten Nf is raw (freshened RHS dummies, unmerged
    // terms); the caller canonicalizes it before insertion.
    auto collect_rewrites(
        std::vector<std::pair<Term, Nf const*>> const& compiled)
        -> std::vector<std::pair<EClassId, Nf const*>>
    {
        auto best = compute_best();
        std::unordered_map<EClassId, Nf const*> nfmemo;
        std::unordered_map<EClassId, Factor const*> fmemo;
        std::vector<std::pair<EClassId, Nf const*>> out;

        for (auto const& [id, ec]: classes_)
            for (NfENode const& sumnode: ec.nodes)
            {
                if (sumnode.kind != NfEKind::Sum)
                    continue;
                std::vector<Term> terms;
                terms.reserve(sumnode.children.size());
                for (EClassId tc: sumnode.children)
                    terms.push_back(reconstruct_term(tc, best, nfmemo, fmemo));

                for (auto const& [lhs_term, rhs]: compiled)
                    for (std::size_t i = 0; i < terms.size(); ++i)
                        if (auto rep = fire_identity_on_term(
                                *ctx_, lhs_term, rhs, terms[i]))
                        {
                            std::vector<Term> nt;
                            nt.reserve(terms.size() + rep->size());
                            for (std::size_t k = 0; k < terms.size(); ++k)
                                if (k == i)
                                    nt.insert(
                                        nt.end(), rep->begin(), rep->end());
                                else
                                    nt.push_back(terms[k]);
                            out.emplace_back(id, make_nf(*ctx_, std::move(nt)));
                        }
            }
        return out;
    }

    auto saturate(std::vector<Identity> const& rules, int max_iterations) -> int
    {
        Context& ctx = *ctx_;

        // Compile once: each rule's LHS / RHS canonicalized to `Nf`.  Only a
        // single-term LHS is matched as a sub-product / sub-chain; a multi-term
        // LHS (a sub-sum pattern) has no Nf matcher yet and is skipped.
        std::vector<std::pair<Term, Nf const*>> compiled;
        compiled.reserve(rules.size());
        for (auto const& r: rules)
        {
            auto const* lhs =
                canonicalize_nf(ctx, steps::canonicalize(ctx, r.lhs));
            auto const* rhs =
                canonicalize_nf(ctx, steps::canonicalize(ctx, r.rhs));
            if (lhs->terms.size() != 1)
                continue;
            compiled.emplace_back(lhs->terms.front(), rhs);
        }

        int passes = 0;
        while (passes < max_iterations)
        {
            ++passes;

            auto rewrites = collect_rewrites(compiled);

            bool changed = false;
            for (auto const& [cls, raw]: rewrites)
            {
                // Re-canonicalize the spliced Nf (re-α-rename freshened RHS
                // dummies, merge like terms) before hash-consing it in.
                auto const* canon = canonicalize_nf(
                    ctx, steps::canonicalize(ctx, raise(ctx, *raw)));
                EClassId const rcls = add_nf(canon);
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

auto NfEGraph::Impl::reconstruct_nf(
    EClassId cls,
    std::unordered_map<EClassId, NfENode> const& best,
    std::unordered_map<EClassId, Nf const*>& nfmemo,
    std::unordered_map<EClassId, Factor const*>& fmemo) -> Nf const*
{
    cls = find(cls);
    if (auto it = nfmemo.find(cls); it != nfmemo.end())
        return it->second;
    NfENode const& n = best.at(cls);
    std::vector<Term> terms;
    terms.reserve(n.children.size());
    for (EClassId ch: n.children)
        terms.push_back(reconstruct_term(ch, best, nfmemo, fmemo));
    Nf const* r = make_nf(*ctx_, std::move(terms));
    nfmemo.emplace(cls, r);
    return r;
}

NfEGraph::NfEGraph(Context& ctx) : impl_(std::make_unique<Impl>(ctx))
{
}
NfEGraph::~NfEGraph() = default;
NfEGraph::NfEGraph(NfEGraph&&) noexcept = default;
auto NfEGraph::operator=(NfEGraph&&) noexcept -> NfEGraph& = default;

auto NfEGraph::add(Nf const* nf) -> EClassId
{
    return impl_->add_nf(nf);
}

auto NfEGraph::add(Expr const* e) -> EClassId
{
    return impl_->add_nf(
        canonicalize_nf(*impl_->ctx_, steps::canonicalize(*impl_->ctx_, e)));
}

void NfEGraph::rebuild()
{
    impl_->rebuild();
}

auto NfEGraph::extract(EClassId id) -> Nf const*
{
    auto best = impl_->compute_best();
    std::unordered_map<EClassId, Nf const*> nfmemo;
    std::unordered_map<EClassId, Factor const*> fmemo;
    return impl_->reconstruct_nf(id, best, nfmemo, fmemo);
}

auto NfEGraph::saturate(std::vector<Identity> const& rules, int max_iterations)
    -> int
{
    return impl_->saturate(rules, max_iterations);
}

auto NfEGraph::merge(EClassId a, EClassId b) -> EClassId
{
    return impl_->merge(a, b);
}

auto NfEGraph::find(EClassId id) -> EClassId
{
    return impl_->find(id);
}

auto NfEGraph::class_count() -> std::size_t
{
    return impl_->classes_.size();
}

auto NfEGraph::node_count() const -> std::size_t
{
    return impl_->node_count();
}

} // namespace tender::nf
