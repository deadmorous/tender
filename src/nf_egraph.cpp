#include <tender/nf_egraph.hpp>

#include <mpk/mix/util/overloads.hpp>
#include <tender/context.hpp>
#include <tender/derivation.hpp> // steps::canonicalize
#include <tender/nf_lower.hpp>   // canonicalize_nf

#include <cstdint>
#include <unordered_map>
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
        return a;
    }

    auto node_count() const -> std::size_t
    {
        return memo_.size();
    }
};

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
