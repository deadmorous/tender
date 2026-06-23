#include <tender/nf.hpp>

#include <tender/tensor_order.hpp>

#include <mpk/mix/util/overloads.hpp>

#include <functional>
#include <stdexcept>

namespace tender::nf
{

using mpk::mix::Overloads;

// ---- builders ----------------------------------------------------------

auto make_atom(Context& ctx, TensorObject obj) -> Factor const*
{
    return ctx.make<Factor>(Atom{std::move(obj)});
}

auto make_contraction(
    Context& ctx,
    std::vector<Factor const*> factors,
    std::vector<COp> ops) -> Factor const*
{
    if (factors.empty())
        throw std::invalid_argument("make_contraction: empty factor chain");
    if (ops.size() != factors.size() - 1)
        throw std::invalid_argument(
            "make_contraction: ops.size() must equal factors.size() - 1");
    return ctx.make<Factor>(
        Contraction{.factors = std::move(factors), .ops = std::move(ops)});
}

auto make_cross(Context& ctx, std::vector<Factor const*> factors)
    -> Factor const*
{
    if (factors.empty())
        throw std::invalid_argument("make_cross: empty factor chain");
    return ctx.make<Factor>(Cross{.factors = std::move(factors)});
}

auto make_paren(Context& ctx, Nf const* body) -> Factor const*
{
    if (body == nullptr)
        throw std::invalid_argument("make_paren: null body");
    return ctx.make<Factor>(Paren{.body = body});
}

auto make_unary(Context& ctx, UnaryOp op, Factor const* operand) -> Factor const*
{
    if (operand == nullptr)
        throw std::invalid_argument("make_unary: null operand");
    return ctx.make<Factor>(Unary{.op = op, .operand = operand});
}

auto make_nf(Context& ctx, std::vector<Term> terms) -> Nf const*
{
    return ctx.make<Nf>(Nf{.terms = std::move(terms)});
}

// ---- structural equality -----------------------------------------------

// Atom equality reuses the shared `tensor_object_cmp` (tensor_order.hpp), the
// same key the `Expr` canonical order uses — no duplicated slot comparison.
namespace
{

auto factor_seq_eq(
    std::vector<Factor const*> const& a,
    std::vector<Factor const*> const& b) -> bool
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!equal(a[i], b[i]))
            return false;
    return true;
}

} // namespace

auto equal(Factor const& a, Factor const& b) -> bool
{
    if (a.node.index() != b.node.index())
        return false;
    return visit(
        Overloads{
            [&](Atom const& fa) -> bool {
                return tensor_object_cmp(fa.obj, std::get<Atom>(b.node).obj)
                       == 0;
            },
            [&](Contraction const& fa) -> bool
            {
                auto const& fb = std::get<Contraction>(b.node);
                return fa.ops == fb.ops
                       && factor_seq_eq(fa.factors, fb.factors);
            },
            [&](Cross const& fa) -> bool
            {
                auto const& fb = std::get<Cross>(b.node);
                return factor_seq_eq(fa.factors, fb.factors);
            },
            [&](Paren const& fa) -> bool
            { return equal(fa.body, std::get<Paren>(b.node).body); },
            [&](Unary const& fa) -> bool
            {
                auto const& fb = std::get<Unary>(b.node);
                return fa.op == fb.op && equal(fa.operand, fb.operand);
            },
        },
        a);
}

auto equal(Term const& a, Term const& b) -> bool
{
    if (!(a.coeff == b.coeff))
        return false;
    if (a.bound.size() != b.bound.size())
        return false;
    for (std::size_t i = 0; i < a.bound.size(); ++i)
        if (a.bound[i].index.id != b.bound[i].index.id
            || a.bound[i].mode != b.bound[i].mode)
            return false;
    return factor_seq_eq(a.scalars, b.scalars)
           && factor_seq_eq(a.tensors, b.tensors);
}

auto equal(Nf const& a, Nf const& b) -> bool
{
    if (a.terms.size() != b.terms.size())
        return false;
    for (std::size_t i = 0; i < a.terms.size(); ++i)
        if (!equal(a.terms[i], b.terms[i]))
            return false;
    return true;
}

// ---- total order -------------------------------------------------------

namespace
{

// Length-then-elementwise order over a factor sequence (shorter first).
auto factor_seq_cmp(
    std::vector<Factor const*> const& a,
    std::vector<Factor const*> const& b) -> int
{
    if (a.size() != b.size())
        return a.size() < b.size() ? -1 : 1;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (int c = compare(*a[i], *b[i]))
            return c;
    return 0;
}

auto rational_cmp(Rational const& a, Rational const& b) -> int
{
    auto o = a <=> b;
    return o < 0 ? -1 : (o > 0 ? 1 : 0);
}

} // namespace

auto compare(Factor const& a, Factor const& b) -> int
{
    auto ka = a.node.index(), kb = b.node.index();
    if (ka != kb)
        return ka < kb ? -1 : 1;
    return visit(
        Overloads{
            [&](Atom const& fa) -> int
            { return tensor_object_cmp(fa.obj, std::get<Atom>(b.node).obj); },
            [&](Contraction const& fa) -> int
            {
                auto const& fb = std::get<Contraction>(b.node);
                if (int c = factor_seq_cmp(fa.factors, fb.factors))
                    return c;
                if (fa.ops.size() != fb.ops.size())
                    return fa.ops.size() < fb.ops.size() ? -1 : 1;
                for (std::size_t i = 0; i < fa.ops.size(); ++i)
                    if (fa.ops[i] != fb.ops[i])
                        return fa.ops[i] < fb.ops[i] ? -1 : 1;
                return 0;
            },
            [&](Cross const& fa) -> int {
                return factor_seq_cmp(
                    fa.factors, std::get<Cross>(b.node).factors);
            },
            [&](Paren const& fa) -> int
            { return compare(*fa.body, *std::get<Paren>(b.node).body); },
            [&](Unary const& fa) -> int
            {
                auto const& fb = std::get<Unary>(b.node);
                if (fa.op != fb.op)
                    return fa.op < fb.op ? -1 : 1;
                return compare(*fa.operand, *fb.operand);
            },
        },
        a);
}

auto compare(Term const& a, Term const& b) -> int
{
    if (int c = factor_seq_cmp(a.tensors, b.tensors))
        return c;
    if (int c = factor_seq_cmp(a.scalars, b.scalars))
        return c;
    if (a.bound.size() != b.bound.size())
        return a.bound.size() < b.bound.size() ? -1 : 1;
    for (std::size_t i = 0; i < a.bound.size(); ++i)
    {
        if (a.bound[i].index.id != b.bound[i].index.id)
            return a.bound[i].index.id < b.bound[i].index.id ? -1 : 1;
        if (a.bound[i].mode != b.bound[i].mode)
            return a.bound[i].mode < b.bound[i].mode ? -1 : 1;
    }
    return rational_cmp(a.coeff, b.coeff);
}

auto compare(Nf const& a, Nf const& b) -> int
{
    if (a.terms.size() != b.terms.size())
        return a.terms.size() < b.terms.size() ? -1 : 1;
    for (std::size_t i = 0; i < a.terms.size(); ++i)
        if (int c = compare(a.terms[i], b.terms[i]))
            return c;
    return 0;
}

// ---- structural hashing ------------------------------------------------

namespace
{

// boost-style hash combiner.
auto hash_mix(std::size_t seed, std::size_t v) -> std::size_t
{
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

auto hash_index_assoc(std::optional<IndexAssoc> const& a) -> std::size_t
{
    if (!a)
        return 0;
    return visit(
        Overloads{
            [](CountableIndex const& c) -> std::size_t
            { return hash_mix(1, static_cast<std::size_t>(c.id)); },
            [](ConcreteIndex const& c) -> std::size_t
            { return hash_mix(2, static_cast<std::size_t>(c.value)); },
            [](LabelIndex const& l) -> std::size_t {
                return hash_mix(
                    3, std::hash<std::string_view>{}(l.name.v.view()));
            },
        },
        *a);
}

auto hash_tensor_object(TensorObject const& t) -> std::size_t
{
    std::size_t h = std::hash<std::string_view>{}(t.name.v.view());
    h = hash_mix(h, t.rank ? static_cast<std::size_t>(*t.rank) + 1 : 0);
    for (auto const& s: t.slots)
    {
        h = hash_mix(h, static_cast<std::size_t>(s.slot.level));
        h = hash_mix(h, static_cast<std::size_t>(s.slot.realm));
        h = hash_mix(h, std::hash<IndexSpace const*>{}(s.slot.space));
        h = hash_mix(h, hash_index_assoc(s.index));
    }
    return h;
}

auto hash_factor_seq(std::vector<Factor const*> const& v) -> std::size_t
{
    std::size_t h = v.size();
    for (auto const* f: v)
        h = hash_mix(h, f ? hash(*f) : 0);
    return h;
}

} // namespace

auto hash(Factor const& f) -> std::size_t
{
    std::size_t const tag = f.node.index();
    return visit(
        Overloads{
            [&](Atom const& a) -> std::size_t
            { return hash_mix(tag, hash_tensor_object(a.obj)); },
            [&](Contraction const& c) -> std::size_t
            {
                std::size_t h = hash_factor_seq(c.factors);
                for (auto op: c.ops)
                    h = hash_mix(h, static_cast<std::size_t>(op));
                return hash_mix(tag, h);
            },
            [&](Cross const& c) -> std::size_t
            { return hash_mix(tag, hash_factor_seq(c.factors)); },
            [&](Paren const& p) -> std::size_t
            { return hash_mix(tag, p.body ? hash(*p.body) : 0); },
            [&](Unary const& u) -> std::size_t
            {
                std::size_t h = static_cast<std::size_t>(u.op);
                h = hash_mix(h, u.operand ? hash(*u.operand) : 0);
                return hash_mix(tag, h);
            },
        },
        f);
}

auto hash(Term const& t) -> std::size_t
{
    std::size_t h = static_cast<std::size_t>(t.coeff.num());
    h = hash_mix(h, static_cast<std::size_t>(t.coeff.den()));
    for (auto const& b: t.bound)
    {
        h = hash_mix(h, static_cast<std::size_t>(b.index.id));
        h = hash_mix(h, static_cast<std::size_t>(b.mode));
    }
    h = hash_mix(h, hash_factor_seq(t.scalars));
    h = hash_mix(h, hash_factor_seq(t.tensors));
    return h;
}

auto hash(Nf const& n) -> std::size_t
{
    std::size_t h = n.terms.size();
    for (auto const& t: n.terms)
        h = hash_mix(h, hash(t));
    return h;
}

} // namespace tender::nf
