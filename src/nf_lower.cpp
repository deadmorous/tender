#include <tender/nf_lower.hpp>

#include <tender/derivation.hpp> // infer_rank

#include <mpk/mix/util/overloads.hpp>

#include <optional>
#include <stdexcept>
#include <variant>

namespace tender::nf
{

using mpk::mix::Overloads;

// ---- pass 2: additive flatten (C3) -------------------------------------

namespace
{

void flatten(Expr const* e, int sign, std::vector<SignedExpr>& out)
{
    if (auto const* s = std::get_if<Sum>(&e->node))
    {
        flatten(s->left, sign, out);
        flatten(s->right, sign, out);
    }
    else if (auto const* d = std::get_if<Difference>(&e->node))
    {
        flatten(d->left, sign, out);
        flatten(d->right, -sign, out);
    }
    else if (auto const* n = std::get_if<Negate>(&e->node))
    {
        flatten(n->operand, -sign, out);
    }
    else
    {
        // Any non-additive node (including a product whose interior contains a
        // sum) is an opaque leaf — no distribution.
        out.push_back({sign, e});
    }
}

} // namespace

auto additive_flatten(Expr const* e) -> std::vector<SignedExpr>
{
    std::vector<SignedExpr> out;
    flatten(e, +1, out);
    return out;
}

// ---- pass 3a: multiplicative flatten (C4) ------------------------------

namespace
{

void flatten_product(Expr const* e, ProductParts& out)
{
    if (auto const* p = std::get_if<TensorProduct>(&e->node))
    {
        flatten_product(p->left, out);
        flatten_product(p->right, out);
    }
    else if (auto const* sl = std::get_if<ScalarLiteral>(&e->node))
    {
        out.coeff *= sl->value;
    }
    else if (auto const* n = std::get_if<Negate>(&e->node))
    {
        out.coeff *= Rational{-1};
        flatten_product(n->operand, out);
    }
    else if (auto const* d = std::get_if<ScalarDiv>(&e->node);
             d != nullptr
             && std::holds_alternative<ScalarLiteral>(d->right->node))
    {
        out.coeff /= std::get<ScalarLiteral>(d->right->node).value;
        flatten_product(d->left, out);
    }
    else
    {
        // A contraction / cross / sum / non-numeric division node is one
        // opaque factor; encapsulation into an Nf Factor is C5/C6.
        out.factors.push_back(e);
    }
}

} // namespace

auto multiplicative_flatten(SignedExpr const& term) -> ProductParts
{
    ProductParts out{.coeff = Rational{term.sign}, .factors = {}};
    flatten_product(term.body, out);
    return out;
}

// ---- pass 3b: factor encapsulation (C5) --------------------------------

namespace
{

// The contraction-family operator of `e`, if it is one.
auto contraction_op(Expr const* e) -> std::optional<COp>
{
    if (std::holds_alternative<Dot>(e->node))
        return COp::Dot;
    if (std::holds_alternative<DDot>(e->node))
        return COp::DDot;
    if (std::holds_alternative<DDotAlt>(e->node))
        return COp::DDotAlt;
    return std::nullopt;
}

auto binop_operands(Expr const* e) -> std::pair<Expr const*, Expr const*>
{
    return visit(
        Overloads{
            [](Dot const& d) { return std::pair{d.left, d.right}; },
            [](DDot const& d) { return std::pair{d.left, d.right}; },
            [](DDotAlt const& d) { return std::pair{d.left, d.right}; },
            [](auto const&) -> std::pair<Expr const*, Expr const*>
            { throw std::logic_error("binop_operands: not a contraction"); }},
        *e);
}

// Flatten a contraction tree into operand / op sequences, dropping bracketing:
// `o(l, r)` becomes flatten(l) ++ [o] ++ flatten(r), so the result holds
// `operands.size() == ops.size() + 1` for any nesting (000057 interface
// theorem).
void flatten_contraction(
    Expr const* e, std::vector<Expr const*>& operands, std::vector<COp>& ops)
{
    auto op = contraction_op(e);
    if (!op)
    {
        operands.push_back(e);
        return;
    }
    auto [l, r] = binop_operands(e);
    flatten_contraction(l, operands, ops);
    ops.push_back(*op);
    flatten_contraction(r, operands, ops);
}

} // namespace

auto encapsulate(Context& ctx, Expr const* factor) -> Factor const*
{
    if (auto const* t = std::get_if<TensorObject>(&factor->node))
        return make_atom(ctx, *t);

    if (contraction_op(factor))
    {
        std::vector<Expr const*> operands;
        std::vector<COp> ops;
        flatten_contraction(factor, operands, ops);
        std::vector<Factor const*> encapsulated;
        encapsulated.reserve(operands.size());
        for (auto const* o: operands)
            encapsulated.push_back(encapsulate(ctx, o));
        return make_contraction(ctx, std::move(encapsulated), std::move(ops));
    }

    throw std::invalid_argument(
        "encapsulate: unsupported factor node (Cross is C6; sums / nested "
        "products / unary invariants await the recursive lower)");
}

// ---- pass 4: region placement (C5) -------------------------------------

auto place_factors(Context& ctx, ProductParts const& pp) -> Term
{
    Term t{.coeff = pp.coeff};
    for (auto const* f: pp.factors)
    {
        auto rank = infer_rank(f);
        if (!rank)
            throw std::invalid_argument(
                "place_factors: factor rank is unknown (region placement needs "
                "a trustworthy infer_rank)");
        auto const* enc = encapsulate(ctx, f);
        if (*rank == 0)
            t.scalars.push_back(enc);
        else
            t.tensors.push_back(enc);
    }
    return t;
}

} // namespace tender::nf
