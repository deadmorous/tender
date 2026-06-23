#include <tender/nf_lower.hpp>

#include <variant>

namespace tender::nf
{

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

} // namespace tender::nf
