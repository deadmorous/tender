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

// A bare rank-1 vector (the only operand cross anticommutation applies to).
auto is_rank1_vector(Expr const* e) -> bool
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    return t && t->rank && *t->rank == 1 && t->slots.empty();
}

// Re-associate a cross around a rank-≥2 fence: `(x×M)×z → x×(M×z)` when M is
// rank ≥ 2 (the ⊗ inside M fences the two crosses onto disjoint legs, so the
// bracketing is immaterial — 000055).  Returns the re-associated `Expr`, or
// nullptr when the pattern does not apply.  Mirrors derivation.cpp's helper of
// the same name (kept local to avoid disturbing that translation unit).
auto reassociate_cross_fence(Context& ctx, Expr const* l, Expr const* r)
    -> Expr const*
{
    // `Cross` / `make_cross` here are the Expr-level ones (tender::), not the
    // Nf factor of the same name in this namespace.
    auto const* inner = std::get_if<tender::Cross>(&l->node);
    if (!inner)
        return nullptr;
    auto const rx = infer_rank(inner->left);
    auto const rm = infer_rank(inner->right);
    auto const rz = infer_rank(r);
    if (rx == std::optional<int>{1} && rm && *rm >= 2
        && rz == std::optional<int>{1})
        return tender::make_cross(
            ctx, inner->left, tender::make_cross(ctx, inner->right, r));
    return nullptr;
}

} // namespace

auto encapsulate(Context& ctx, Expr const* factor) -> SignedFactor
{
    if (auto const* t = std::get_if<TensorObject>(&factor->node))
        return {+1, make_atom(ctx, *t)};

    if (contraction_op(factor))
    {
        std::vector<Expr const*> operands;
        std::vector<COp> ops;
        flatten_contraction(factor, operands, ops);
        std::vector<Factor const*> encapsulated;
        encapsulated.reserve(operands.size());
        int sign = +1;
        for (auto const* o: operands)
        {
            auto sf = encapsulate(ctx, o);
            sign *= sf.sign;
            encapsulated.push_back(sf.factor);
        }
        return {
            sign,
            make_contraction(ctx, std::move(encapsulated), std::move(ops))};
    }

    // Unary invariants: linear, so the operand's lifted sign passes through.
    if (auto const* u = std::get_if<Trace>(&factor->node))
    {
        auto sf = encapsulate(ctx, u->operand);
        return {sf.sign, make_unary(ctx, UnaryOp::Trace, sf.factor)};
    }
    if (auto const* u = std::get_if<VectorInvariant>(&factor->node))
    {
        auto sf = encapsulate(ctx, u->operand);
        return {sf.sign, make_unary(ctx, UnaryOp::VectorInvariant, sf.factor)};
    }
    if (auto const* u = std::get_if<Transpose>(&factor->node))
    {
        auto sf = encapsulate(ctx, u->operand);
        return {sf.sign, make_unary(ctx, UnaryOp::Transpose, sf.factor)};
    }

    if (auto const* c = std::get_if<tender::Cross>(&factor->node))
    {
        // Anticommutation: a rank-1 pair is ordered canonically, lifting the
        // sign `a×b = -(b×a)`.  Mirrors the canon Cross arm.
        if (is_rank1_vector(c->left) && is_rank1_vector(c->right))
        {
            auto sl = encapsulate(ctx, c->left);
            auto sr = encapsulate(ctx, c->right);
            int sign = sl.sign * sr.sign;
            if (compare(*sl.factor, *sr.factor) > 0)
                return {-sign, make_cross(ctx, {sr.factor, sl.factor})};
            return {sign, make_cross(ctx, {sl.factor, sr.factor})};
        }
        // Rank-≥2 fence: re-associate, then encapsulate the result.
        if (auto const* ra = reassociate_cross_fence(ctx, c->left, c->right))
            return encapsulate(ctx, ra);
        // General binary cross (e.g. a nested cross operand): structural.
        auto sl = encapsulate(ctx, c->left);
        auto sr = encapsulate(ctx, c->right);
        return {sl.sign * sr.sign, make_cross(ctx, {sl.factor, sr.factor})};
    }

    throw std::invalid_argument(
        "encapsulate: unsupported factor node (sums → Paren and nested ⊗ await "
        "the recursive lower / fence distribution)");
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
        auto enc = encapsulate(ctx, f);
        t.coeff *= Rational{enc.sign}; // lift anticommutation sign into coeff
        if (*rank == 0)
            t.scalars.push_back(enc.factor);
        else
            t.tensors.push_back(enc.factor);
    }
    return t;
}

// ---- per-term lowering (passes 3+4) ------------------------------------

auto lower_term(Context& ctx, SignedExpr const& term) -> Term
{
    // Push contractions through ⊗ fences (never over sums), then flatten +
    // place.  distribute_contraction already iterates to a fixpoint.
    auto const* distributed = steps::distribute_contraction(ctx, term.body);
    auto pp = multiplicative_flatten(SignedExpr{term.sign, distributed});
    return place_factors(ctx, pp);
}

} // namespace tender::nf
