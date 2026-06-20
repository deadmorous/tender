// End-to-end feasibility examples for the basis bridge (vibe 000049 §4).
// These exercise the whole Stage-3/4 pipeline as it composes with the existing
// coordinate engine, and are maintained as the system grows (CLAUDE.md #5).

#include <tender/basis.hpp>
#include <tender/coord_system.hpp>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/index_space.hpp>

#include <gtest/gtest.h>

using namespace tender;

namespace
{

auto invariant_vec(Context& ctx, char const* name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 1);
}

// Reduce an invariant expression to its scalar coordinate form in WCS:
// expand each vector, turn basis dots into Kronecker deltas, then evaluate
// the deltas concretely and re-fold the result into a single Einstein sum.
auto reduce_in_wcs(Context& ctx, Basis const& b, Expr const* e) -> Expr const*
{
    e = expand_in_basis(ctx, e, b, Variance::Covariant);
    e = simplify_basis_dot(ctx, e, b);
    e = steps::canonicalize(ctx, e);
    e = steps::unroll_sums(ctx, e);
    e = steps::eval_delta_concrete(ctx, e);
    e = steps::fold_arithmetic(ctx, e);
    e = steps::canonicalize(ctx, e);
    return steps::fold_sums(ctx, e);
}

} // namespace

// I = Σ_i e_i ⊗ e^i, folded straight back to the invariant I — the smallest
// proof that expand and reassemble are inverse for a rank-2 object.
TEST(BasisFeasibility, IdentityRoundTrip)
{
    Context ctx;
    auto b = wcs(ctx);
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);

    auto const* expanded = steps::canonicalize(
        ctx, expand_in_basis(ctx, A, b, Variance::Covariant));
    EXPECT_TRUE(structural_eq(reassemble(ctx, expanded, b), A));
}

// a · b reduces, through the basis, to the scalar coordinate contraction
// Σ_i a_i b_i.
TEST(BasisFeasibility, DotReducesToCoordinateContraction)
{
    Context ctx;
    auto b = wcs(ctx);
    auto const* a = invariant_vec(ctx, "a");
    auto const* c = invariant_vec(ctx, "b");

    auto const* reduced = reduce_in_wcs(ctx, b, make_dot(ctx, a, c));

    // Expected: Σ_i a_i b_i.
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto const* a_i = make_tensor_object(
        ctx,
        make_tensor_name("a"),
        {SlotBinding{
            IndexSlot{Level::Lower, Realm::Orthonormal, space_3d()},
            IndexAssoc{i}}},
        0);
    auto const* b_i = make_tensor_object(
        ctx,
        make_tensor_name("b"),
        {SlotBinding{
            IndexSlot{Level::Lower, Realm::Orthonormal, space_3d()},
            IndexAssoc{i}}},
        0);
    auto const* expected =
        make_explicit_sum(ctx, i, make_tensor_product(ctx, a_i, b_i));
    EXPECT_TRUE(algebraic_eq(ctx, reduced, expected));
}

// a · b = b · a, derived from first principles: both sides reduce through the
// basis to the same scalar coordinate form (the coordinates are scalars and
// commute).
TEST(BasisFeasibility, DotProductCommutes)
{
    Context ctx;
    auto b = wcs(ctx);
    auto const* a = invariant_vec(ctx, "a");
    auto const* c = invariant_vec(ctx, "b");

    auto const* ab = reduce_in_wcs(ctx, b, make_dot(ctx, a, c));
    auto const* ba = reduce_in_wcs(ctx, b, make_dot(ctx, c, a));
    EXPECT_TRUE(algebraic_eq(ctx, ab, ba));
}
