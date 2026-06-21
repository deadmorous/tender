// End-to-end feasibility examples for the basis bridge (vibe 000049 §4).
// These exercise the whole Stage-3/4 pipeline as it composes with the existing
// coordinate engine, and are maintained as the system grows (CLAUDE.md #5).

#include <tender/basis.hpp>
#include <tender/coord_system.hpp>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/identity.hpp>
#include <tender/index_space.hpp>

#include <gtest/gtest.h>

#include <vector>

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

// A rank-2 invariant expands to Σ_i Σ_j A_{ij} e_i e_j and folds straight back
// — the smallest proof that expand and reassemble are inverse beyond rank 1.
TEST(BasisFeasibility, Rank2RoundTrip)
{
    Context ctx;
    auto b = wcs(ctx);
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);

    auto const* expanded = steps::canonicalize(
        ctx, expand_in_basis(ctx, A, b, Variance::Covariant));
    EXPECT_TRUE(structural_eq(reassemble(ctx, expanded, b), A));
}

// The identity's own resolution I = Σ_i e_i ⊗ e^i, derived from its coordinate
// I^i_j = e^i·I·e_j = e^i·e_j = δ^i_j, makes the round trip exactly.
TEST(BasisFeasibility, IdentityRoundTrip)
{
    Context ctx;
    auto b = wcs(ctx);
    auto const* I = make_identity(ctx);

    auto const* expanded = steps::canonicalize(
        ctx, expand_in_basis(ctx, I, b, Variance::Covariant));
    EXPECT_TRUE(structural_eq(reassemble(ctx, expanded, b), I));
}

// In an OBLIQUE basis the same I = Σ_i e_i ⊗ e^i round trip holds (covariant
// and contravariant vectors are now distinct), and the identity's all-covariant
// coordinate is the metric: I_ij = e_i·I·e_j = e_i·e_j = g_ij.
TEST(BasisFeasibility, ObliqueIdentityRoundTripAndMetric)
{
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* bb = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    auto const* cc = make_tensor_object(ctx, make_tensor_name("c"), {}, 1);
    auto basis = make_oblique_basis(ctx, space_3d(), {a, bb, cc});
    auto const* I = make_identity(ctx);

    // Round trip.
    auto const* expanded = steps::canonicalize(
        ctx, expand_in_basis(ctx, I, basis, Variance::Covariant));
    EXPECT_TRUE(structural_eq(reassemble(ctx, expanded, basis), I));

    // The all-covariant coordinate I_ij = e_i·I·e_j reduces to the metric g_ij.
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* coord = make_dot(
        ctx,
        make_dot(ctx, basis.covariant_vector(ctx, i), I),
        basis.covariant_vector(ctx, j));
    auto const* reduced =
        simplify_basis_dot(ctx, steps::contract_identity(ctx, coord), basis);

    auto const* expected = make_metric(
        ctx,
        Realm::Oblique,
        space_3d(),
        Level::Lower,
        Level::Lower,
        IndexAssoc{j},
        IndexAssoc{i});
    EXPECT_TRUE(algebraic_eq(ctx, reduced, expected));
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

// (e_i × e_j) · e_k = ε_ijk for an orthonormal right-handed basis, derived by
// dotting the cross-product formula with a basis vector:
//   e_i × e_j → ε_ijl e^l,   (… ) · e_k → ε_ijl δ^l_k → ε_ijk.
// The final ε-δ substitution is supplied as a data identity (it is the same
// shape as delta-contraction, which the generic matcher already handles).
TEST(BasisFeasibility, CrossDotIsLeviCivita)
{
    Context ctx;
    auto b = wcs(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto k = CountableIndex{ctx.alloc_index_id()};

    auto const* expr = make_dot(
        ctx,
        make_cross(ctx, b.covariant_vector(ctx, i), b.covariant_vector(ctx, j)),
        b.covariant_vector(ctx, k));
    // Cross → ε e^l, dot → ε δ, then canonicalize.
    auto const* reduced = steps::canonicalize(
        ctx, simplify_basis_dot(ctx, simplify_basis_cross(ctx, expr, b), b));

    // Close with Σ_l ε_{a b l} δ_{l c} = ε_{a b c}.
    std::vector<Level> const lll{Level::Lower, Level::Lower, Level::Lower};
    auto a2 = CountableIndex{ctx.alloc_index_id()};
    auto b2 = CountableIndex{ctx.alloc_index_id()};
    auto c2 = CountableIndex{ctx.alloc_index_id()};
    auto l2 = CountableIndex{ctx.alloc_index_id()};
    auto const* lhs = make_explicit_sum(
        ctx,
        l2,
        make_tensor_product(
            ctx,
            make_levi_civita(
                ctx,
                Realm::Orthonormal,
                space_3d(),
                lll,
                {IndexAssoc{a2}, IndexAssoc{b2}, IndexAssoc{l2}}),
            make_delta(
                ctx,
                Realm::Orthonormal,
                space_3d(),
                Level::Lower,
                Level::Lower,
                IndexAssoc{l2},
                IndexAssoc{c2})));
    auto const* rhs = make_levi_civita(
        ctx,
        Realm::Orthonormal,
        space_3d(),
        lll,
        {IndexAssoc{a2}, IndexAssoc{b2}, IndexAssoc{c2}});
    auto const* result =
        apply_identity(ctx, reduced, Identity{"eps-dot", lhs, rhs});

    auto const* expected = make_levi_civita(
        ctx,
        Realm::Orthonormal,
        space_3d(),
        lll,
        {IndexAssoc{i}, IndexAssoc{j}, IndexAssoc{k}});
    EXPECT_TRUE(algebraic_eq(ctx, result, expected));
}

// a × I = I × a for any vector a, derived through the basis: expand, distribute
// the cross over the identity dyad I = Σ_m e_m ⊗ e^m, reduce each e_i × e_m to
// ε e^k, then canonicalize.  Exercises contraction-over-⊗ distribution, ε
// cyclic symmetry, and nested-ExplicitSum ordering all at once.
TEST(BasisFeasibility, CrossWithIdentityCommutes)
{
    Context ctx;
    auto b = wcs(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* I = make_identity(ctx);
    auto reduce = [&](Expr const* e)
    {
        // simplify_basis_cross distributes the cross over the identity dyad
        // internally, so no explicit distribute_contraction is needed.
        e = expand_in_basis(ctx, e, b, Variance::Covariant);
        e = simplify_basis_cross(ctx, e, b);
        return steps::canonicalize(ctx, e);
    };
    EXPECT_TRUE(algebraic_eq(
        ctx, reduce(make_cross(ctx, a, I)), reduce(make_cross(ctx, I, a))));
}

// vec(I) = 0: the vector invariant of the identity vanishes.  Through the
// basis I = Σ e_i⊗e_i, so vec(I) = Σ e_i × e_i, and each e_i × e_i = 0 by the
// antisymmetry of the cross product (ε with a repeated index).
TEST(BasisFeasibility, VectorInvariantOfIdentityIsZero)
{
    Context ctx;
    auto b = wcs(ctx);
    auto const* e = make_vector_invariant(ctx, make_identity(ctx));
    e = expand_in_basis(ctx, e, b, Variance::Covariant);
    e = steps::expand_dyad_ops(ctx, e);
    e = simplify_basis_cross(ctx, e, b);
    e = steps::unroll_sums(ctx, e);
    e = steps::eval_eps_concrete(ctx, e);
    e = steps::canonicalize(ctx, steps::fold_arithmetic(ctx, e));
    EXPECT_TRUE(algebraic_eq(ctx, e, make_scalar(ctx, Rational{0})));
}
