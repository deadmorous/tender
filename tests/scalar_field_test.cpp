// Scalar fields (vibe 000069 M1): coordinate variables, elementary functions,
// powers — construction, structural equality, rank/component classification,
// rendering, and the canonicalize (Expr → Nf → Expr) round trip.

#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/render.hpp>

#include <gtest/gtest.h>

using namespace tender;

namespace
{

auto latex(Expr const* e) -> std::string
{
    IndexNameMap map;
    return render_latex(*e, map);
}

// r cos(φ): a coordinate, a scalar function, and a scalar product.
auto polar_x(Context& ctx) -> Expr const*
{
    auto* r = make_coordinate(ctx, make_tensor_name("r"), 0, 0);
    auto* phi = make_coordinate(ctx, make_tensor_name("\\varphi"), 0, 1);
    auto* cos_phi = make_scalar_fn(ctx, ScalarFnKind::Cos, phi);
    return make_tensor_product(ctx, r, cos_phi);
}

} // namespace

TEST(ScalarField, CoordinateIsRankZeroComponentValued)
{
    Context ctx;
    auto* r = make_coordinate(ctx, make_tensor_name("r"), 3, 1);
    EXPECT_EQ(infer_rank(r), std::optional<int>{0});
    EXPECT_TRUE(is_component_valued(r));

    auto const* t = std::get_if<TensorObject>(&r->node);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(t->traits.has_value());
    ASSERT_TRUE(t->traits->coordinate.has_value());
    EXPECT_EQ(t->traits->coordinate->chart_id, 3);
    EXPECT_EQ(t->traits->coordinate->slot, 1);
}

TEST(ScalarField, ScalarFnAndPowAreRankZero)
{
    Context ctx;
    auto* phi = make_coordinate(ctx, make_tensor_name("\\varphi"));
    auto* s = make_scalar_fn(ctx, ScalarFnKind::Sin, phi);
    auto* p = make_pow(ctx, s, make_scalar(ctx, Rational{2}));
    EXPECT_EQ(infer_rank(s), std::optional<int>{0});
    EXPECT_EQ(infer_rank(p), std::optional<int>{0});
    EXPECT_TRUE(is_component_valued(s));
    EXPECT_TRUE(is_component_valued(p));
}

TEST(ScalarField, StructuralEquality)
{
    Context ctx;
    auto* phi1 = make_coordinate(ctx, make_tensor_name("\\varphi"));
    auto* phi2 = make_coordinate(ctx, make_tensor_name("\\varphi"));
    auto* sin1 = make_scalar_fn(ctx, ScalarFnKind::Sin, phi1);
    auto* sin2 = make_scalar_fn(ctx, ScalarFnKind::Sin, phi2);
    auto* cos1 = make_scalar_fn(ctx, ScalarFnKind::Cos, phi1);
    EXPECT_TRUE(structural_eq(sin1, sin2));
    EXPECT_FALSE(structural_eq(sin1, cos1));

    auto* two = make_scalar(ctx, Rational{2});
    auto* three = make_scalar(ctx, Rational{3});
    EXPECT_TRUE(
        structural_eq(make_pow(ctx, sin1, two), make_pow(ctx, sin2, two)));
    EXPECT_FALSE(
        structural_eq(make_pow(ctx, sin1, two), make_pow(ctx, sin1, three)));
}

TEST(ScalarField, Rendering)
{
    Context ctx;
    auto* r = make_coordinate(ctx, make_tensor_name("r"));
    EXPECT_EQ(latex(r), "r");

    auto* phi = make_coordinate(ctx, make_tensor_name("\\varphi"));
    auto* cos_phi = make_scalar_fn(ctx, ScalarFnKind::Cos, phi);
    EXPECT_EQ(latex(cos_phi), "\\cos\\left(\\varphi\\right)");

    auto* sqrt = make_scalar_fn(
        ctx,
        ScalarFnKind::Sqrt,
        make_pow(ctx, r, make_scalar(ctx, Rational{2})));
    EXPECT_EQ(latex(sqrt), "\\sqrt{r^{2}}");

    // r cos(φ): two scalar *fields* juxtapose (\, ); \cdot is reserved for
    // purely-numeric products.
    EXPECT_EQ(latex(polar_x(ctx)), "r \\, \\cos\\left(\\varphi\\right)");
}

TEST(ScalarField, CanonicalizeRoundTrip)
{
    Context ctx;
    // r cos φ canonicalizes (Expr → Nf → Expr) without error and renders the
    // same up to the canonical scalar ordering (a single term, two scalars).
    auto* x = polar_x(ctx);
    auto* c = steps::canonicalize(ctx, x);
    EXPECT_TRUE(algebraic_eq(ctx, x, c));

    // cos² φ + sin² φ survives as a two-term sum of powers (the simplifier that
    // folds it to 1 is M3 — here we only check the field algebra round-trips).
    auto* phi = make_coordinate(ctx, make_tensor_name("\\varphi"));
    auto* two = make_scalar(ctx, Rational{2});
    auto* cos2 =
        make_pow(ctx, make_scalar_fn(ctx, ScalarFnKind::Cos, phi), two);
    auto* sin2 =
        make_pow(ctx, make_scalar_fn(ctx, ScalarFnKind::Sin, phi), two);
    auto* sum = make_sum(ctx, cos2, sin2);
    auto* cs = steps::canonicalize(ctx, sum);
    EXPECT_TRUE(algebraic_eq(ctx, sum, cs));
    // Idempotent: canonicalizing again is a fixed point.
    EXPECT_TRUE(structural_eq(cs, steps::canonicalize(ctx, cs)));
}

TEST(ScalarField, ScalarProductFoldsLikeTerms)
{
    Context ctx;
    // 2·r·cos φ + r·cos φ collects to 3·r·cos φ through the Nf coefficient
    // fold, proving scalar fields participate in like-term collection.
    auto* term = polar_x(ctx);
    auto* two_term =
        make_tensor_product(ctx, make_scalar(ctx, Rational{2}), term);
    auto* sum = make_sum(ctx, two_term, term);
    auto* three_term =
        make_tensor_product(ctx, make_scalar(ctx, Rational{3}), term);
    EXPECT_TRUE(algebraic_eq(ctx, sum, three_term));
}
