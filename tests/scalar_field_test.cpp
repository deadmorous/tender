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

// ---- partial differentiation (vibe 000069 M2) --------------------------

namespace
{

auto coord(Context& ctx, char const* name, int slot) -> Expr const*
{
    return make_coordinate(ctx, make_tensor_name(name), 0, slot);
}

} // namespace

TEST(Partial, CoordinateAndConstants)
{
    Context ctx;
    auto* r = coord(ctx, "r", 0);
    auto* phi = coord(ctx, "\\varphi", 1);
    // ∂_r r = 1, ∂_r φ = 0 (a different coordinate), ∂_r 5 = 0.
    EXPECT_TRUE(algebraic_eq(
        ctx, steps::partial(ctx, r, r), make_scalar(ctx, Rational{1})));
    EXPECT_TRUE(algebraic_eq(
        ctx, steps::partial(ctx, phi, r), make_scalar(ctx, Rational{0})));
    EXPECT_TRUE(algebraic_eq(
        ctx,
        steps::partial(ctx, make_scalar(ctx, Rational{5}), r),
        make_scalar(ctx, Rational{0})));
    // ∂_r of a constant reference vector i = 0.
    auto* i = make_tensor_object(ctx, make_tensor_name("i"), {}, 1);
    EXPECT_TRUE(algebraic_eq(
        ctx, steps::partial(ctx, i, r), make_scalar(ctx, Rational{0})));
}

TEST(Partial, RejectsNonCoordinate)
{
    Context ctx;
    auto* r = coord(ctx, "r", 0);
    auto* plain = make_tensor_object(ctx, make_tensor_name("a"), {}, 0);
    EXPECT_THROW(steps::partial(ctx, r, plain), std::invalid_argument);
}

TEST(Partial, ChainRuleOnElementaryFunctions)
{
    Context ctx;
    auto* phi = coord(ctx, "\\varphi", 0);
    // ∂_φ sin φ = cos φ;  ∂_φ cos φ = −sin φ.
    EXPECT_TRUE(algebraic_eq(
        ctx,
        steps::partial(ctx, make_scalar_fn(ctx, ScalarFnKind::Sin, phi), phi),
        make_scalar_fn(ctx, ScalarFnKind::Cos, phi)));
    EXPECT_TRUE(algebraic_eq(
        ctx,
        steps::partial(ctx, make_scalar_fn(ctx, ScalarFnKind::Cos, phi), phi),
        make_negate(ctx, make_scalar_fn(ctx, ScalarFnKind::Sin, phi))));
}

TEST(Partial, PowerRule)
{
    Context ctx;
    auto* r = coord(ctx, "r", 0);
    // ∂_r r² = 2 r¹ (M2 leaves the r^{2-1} unfolded; M3 folds the exponent).
    auto* r2 = make_pow(ctx, r, make_scalar(ctx, Rational{2}));
    auto* expected = make_tensor_product(
        ctx,
        make_scalar(ctx, Rational{2}),
        make_pow(ctx, r, make_scalar(ctx, Rational{1})));
    EXPECT_TRUE(algebraic_eq(ctx, steps::partial(ctx, r2, r), expected));
}

TEST(Partial, QuotientRule)
{
    Context ctx;
    auto* r = coord(ctx, "r", 0);
    // ∂_r (1/r) = (−1)/r².  The quotient rule emits the sign in the numerator;
    // canonicalize keeps that distinct from −(1/r²) (it does not lift a sign
    // out of a division), so the expected form carries the −1 in the numerator
    // too.
    auto* inv = make_scalar_div(ctx, make_scalar(ctx, Rational{1}), r);
    auto* expected = make_scalar_div(
        ctx,
        make_scalar(ctx, Rational{-1}),
        make_pow(ctx, r, make_scalar(ctx, Rational{2})));
    EXPECT_TRUE(algebraic_eq(ctx, steps::partial(ctx, inv, r), expected));
}

// ---- targeted scalar simplifier (vibe 000069 M3) -----------------------

TEST(SimplifyScalars, Pythagorean)
{
    Context ctx;
    auto* phi = coord(ctx, "\\varphi", 0);
    auto* cos2 = make_pow(
        ctx,
        make_scalar_fn(ctx, ScalarFnKind::Cos, phi),
        make_scalar(ctx, Rational{2}));
    auto* sin2 = make_pow(
        ctx,
        make_scalar_fn(ctx, ScalarFnKind::Sin, phi),
        make_scalar(ctx, Rational{2}));
    EXPECT_TRUE(algebraic_eq(
        ctx,
        steps::simplify_scalars(ctx, make_sum(ctx, cos2, sin2)),
        make_scalar(ctx, Rational{1})));
}

TEST(SimplifyScalars, PythagoreanWithCommonFactor)
{
    Context ctx;
    // r² sin²φ + r² cos²φ → r²  (the g_φφ metric component).
    auto* r =
        make_coordinate(ctx, make_tensor_name("r"), 0, 0, /*nonneg=*/true);
    auto* phi = coord(ctx, "\\varphi", 1);
    auto* r2 = make_pow(ctx, r, make_scalar(ctx, Rational{2}));
    auto* two = make_scalar(ctx, Rational{2});
    auto* term_s = make_tensor_product(
        ctx,
        r2,
        make_pow(ctx, make_scalar_fn(ctx, ScalarFnKind::Sin, phi), two));
    auto* term_c = make_tensor_product(
        ctx,
        r2,
        make_pow(ctx, make_scalar_fn(ctx, ScalarFnKind::Cos, phi), two));
    EXPECT_TRUE(algebraic_eq(
        ctx, steps::simplify_scalars(ctx, make_sum(ctx, term_s, term_c)), r2));
}

TEST(SimplifyScalars, RootOfSquareNeedsNonneg)
{
    Context ctx;
    auto* r =
        make_coordinate(ctx, make_tensor_name("r"), 0, 0, /*nonneg=*/true);
    auto* s =
        make_coordinate(ctx, make_tensor_name("s"), 0, 0, /*nonneg=*/false);
    auto* sqrt_r2 = make_scalar_fn(
        ctx,
        ScalarFnKind::Sqrt,
        make_pow(ctx, r, make_scalar(ctx, Rational{2})));
    auto* sqrt_s2 = make_scalar_fn(
        ctx,
        ScalarFnKind::Sqrt,
        make_pow(ctx, s, make_scalar(ctx, Rational{2})));
    // √(r²) → r because r ≥ 0; √(s²) stays (s sign unknown).
    EXPECT_TRUE(algebraic_eq(ctx, steps::simplify_scalars(ctx, sqrt_r2), r));
    EXPECT_TRUE(
        algebraic_eq(ctx, steps::simplify_scalars(ctx, sqrt_s2), sqrt_s2));
}

TEST(SimplifyScalars, PowerCleanup)
{
    Context ctx;
    auto* r = coord(ctx, "r", 0);
    EXPECT_TRUE(algebraic_eq(
        ctx,
        steps::simplify_scalars(
            ctx, make_pow(ctx, r, make_scalar(ctx, Rational{1}))),
        r));
    EXPECT_TRUE(algebraic_eq(
        ctx,
        steps::simplify_scalars(
            ctx, make_pow(ctx, r, make_scalar(ctx, Rational{0}))),
        make_scalar(ctx, Rational{1})));
}

TEST(SimplifyScalars, ScaleFactorThroughRoot)
{
    Context ctx;
    // h_φ = √(r² sin²φ + r² cos²φ) → √(r²) → r  (the polar scale factor).
    auto* r =
        make_coordinate(ctx, make_tensor_name("r"), 0, 0, /*nonneg=*/true);
    auto* phi = coord(ctx, "\\varphi", 1);
    auto* r2 = make_pow(ctx, r, make_scalar(ctx, Rational{2}));
    auto* two = make_scalar(ctx, Rational{2});
    auto* g_phiphi = make_sum(
        ctx,
        make_tensor_product(
            ctx,
            r2,
            make_pow(ctx, make_scalar_fn(ctx, ScalarFnKind::Sin, phi), two)),
        make_tensor_product(
            ctx,
            r2,
            make_pow(ctx, make_scalar_fn(ctx, ScalarFnKind::Cos, phi), two)));
    auto* h_phi = make_scalar_fn(ctx, ScalarFnKind::Sqrt, g_phiphi);
    EXPECT_TRUE(algebraic_eq(ctx, steps::simplify_scalars(ctx, h_phi), r));
}

TEST(Partial, PolarTangentVectors)
{
    Context ctx;
    auto* r = coord(ctx, "r", 0);
    auto* phi = coord(ctx, "\\varphi", 1);
    auto* i = make_tensor_object(ctx, make_tensor_name("i"), {}, 1);
    auto* j = make_tensor_object(ctx, make_tensor_name("j"), {}, 1);
    auto* cos = make_scalar_fn(ctx, ScalarFnKind::Cos, phi);
    auto* sin = make_scalar_fn(ctx, ScalarFnKind::Sin, phi);

    // R = r cos φ · i + r sin φ · j.
    auto* x = make_tensor_product(ctx, make_tensor_product(ctx, r, cos), i);
    auto* y = make_tensor_product(ctx, make_tensor_product(ctx, r, sin), j);
    auto* R = make_sum(ctx, x, y);

    // g_r = ∂_r R = cos φ · i + sin φ · j.
    auto* g_r = make_sum(
        ctx,
        make_tensor_product(ctx, cos, i),
        make_tensor_product(ctx, sin, j));
    EXPECT_TRUE(algebraic_eq(ctx, steps::partial(ctx, R, r), g_r));

    // g_φ = ∂_φ R = −r sin φ · i + r cos φ · j.
    auto* g_phi = make_sum(
        ctx,
        make_negate(
            ctx, make_tensor_product(ctx, make_tensor_product(ctx, r, sin), i)),
        make_tensor_product(ctx, make_tensor_product(ctx, r, cos), j));
    EXPECT_TRUE(algebraic_eq(ctx, steps::partial(ctx, R, phi), g_phi));
}
