// Coordinate charts (vibe 000069 M4): derive the orthogonal-curvilinear
// geometry — radius vector, tangent (holonomic) basis g_i, metric g_ij, scale
// factors h_i, and the physical orthonormal frame e_i — from a coordinate
// mapping, and validate the derived frame against the hand-written
// coord_system bases.

#include <tender/chart.hpp>
#include <tender/coord_system.hpp>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>

#include <gtest/gtest.h>

using namespace tender;

namespace
{

auto eq(Context& ctx, Expr const* a, Expr const* b) -> bool
{
    return algebraic_eq(ctx, a, b);
}

auto frame(Context& ctx, char const* name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 1);
}

// 2D Cartesian reference frame i, j (the embedding target for polar).
auto cartesian_2d(Context& ctx) -> Basis
{
    return make_orthonormal_basis(
        ctx,
        space_2d(),
        {frame(ctx, "i"), frame(ctx, "j")},
        make_tensor_name("e"),
        Handedness::Right,
        BasisNaming{
            .value_names = {make_index_name("x"), make_index_name("y")},
            .vector_symbols = {make_tensor_name("i"), make_tensor_name("j")}});
}

auto cos_(Context& ctx, Expr const* u) -> Expr const*
{
    return make_scalar_fn(ctx, ScalarFnKind::Cos, u);
}
auto sin_(Context& ctx, Expr const* u) -> Expr const*
{
    return make_scalar_fn(ctx, ScalarFnKind::Sin, u);
}
auto mul(Context& ctx, Expr const* a, Expr const* b) -> Expr const*
{
    return make_tensor_product(ctx, a, b);
}

// The polar chart: x = r cos θ, y = r sin θ over the 2D Cartesian frame.
struct Polar final
{
    Basis ref;
    Expr const* r;
    Expr const* th;
    CoordinateChart chart;
};

auto make_polar(Context& ctx) -> Polar
{
    auto ref = cartesian_2d(ctx);
    auto* r =
        make_coordinate(ctx, make_tensor_name("r"), 1, 0, /*nonneg=*/true);
    auto* th = make_coordinate(ctx, make_tensor_name("\\theta"), 1, 1);
    auto* x = mul(ctx, r, cos_(ctx, th));
    auto* y = mul(ctx, r, sin_(ctx, th));
    return Polar{ref, r, th, CoordinateChart{ref, {r, th}, {x, y}}};
}

} // namespace

// Step 3: the tangent (holonomic) basis g_i = ∂R/∂q^i.
TEST(Chart, PolarTangentVectors)
{
    Context ctx;
    auto p = make_polar(ctx);
    auto* i = p.ref.basis(0);
    auto* j = p.ref.basis(1);

    // g_r = cos θ i + sin θ j
    auto* gr = tangent_vector(ctx, p.chart, 0);
    auto* gr_exp = make_sum(
        ctx, mul(ctx, cos_(ctx, p.th), i), mul(ctx, sin_(ctx, p.th), j));
    EXPECT_TRUE(eq(ctx, gr, gr_exp));

    // g_θ = −r sin θ i + r cos θ j
    auto* gt = tangent_vector(ctx, p.chart, 1);
    auto* gt_exp = make_sum(
        ctx,
        make_negate(ctx, mul(ctx, mul(ctx, p.r, sin_(ctx, p.th)), i)),
        mul(ctx, mul(ctx, p.r, cos_(ctx, p.th)), j));
    EXPECT_TRUE(eq(ctx, gt, gt_exp));
}

// Step 4: metric g_rr = 1, g_θθ = r², g_rθ = 0 (needs cos²+sin² = 1).
TEST(Chart, PolarMetric)
{
    Context ctx;
    auto p = make_polar(ctx);
    EXPECT_TRUE(
        eq(ctx,
           metric_component(ctx, p.chart, 0, 0),
           make_scalar(ctx, Rational{1})));
    EXPECT_TRUE(
        eq(ctx,
           metric_component(ctx, p.chart, 1, 1),
           make_pow(ctx, p.r, make_scalar(ctx, Rational{2}))));
    EXPECT_TRUE(
        eq(ctx,
           metric_component(ctx, p.chart, 0, 1),
           make_scalar(ctx, Rational{0})));
}

// Step 4: scale factors h_r = 1, h_θ = √(r²) = r (needs r ≥ 0).
TEST(Chart, PolarScaleFactors)
{
    Context ctx;
    auto p = make_polar(ctx);
    EXPECT_TRUE(
        eq(ctx, scale_factor(ctx, p.chart, 0), make_scalar(ctx, Rational{1})));
    EXPECT_TRUE(eq(ctx, scale_factor(ctx, p.chart, 1), p.r));
}

// Step 5: physical frame e_r = cos θ i + sin θ j, e_θ = −sin θ i + cos θ j.
TEST(Chart, PolarPhysicalBasis)
{
    Context ctx;
    auto p = make_polar(ctx);
    auto frame_basis = physical_basis(ctx, p.chart);
    auto* i = p.ref.basis(0);
    auto* j = p.ref.basis(1);

    EXPECT_EQ(frame_basis.dim(), 2);
    EXPECT_TRUE(frame_basis.is_orthonormal());

    auto* er = frame_basis.basis(0);
    auto* er_exp = make_sum(
        ctx, mul(ctx, cos_(ctx, p.th), i), mul(ctx, sin_(ctx, p.th), j));
    EXPECT_TRUE(eq(ctx, er, er_exp));

    auto* et = frame_basis.basis(1);
    auto* et_exp = make_sum(
        ctx,
        make_negate(ctx, mul(ctx, sin_(ctx, p.th), i)),
        mul(ctx, cos_(ctx, p.th), j));
    EXPECT_TRUE(eq(ctx, et, et_exp));
}

// The derived polar frame has the same shape (dim, space, orthonormality, and
// coordinate value names) as the hand-written polar_2d coord_system frame.
TEST(Chart, PolarMatchesHandWritten)
{
    Context ctx;
    auto p = make_polar(ctx);
    auto derived = physical_basis(ctx, p.chart);
    auto hand = polar_2d(ctx);

    EXPECT_EQ(derived.dim(), hand.dim());
    EXPECT_EQ(derived.space(), hand.space());
    EXPECT_EQ(derived.is_orthonormal(), hand.is_orthonormal());
    for (int v: derived.space()->values())
        EXPECT_EQ(derived.value_name(v), hand.value_name(v));
}

// Cylindrical: x = r cos θ, y = r sin θ, z = z over WCS.  Metric (1, r², 1),
// physical frame e_r, e_θ in the plane and e_z = k.
TEST(Chart, Cylindrical)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* r = make_coordinate(ctx, make_tensor_name("r"), 2, 0, true);
    auto* th = make_coordinate(ctx, make_tensor_name("\\theta"), 2, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 2, 2);
    CoordinateChart chart{
        ref,
        {r, th, z},
        {mul(ctx, r, cos_(ctx, th)), mul(ctx, r, sin_(ctx, th)), z}};

    EXPECT_TRUE(
        eq(ctx,
           metric_component(ctx, chart, 0, 0),
           make_scalar(ctx, Rational{1})));
    EXPECT_TRUE(
        eq(ctx,
           metric_component(ctx, chart, 1, 1),
           make_pow(ctx, r, make_scalar(ctx, Rational{2}))));
    EXPECT_TRUE(
        eq(ctx,
           metric_component(ctx, chart, 2, 2),
           make_scalar(ctx, Rational{1})));
    EXPECT_TRUE(eq(ctx, scale_factor(ctx, chart, 1), r));

    auto fb = physical_basis(ctx, chart);
    EXPECT_EQ(fb.dim(), 3);
    // e_z = k (the constant reference vector survives unchanged).
    EXPECT_TRUE(eq(ctx, fb.basis(2), ref.basis(2)));
    // e_θ = −sin θ i + cos θ j.
    auto* et_exp = make_sum(
        ctx,
        make_negate(ctx, mul(ctx, sin_(ctx, th), ref.basis(0))),
        mul(ctx, cos_(ctx, th), ref.basis(1)));
    EXPECT_TRUE(eq(ctx, fb.basis(1), et_exp));
}

// Spherical: x = r sinθ cosφ, y = r sinθ sinφ, z = r cosθ.  The φ row needs the
// two-trig-square Pythagorean fold (g_φφ = r² sin²θ) and the product square
// root (h_φ = r sin θ).
TEST(Chart, SphericalAzimuthRow)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* r = make_coordinate(ctx, make_tensor_name("r"), 3, 0, true);
    auto* th = make_coordinate(ctx, make_tensor_name("\\theta"), 3, 1);
    auto* ph = make_coordinate(ctx, make_tensor_name("\\phi"), 3, 2);
    CoordinateChart chart{
        ref,
        {r, th, ph},
        {mul(ctx, mul(ctx, r, sin_(ctx, th)), cos_(ctx, ph)),
         mul(ctx, mul(ctx, r, sin_(ctx, th)), sin_(ctx, ph)),
         mul(ctx, r, cos_(ctx, th))}};

    // g_φφ = r² sin²θ.
    auto* gpp = metric_component(ctx, chart, 2, 2);
    auto* gpp_exp =
        mul(ctx,
            make_pow(ctx, r, make_scalar(ctx, Rational{2})),
            make_pow(ctx, sin_(ctx, th), make_scalar(ctx, Rational{2})));
    EXPECT_TRUE(eq(ctx, gpp, gpp_exp));

    // h_φ = r sin θ.
    EXPECT_TRUE(
        eq(ctx, scale_factor(ctx, chart, 2), mul(ctx, r, sin_(ctx, th))));

    // e_φ = −sin φ i + cos φ j.
    auto fb = physical_basis(ctx, chart);
    auto* ep_exp = make_sum(
        ctx,
        make_negate(ctx, mul(ctx, sin_(ctx, ph), ref.basis(0))),
        mul(ctx, cos_(ctx, ph), ref.basis(1)));
    EXPECT_TRUE(eq(ctx, fb.basis(2), ep_exp));
}
