// Coordinate charts (vibe 000069 M4): derive the orthogonal-curvilinear
// geometry — radius vector, tangent (holonomic) basis g_i, metric g_ij, scale
// factors h_i, and the physical orthonormal frame e_i — from a coordinate
// mapping, and validate the derived frame against the hand-written
// coord_system bases.

#include <tender/chart.hpp>
#include <tender/context.hpp>
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

namespace
{

auto s0(Context& ctx) -> Expr const*
{
    return make_scalar(ctx, Rational{0});
}
auto s1(Context& ctx) -> Expr const*
{
    return make_scalar(ctx, Rational{1});
}

// Compare a connection-coefficient vector against an expected list.
auto coeffs_eq(
    Context& ctx,
    std::vector<Expr const*> const& got,
    std::vector<Expr const*> const& exp) -> bool
{
    if (got.size() != exp.size())
        return false;
    for (std::size_t k = 0; k < got.size(); ++k)
        if (!algebraic_eq(ctx, got[k], exp[k]))
            return false;
    return true;
}

} // namespace

// Step 6: ∂_{q^j} e_i in the reference frame.  ∂_φ e_r = e_φ, ∂_φ e_φ = −e_r,
// and the radial derivatives vanish.
TEST(Chart, PolarBasisDerivative)
{
    Context ctx;
    auto p = make_polar(ctx);
    auto fb = physical_basis(ctx, p.chart);

    EXPECT_TRUE(eq(ctx, basis_derivative(ctx, p.chart, 0, 0), s0(ctx)));
    EXPECT_TRUE(eq(ctx, basis_derivative(ctx, p.chart, 1, 0), s0(ctx)));
    // ∂_φ e_r = e_φ.
    EXPECT_TRUE(eq(ctx, basis_derivative(ctx, p.chart, 0, 1), fb.basis(1)));
    // ∂_φ e_φ = −e_r.
    EXPECT_TRUE(
        eq(ctx,
           basis_derivative(ctx, p.chart, 1, 1),
           make_negate(ctx, fb.basis(0))));
}

// Step 6 as connection (rotation) coefficients γ^k_{ij}: ∂_φ e_r = e_φ gives
// {0, 1}, ∂_φ e_φ = −e_r gives {−1, 0}, the radial rows vanish.
TEST(Chart, PolarConnectionCoefficients)
{
    Context ctx;
    auto p = make_polar(ctx);

    EXPECT_TRUE(coeffs_eq(
        ctx, connection_coefficients(ctx, p.chart, 0, 0), {s0(ctx), s0(ctx)}));
    EXPECT_TRUE(coeffs_eq(
        ctx, connection_coefficients(ctx, p.chart, 1, 0), {s0(ctx), s0(ctx)}));
    EXPECT_TRUE(coeffs_eq(
        ctx, connection_coefficients(ctx, p.chart, 0, 1), {s0(ctx), s1(ctx)}));
    EXPECT_TRUE(coeffs_eq(
        ctx,
        connection_coefficients(ctx, p.chart, 1, 1),
        {make_scalar(ctx, Rational{-1}), s0(ctx)}));
}

// Spherical rotation coefficients (the curvilinear payoff): ∂_φ e_r = sinθ e_φ,
// ∂_φ e_θ = cosθ e_φ, ∂_φ e_φ = −sinθ e_r − cosθ e_θ.
TEST(Chart, SphericalConnectionCoefficients)
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

    // ∂_θ e_r = e_θ.
    EXPECT_TRUE(coeffs_eq(
        ctx,
        connection_coefficients(ctx, chart, 0, 1),
        {s0(ctx), s1(ctx), s0(ctx)}));
    // ∂_φ e_r = sinθ e_φ.
    EXPECT_TRUE(coeffs_eq(
        ctx,
        connection_coefficients(ctx, chart, 0, 2),
        {s0(ctx), s0(ctx), sin_(ctx, th)}));
    // ∂_φ e_θ = cosθ e_φ.
    EXPECT_TRUE(coeffs_eq(
        ctx,
        connection_coefficients(ctx, chart, 1, 2),
        {s0(ctx), s0(ctx), cos_(ctx, th)}));
    // ∂_φ e_φ = −sinθ e_r − cosθ e_θ.
    EXPECT_TRUE(coeffs_eq(
        ctx,
        connection_coefficients(ctx, chart, 2, 2),
        {make_negate(ctx, sin_(ctx, th)),
         make_negate(ctx, cos_(ctx, th)),
         s0(ctx)}));
}

namespace
{

// The cylindrical chart x = r cos θ, y = r sin θ, z = z over WCS.
struct Cyl final
{
    Basis ref;
    Expr const* r;
    Expr const* th;
    Expr const* z;
    CoordinateChart chart;
};

auto make_cyl(Context& ctx) -> Cyl
{
    auto ref = wcs(ctx);
    auto* r = make_coordinate(ctx, make_tensor_name("r"), 2, 0, true);
    auto* th = make_coordinate(ctx, make_tensor_name("\\theta"), 2, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 2, 2);
    return Cyl{
        ref,
        r,
        th,
        z,
        CoordinateChart{
            ref,
            {r, th, z},
            {mul(ctx, r, cos_(ctx, th)), mul(ctx, r, sin_(ctx, th)), z}}};
}

// Equality of two invariant tensors written in the reference frame: distribute
// ⊗ over + (canonicalize alone does not) and simplify, then compare.
auto inv_eq(Context& ctx, Expr const* a, Expr const* b) -> bool
{
    auto* na = steps::simplify_scalars(ctx, steps::expand_products(ctx, a));
    auto* nb = steps::simplify_scalars(ctx, steps::expand_products(ctx, b));
    return algebraic_eq(ctx, na, nb);
}

} // namespace

// The operators are ∇ applied formally and return invariant tensors.  ∇R is the
// identity tensor I = Σ_i e_i ⊗ e_i, div R = 3, rot R = 0 — no component
// bookkeeping, no special-casing.
TEST(Chart, CartesianGradientIsIdentity)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 7, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 7, 2);
    CoordinateChart chart{ref, {x, y, z}, {x, y, z}};
    auto* R = radius_vector(ctx, chart);

    // ∇R = I: the operator now folds the concrete resolution Σ_k e_k⊗e_k back
    // to the identity tensor itself (vibe 000070 P4), so the result is
    // structurally I — not just inv_eq to the expanded dyad sum.
    EXPECT_TRUE(structural_eq(gradient(ctx, chart, R), make_identity(ctx)));
    // The raw (unfolded) form is still available and equals the dyad sum.
    Expr const* dyads = nullptr;
    for (int k = 0; k < 3; ++k)
    {
        auto* dyad = mul(ctx, ref.basis(k), ref.basis(k));
        dyads = dyads ? make_sum(ctx, dyads, dyad) : dyad;
    }
    EXPECT_TRUE(
        inv_eq(ctx, gradient(ctx, chart, R, /*fold_identity=*/false), dyads));
    EXPECT_TRUE(
        eq(ctx, divergence(ctx, chart, R), make_scalar(ctx, Rational{3})));
    EXPECT_TRUE(eq(ctx, rot(ctx, chart, R), s0(ctx)));
}

// The operators reduce a field built with the identity tensor and a cross —
// R×I, the rank-2 skew tensor with (R×I)·a = R×a — instead of crashing (vibe
// 000070 P6).  ∇×(R×I) = I − 3I = −2I.  Previously this raised in Nf lowering.
TEST(Chart, RotOfRCrossIdentity)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 7, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 7, 2);
    CoordinateChart chart{ref, {x, y, z}, {x, y, z}};
    auto* R = radius_vector(ctx, chart);
    auto* RxI = make_cross(ctx, R, make_identity(ctx));

    auto* want = make_negate(
        ctx, mul(ctx, make_scalar(ctx, Rational{2}), make_identity(ctx)));
    EXPECT_TRUE(structural_eq(rot(ctx, chart, RxI), want)); // ∇×(R×I) = −2I
}

// A cross of two constant (non-field) vectors differentiates to zero by Leibniz
// (0×b + a×0); the operator folds that away gracefully rather than crashing on
// the zero-operand crosses (vibe 000070 P6, graceful path).
TEST(Chart, RotOfConstantCrossIsZero)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 7, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 7, 2);
    CoordinateChart chart{ref, {x, y, z}, {x, y, z}};
    auto* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    EXPECT_TRUE(eq(ctx, rot(ctx, chart, make_cross(ctx, a, b)), s0(ctx)));
}

// The public frame dot/cross reduce contractions in the reference frame (vibe
// 000070 P8): i·i = 1, i·j = 0, i×j = k.
TEST(Chart, FrameDotAndCross)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 7, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 7, 2);
    CoordinateChart chart{ref, {x, y, z}, {x, y, z}};

    EXPECT_TRUE(
        eq(ctx,
           frame_dot(ctx, chart, ref.basis(0), ref.basis(0)),
           make_scalar(ctx, Rational{1})));
    EXPECT_TRUE(
        eq(ctx, frame_dot(ctx, chart, ref.basis(0), ref.basis(1)), s0(ctx)));
    EXPECT_TRUE(structural_eq(
        frame_cross(ctx, chart, ref.basis(0), ref.basis(1)), ref.basis(2)));
}

// Both the 3D-only operators reject a 2D chart instead of misbehaving.
TEST(Chart, FrameCrossAndRotRequire3D)
{
    Context ctx;
    auto p = make_polar(ctx); // 2D
    auto* i = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto* j = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    EXPECT_THROW((void)frame_cross(ctx, p.chart, i, j), std::invalid_argument);
    EXPECT_THROW((void)rot(ctx, p.chart, i), std::invalid_argument);
}

// physical_frame registers the connection table so ∂ of a frame-vector atom is
// resolved intrinsically (vibe 000071): ∂_θ e_r = e_θ, ∂_θ e_θ = −e_r,
// ∂_r e_r = 0 — the symbolic e_k atoms, no reference-frame expansion.
TEST(Chart, PhysicalFrameConnectionTable)
{
    Context ctx;
    auto c = make_cyl(ctx); // coords chart_id 2: r (slot 0), θ (slot 1), z (2)
    auto fb = physical_frame(ctx, c.chart);

    auto const* conn = ctx.connection(fb.basis_id());
    ASSERT_NE(conn, nullptr);
    EXPECT_EQ(conn->chart_id, 2);
    EXPECT_EQ(static_cast<int>(conn->deriv.size()), 3);

    auto* e_r = fb.direction(ctx, 0);
    auto* e_th = fb.direction(ctx, 1);
    // deriv[i][j] = ∂_{q^j} e_i.
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, conn->deriv[0][1]),
        steps::canonicalize(ctx, e_th))); // ∂_θ e_r = e_θ
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, conn->deriv[1][1]),
        steps::canonicalize(ctx, make_negate(ctx, e_r)))); // ∂_θ e_θ = −e_r
    EXPECT_TRUE(eq(ctx, conn->deriv[0][0], s0(ctx)));      // ∂_r e_r = 0
}

// ∇f: the 1/h_i factors are the curvilinear content — for cylindrical
// ∇ = e_r ∂_r + (1/r) e_θ ∂_θ + e_z ∂_z, so ∇θ = (1/r) e_θ and ∇r² = 2r e_r,
// each returned as the invariant vector in the reference frame.
TEST(Chart, CylindricalGradient)
{
    Context ctx;
    auto c = make_cyl(ctx);
    auto fb = physical_basis(ctx, c.chart);

    // ∇θ = (1/r) e_θ
    EXPECT_TRUE(inv_eq(
        ctx,
        gradient(ctx, c.chart, c.th),
        mul(ctx, make_scalar_div(ctx, s1(ctx), c.r), fb.basis(1))));
    // ∇r² = 2r e_r
    auto* r2 = make_pow(ctx, c.r, make_scalar(ctx, Rational{2}));
    EXPECT_TRUE(inv_eq(
        ctx,
        gradient(ctx, c.chart, r2),
        mul(ctx, mul(ctx, make_scalar(ctx, Rational{2}), c.r), fb.basis(0))));
}

// div of the radial field v = r e_r is 2 (= (1/r) ∂_r(r·r)).
TEST(Chart, CylindricalDivergence)
{
    Context ctx;
    auto c = make_cyl(ctx);
    auto* v = mul(ctx, c.r, physical_basis(ctx, c.chart).basis(0)); // r e_r
    EXPECT_TRUE(
        eq(ctx, divergence(ctx, c.chart, v), make_scalar(ctx, Rational{2})));
}

// Δr² = 4 in cylindrical (and Δr² = 6 in spherical) — div(grad) composed.
TEST(Chart, CylindricalLaplacian)
{
    Context ctx;
    auto c = make_cyl(ctx);
    auto* r2 = make_pow(ctx, c.r, make_scalar(ctx, Rational{2}));
    EXPECT_TRUE(
        eq(ctx, laplacian(ctx, c.chart, r2), make_scalar(ctx, Rational{4})));
}

TEST(Chart, SphericalLaplacian)
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
    auto* r2 = make_pow(ctx, r, make_scalar(ctx, Rational{2}));
    EXPECT_TRUE(
        eq(ctx, laplacian(ctx, chart, r2), make_scalar(ctx, Rational{6})));
}

// rot of the rigid-rotation field v = r e_θ is the uniform vorticity 2 e_z =
// 2k.
TEST(Chart, CylindricalRot)
{
    Context ctx;
    auto c = make_cyl(ctx);
    auto* v = mul(ctx, c.r, physical_basis(ctx, c.chart).basis(1)); // r e_θ
    EXPECT_TRUE(inv_eq(
        ctx,
        rot(ctx, c.chart, v),
        mul(ctx, make_scalar(ctx, Rational{2}), c.ref.basis(2))));
}
