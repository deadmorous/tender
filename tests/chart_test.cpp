// Coordinate charts (vibe 000069 M4): derive the orthogonal-curvilinear
// geometry — radius vector, tangent (holonomic) basis g_i, metric g_ij, scale
// factors h_i, and the physical orthonormal frame e_i — from a coordinate
// mapping, and validate the derived frame against the hand-written
// coord_system bases.

#include <tender/basis.hpp>
#include <tender/chart.hpp>
#include <tender/context.hpp>
#include <tender/coord_system.hpp>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/render.hpp>

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

// vibe 000073 Gap 2: div of a dyad whose *contracted* leg is e_θ used to throw
// "encapsulate: unsupported factor node" — the connection term ∂_θ e_θ = −e_r
// left a raw ⊗ inside a dot operand (a Negate-wrapped ⊗ that distribute_-
// contraction did not see through).  Now it reduces to −(a/r) e_r +
// (∂_θa/r)e_θ.
TEST(Chart, CylindricalDivEThetaDyad)
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
    auto fb = physical_frame(ctx, chart);
    auto* a = make_field(
        ctx,
        make_tensor_name("a"),
        0,
        {CoordinateRef{2, 0, true}, CoordinateRef{2, 1, false}});
    auto* et = fb.direction(ctx, 1);
    auto* T = mul(ctx, a, mul(ctx, et, et)); // a e_θ ⊗ e_θ
    auto* div = divergence(ctx, chart, T);   // must not throw
    auto* datheta = make_field_derivative(
        ctx, a, make_tensor_name("\\theta"), CoordinateRef{2, 1, false});
    auto* want = make_sum(
        ctx,
        make_negate(
            ctx, mul(ctx, make_scalar_div(ctx, a, r), fb.direction(ctx, 0))),
        mul(ctx, make_scalar_div(ctx, datheta, r), fb.direction(ctx, 1)));
    EXPECT_TRUE(eq(ctx, div, want));
}

// vibe 000073: a bare basis has no connection Ω, so expanding a field
// derivative ∂T on a moving frame would silently drop the connection terms.
// expand_in_basis refuses loudly; on a constant (Cartesian) frame, where
// ∂ e_i = 0, the same expansion is legal and allowed.
TEST(Chart, ExpandInBasisRefusesFieldDerivativeOnMovingFrame)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* r = make_coordinate(ctx, make_tensor_name("r"), 2, 0, true);
    auto* th = make_coordinate(ctx, make_tensor_name("\\theta"), 2, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 2, 2);
    CoordinateChart cyl{
        ref,
        {r, th, z},
        {mul(ctx, r, cos_(ctx, th)), mul(ctx, r, sin_(ctx, th)), z}};
    auto moving = physical_frame(ctx, cyl); // registers the curvilinear
                                            // connection
    auto* T = make_field(ctx, make_tensor_name("T"), 2, {});
    auto* dT = make_field_derivative(
        ctx, T, make_tensor_name("r"), CoordinateRef{2, 0, true});
    EXPECT_THROW(
        (void)expand_in_basis(ctx, dT, moving, Variance::Covariant),
        std::invalid_argument);

    // Cartesian chart: constant frame, ∂ e_i = 0, so the same expansion is
    // fine.
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 8, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 8, 1);
    auto* zc = make_coordinate(ctx, make_tensor_name("z"), 8, 2);
    CoordinateChart cart{ref, {x, y, zc}, {x, y, zc}};
    auto flat = physical_frame(ctx, cart);
    auto* S = make_field(ctx, make_tensor_name("S"), 2, {});
    auto* dS = make_field_derivative(
        ctx, S, make_tensor_name("x"), CoordinateRef{8, 0, false});
    EXPECT_NO_THROW((void)expand_in_basis(ctx, dS, flat, Variance::Covariant));
}

// vibe 000073 Gap 3: physical_basis and physical_frame return the *same* Basis
// identity, in either call order — previously physical_basis minted a fresh id
// each call, so its e_i differed structurally from the operators' and silently
// defeated simplify_basis_dot.
TEST(Chart, PhysicalBasisAndFrameShareIdentity)
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
    auto frame = physical_frame(ctx, chart);
    auto basis = physical_basis(ctx, chart);
    EXPECT_EQ(frame.basis_id(), basis.basis_id());
    for (int k = 0; k < 3; ++k)
        EXPECT_TRUE(
            structural_eq(frame.direction(ctx, k), basis.direction(ctx, k)));
    // Repeated physical_basis calls are idempotent (same id).
    EXPECT_EQ(physical_basis(ctx, chart).basis_id(), basis.basis_id());
}

// vibe 000073: a component of a field is itself a field, so expand_in_basis
// mints components that still differentiate (∂_r T_rr ≠ 0).  Before the fix the
// components were constants and div dropped every derivative term.
TEST(Chart, ExpandInBasisComponentsAreFields)
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
    auto fb = physical_frame(ctx, chart);
    auto* T = make_field(ctx, make_tensor_name("T"), 2, {});
    auto* Tc = expand_in_basis(ctx, T, fb, Variance::Covariant);
    // The expanded components depend on the coordinates, so ∂_r is nonzero.
    EXPECT_FALSE(
        eq(ctx, steps::partial(ctx, Tc, r), make_scalar(ctx, Rational{0})));

    // vibe 000073 Gap 4: a LaTeX-command index (\theta) followed by a Latin one
    // (r) must render as "\theta r", not the invalid control word "\thetar".
    auto* unrolled = steps::canonicalize(ctx, steps::unroll_sums(ctx, Tc));
    IndexNameMap map;
    auto s = render_latex(*unrolled, map, &ctx);
    EXPECT_EQ(s.find("\\thetar"), std::string::npos);
    EXPECT_NE(s.find("T_{\\theta r}"), std::string::npos);
    EXPECT_NE(s.find("T_{rr}"), std::string::npos); // plain Latin stays
                                                    // unspaced
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
    auto fb = physical_frame(ctx, chart);
    // The position vector on the (Cartesian) frame: R = x e_x + y e_y + z e_z.
    Expr const* R = nullptr;
    Expr const* coord[] = {x, y, z};
    for (int k = 0; k < 3; ++k)
    {
        auto* term = mul(ctx, coord[k], fb.direction(ctx, k));
        R = R ? make_sum(ctx, R, term) : term;
    }

    // ∇R = I intrinsically: the resolution Σ_k e_k⊗e_k folds to the identity
    // tensor, now in the chart's own frame (vibe 000071).
    EXPECT_TRUE(structural_eq(gradient(ctx, chart, R), make_identity(ctx)));
    EXPECT_TRUE(
        eq(ctx, divergence(ctx, chart, R), make_scalar(ctx, Rational{3})));
    EXPECT_TRUE(eq(ctx, rot(ctx, chart, R), s0(ctx)));
}

// A field built with the identity tensor and a cross — R×I, the rank-2 skew
// tensor — reduces on the frame instead of crashing (vibe 000071): the operator
// expands I → Σ e_k⊗e_k and reduces the frame crosses.  ∇×(R×I) = −2I.
TEST(Chart, RotOfRCrossIdentity)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 7, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 7, 2);
    CoordinateChart chart{ref, {x, y, z}, {x, y, z}};
    auto fb = physical_frame(ctx, chart);
    Expr const* R = nullptr;
    Expr const* coord[] = {x, y, z};
    for (int k = 0; k < 3; ++k)
    {
        auto* term = mul(ctx, coord[k], fb.direction(ctx, k));
        R = R ? make_sum(ctx, R, term) : term;
    }
    auto* RxI = make_cross(ctx, R, make_identity(ctx));
    auto* want = make_negate(
        ctx, mul(ctx, make_scalar(ctx, Rational{2}), make_identity(ctx)));
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, rot(ctx, chart, RxI)),
        steps::canonicalize(ctx, want))); // ∇×(R×I) = −2I
}

// A cross of two constant (non-field) vectors differentiates to zero by
// Leibniz; the operator yields 0 rather than crashing.
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

// The public frame dot/cross reduce contractions in the chart's physical frame
// (vibe 000071): e_x·e_x = 1, e_x·e_y = 0, e_x×e_y = e_z on the frame's atoms.
TEST(Chart, FrameDotAndCross)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 7, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 7, 2);
    CoordinateChart chart{ref, {x, y, z}, {x, y, z}};
    auto fb = physical_frame(ctx, chart);

    EXPECT_TRUE(
        eq(ctx,
           frame_dot(ctx, chart, fb.direction(ctx, 0), fb.direction(ctx, 0)),
           make_scalar(ctx, Rational{1})));
    EXPECT_TRUE(
        eq(ctx,
           frame_dot(ctx, chart, fb.direction(ctx, 0), fb.direction(ctx, 1)),
           s0(ctx)));
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(
            ctx,
            frame_cross(ctx, chart, fb.direction(ctx, 0), fb.direction(ctx, 1))),
        steps::canonicalize(ctx, fb.direction(ctx, 2))));
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

// Intrinsic differentiation (vibe 000071 P2): ∂ of a frame-vector atom is
// resolved through the connection, staying on the symbolic e_k — no trig, no
// WCS.  Leibniz then differentiates a curvilinear field f(r) e_r.
TEST(Chart, IntrinsicBasisVectorDifferentiation)
{
    Context ctx;
    auto c = make_cyl(ctx); // chart_id 2: r(0), θ(1), z(2)
    auto fb = physical_frame(ctx, c.chart);
    auto* e_r = fb.direction(ctx, 0);
    auto* e_th = fb.direction(ctx, 1);

    // ∂_θ e_r = e_θ, ∂_r e_r = 0 — directly on the atoms.
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, steps::partial(ctx, e_r, c.th)),
        steps::canonicalize(ctx, e_th)));
    EXPECT_TRUE(eq(ctx, steps::partial(ctx, e_r, c.r), s0(ctx)));

    // Leibniz on a curvilinear field: ∂_θ(f(r) e_r) = f ∂_θ e_r = f e_θ.
    auto* f = make_field(
        ctx, make_tensor_name("f"), 0, {CoordinateRef{2, 0, false}}); // f(r)
    auto* field_vec = make_tensor_product(ctx, f, e_r);
    auto* got = steps::simplify_scalars(
        ctx, steps::canonicalize(ctx, steps::partial(ctx, field_vec, c.th)));
    auto* want = steps::canonicalize(ctx, make_tensor_product(ctx, f, e_th));
    EXPECT_TRUE(structural_eq(got, want));
}

// The intrinsic differentiation ∂_j e_i = Σ_k γ^k_{ij} e_k is basis-agnostic
// (vibe 000071 P5, decision 3): it works on an oblique basis via its registered
// connection, and the dot is metric-aware (e_i·e_j = g_ij, not δ_ij) — the
// machinery is not hard-wired to orthonormal frames.
TEST(Chart, IntrinsicDifferentiationIsBasisAgnostic)
{
    Context ctx;
    // An oblique (non-orthonormal) basis a, b, c.
    auto ob = make_oblique_basis(
        ctx, space_3d(), {frame(ctx, "a"), frame(ctx, "b"), frame(ctx, "c")});
    ASSERT_NE(ob.basis_id(), 0);
    EXPECT_FALSE(ob.is_orthonormal());

    // Register a connection for a chart (id 9, one coordinate u): ∂_u e_0 =
    // e_1.
    auto* u = make_coordinate(ctx, make_tensor_name("u"), 9, 0);
    auto* conn = ctx.make<BasisConnection>();
    conn->chart_id = 9;
    conn->basis_id = ob.basis_id();
    auto const vals = ob.space()->values();
    conn->values.assign(vals.begin(), vals.end());
    conn->deriv = {{ob.direction(ctx, 1)}, {s0(ctx)}, {s0(ctx)}}; // ∂_u e_i,
                                                                  // one coord
    ctx.register_connection(conn, ob.basis_id());

    // ∂_u e_0 = e_1, resolved through the connection on the oblique frame.
    EXPECT_TRUE(structural_eq(
        steps::partial(ctx, ob.direction(ctx, 0), u), ob.direction(ctx, 1)));

    // The dot on the oblique basis yields the metric g (same-variance), not the
    // Kronecker δ an orthonormal frame would give.
    auto* dot = make_dot(ctx, ob.direction(ctx, 0), ob.direction(ctx, 1));
    IndexNameMap map;
    auto const tex = render_latex(*simplify_basis_dot(ctx, dot, ob), map);
    EXPECT_NE(tex.find("g"), std::string::npos); // metric, not δ
}

// Basis-to-basis expansion (vibe 000071 P4): a frame result can be brought to
// WCS on demand (e_r → cos θ i + sin θ j), and expressed in another frame; the
// round-trip WCS → frame recovers the frame vector.
TEST(Chart, BasisToBasisExpansion)
{
    Context ctx;
    auto c = make_cyl(ctx);
    auto fb = physical_frame(ctx, c.chart);
    auto* e_r = fb.direction(ctx, 0);

    // to_reference: e_r = cos θ i + sin θ j (WCS).
    auto* i = c.ref.basis(0);
    auto* j = c.ref.basis(1);
    auto* er_wcs = make_sum(
        ctx, mul(ctx, cos_(ctx, c.th), i), mul(ctx, sin_(ctx, c.th), j));
    EXPECT_TRUE(eq(ctx, to_reference(ctx, e_r), er_wcs));

    // express: WCS i in the cylindrical frame is cos θ e_r − sin θ e_θ.
    auto* e_th = fb.direction(ctx, 1);
    auto* want = make_difference(
        ctx, mul(ctx, cos_(ctx, c.th), e_r), mul(ctx, sin_(ctx, c.th), e_th));
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, express(ctx, c.chart, i)),
        steps::canonicalize(ctx, want)));

    // Round-trip WCS → frame recovers e_r.
    EXPECT_TRUE(eq(ctx, express(ctx, c.chart, to_reference(ctx, e_r)), e_r));
}

// vibe 000072 Obs 1: an identity (Cartesian) chart's physical frame IS the
// reference basis — same basis id and vectors — so it prints i, j, k (not
// e_x, e_y, e_z) and a completed resolution of identity folds without a naming
// split.  A curvilinear frame stays a distinct "e" basis.
TEST(Chart, CartesianFrameReusesReferenceBasis)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 7, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 7, 2);
    CoordinateChart cart{ref, {x, y, z}, {x, y, z}};
    auto fb = physical_frame(ctx, cart);
    EXPECT_EQ(fb.basis_id(), ref.basis_id());
    for (int k = 0; k < 3; ++k)
        EXPECT_TRUE(structural_eq(fb.direction(ctx, k), ref.direction(ctx, k)));

    // A genuinely curvilinear frame is a *different* basis.
    auto c = make_cyl(ctx);
    EXPECT_NE(physical_frame(ctx, c.chart).basis_id(), ref.basis_id());
}

// vibe 000072 Obs 2: the physical-frame cache is validated by the chart's
// geometry fingerprint, so reusing a chart_id for a different chart rebuilds
// the frame instead of returning the stale one.
TEST(Chart, PhysicalFrameRebuildsOnChartIdReuse)
{
    Context ctx;
    auto ref = wcs(ctx);
    // Spherical on chart_id 5.
    auto* r = make_coordinate(ctx, make_tensor_name("r"), 5, 0, true);
    auto* th = make_coordinate(ctx, make_tensor_name("\\theta"), 5, 1);
    auto* ph = make_coordinate(ctx, make_tensor_name("\\varphi"), 5, 2);
    CoordinateChart sph{
        ref,
        {r, th, ph},
        {mul(ctx, mul(ctx, r, sin_(ctx, th)), cos_(ctx, ph)),
         mul(ctx, mul(ctx, r, sin_(ctx, th)), sin_(ctx, ph)),
         mul(ctx, r, cos_(ctx, th))}};
    int const sid = physical_frame(ctx, sph).basis_id();

    // Reuse chart_id 5 for a cylindrical chart (different embedding).
    auto* z2 = make_coordinate(ctx, make_tensor_name("z"), 5, 2);
    CoordinateChart cyl{
        ref,
        {r, th, z2},
        {mul(ctx, r, cos_(ctx, th)), mul(ctx, r, sin_(ctx, th)), z2}};
    int const cid = physical_frame(ctx, cyl).basis_id();
    EXPECT_NE(cid, sid); // not the stale spherical frame

    // Re-fetching the spherical chart rebuilds its own frame, distinct from the
    // cylindrical one that now holds chart_id 5's slot.
    EXPECT_NE(physical_frame(ctx, sph).basis_id(), cid);
}

// vibe 000072 Obs 6: position() is the radius vector in the chart's own frame
// (cylindrical r e_r + z e_z), so grad(position) folds to I with no mixed
// frame.
TEST(Chart, IntrinsicPosition)
{
    Context ctx;
    auto c = make_cyl(ctx);
    auto fb = physical_frame(ctx, c.chart);
    auto* want = make_sum(
        ctx,
        mul(ctx, c.r, fb.direction(ctx, 0)),
        mul(ctx, c.z, fb.direction(ctx, 2)));
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, position(ctx, c.chart)),
        steps::canonicalize(ctx, want)));
    EXPECT_TRUE(structural_eq(
        gradient(ctx, c.chart, position(ctx, c.chart)), make_identity(ctx)));
}

// vibe 000072 Obs 8: express re-expresses *every* leg (not just one) and folds
// a completed resolution of identity.  A cross-frame ∇R re-expressed in the
// Cartesian frame becomes I; a rank-2 dyad round-trips through the frame.
TEST(Chart, ExpressAllLegsAndFold)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 7, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 7, 2);
    CoordinateChart cart{ref, {x, y, z}, {x, y, z}};
    auto c = make_cyl(ctx);

    // ∇R (cyl frame, from the WCS radius vector) re-expressed in Cartesian → I.
    auto* gradR = gradient(ctx, c.chart, radius_vector(ctx, c.chart));
    EXPECT_TRUE(structural_eq(express(ctx, cart, gradR), make_identity(ctx)));

    // A rank-2 dyad i⊗j re-expressed in the cylindrical frame has *both* legs
    // on e_r/e_θ (a one-leg projection would leave i or j behind):
    //   i⊗j = sinθcosθ e_r e_r + cos²θ e_r e_θ − sin²θ e_θ e_r − sinθcosθ e_θ
    //   e_θ.
    auto fbc = physical_frame(ctx, c.chart);
    auto* er = fbc.direction(ctx, 0);
    auto* et = fbc.direction(ctx, 1);
    auto* cs = cos_(ctx, c.th);
    auto* sn = sin_(ctx, c.th);
    auto* ij = make_tensor_product(ctx, c.ref.basis(0), c.ref.basis(1));
    auto* want = make_sum(
        ctx,
        make_sum(
            ctx,
            make_sum(
                ctx,
                mul(ctx, mul(ctx, sn, cs), mul(ctx, er, er)),
                mul(ctx, mul(ctx, cs, cs), mul(ctx, er, et))),
            make_negate(ctx, mul(ctx, mul(ctx, sn, sn), mul(ctx, et, er)))),
        make_negate(ctx, mul(ctx, mul(ctx, sn, cs), mul(ctx, et, et))));
    EXPECT_TRUE(structural_eq(
        steps::simplify_scalars(
            ctx, steps::canonicalize(ctx, express(ctx, c.chart, ij))),
        steps::simplify_scalars(ctx, steps::canonicalize(ctx, want))));
}

// vibe 000072 Obs 3: validate_chart rejects coords not in slot order 0..n-1, a
// foreign chart_id, or an embedding whose size differs from reference.dim().
TEST(Chart, ValidateChartRejectsBadShape)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* r = make_coordinate(ctx, make_tensor_name("r"), 2, 0, true);
    auto* th = make_coordinate(ctx, make_tensor_name("\\theta"), 2, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 2, 2);
    auto* emb0 = mul(ctx, r, cos_(ctx, th));
    auto* emb1 = mul(ctx, r, sin_(ctx, th));
    EXPECT_NO_THROW(
        validate_chart(CoordinateChart{ref, {r, th, z}, {emb0, emb1, z}}));
    EXPECT_THROW(
        validate_chart(CoordinateChart{ref, {th, r, z}, {emb0, emb1, z}}),
        std::invalid_argument); // slots 1,0,2
    EXPECT_THROW(
        validate_chart(CoordinateChart{ref, {r, th, z}, {emb0, emb1}}),
        std::invalid_argument); // embedding size 2 ≠ 3
}

// ∇f: the 1/h_i factors are the curvilinear content — for cylindrical
// ∇ = e_r ∂_r + (1/r) e_θ ∂_θ + e_z ∂_z, so ∇θ = (1/r) e_θ and ∇r² = 2r e_r,
// each returned intrinsically on the frame's own e_r, e_θ (vibe 000071).
TEST(Chart, CylindricalGradient)
{
    Context ctx;
    auto c = make_cyl(ctx);
    auto fb = physical_frame(ctx, c.chart);

    // ∇θ = (1/r) e_θ
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, gradient(ctx, c.chart, c.th)),
        steps::canonicalize(
            ctx,
            mul(ctx,
                make_scalar_div(ctx, s1(ctx), c.r),
                fb.direction(ctx, 1)))));
    // ∇r² = 2r e_r
    auto* r2 = make_pow(ctx, c.r, make_scalar(ctx, Rational{2}));
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, gradient(ctx, c.chart, r2)),
        steps::canonicalize(
            ctx,
            mul(ctx,
                mul(ctx, make_scalar(ctx, Rational{2}), c.r),
                fb.direction(ctx, 0)))));
}

// div of the radial field v = r e_r is 2 (= (1/r) ∂_r(r·r)).
TEST(Chart, CylindricalDivergence)
{
    Context ctx;
    auto c = make_cyl(ctx);
    auto fb = physical_frame(ctx, c.chart);
    auto* v = mul(ctx, c.r, fb.direction(ctx, 0)); // r e_r
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

// rot of the rigid-rotation field v = r e_θ is the uniform vorticity 2 e_z,
// intrinsically on the frame's own e_z (vibe 000071).
TEST(Chart, CylindricalRot)
{
    Context ctx;
    auto c = make_cyl(ctx);
    auto fb = physical_frame(ctx, c.chart);
    auto* v = mul(ctx, c.r, fb.direction(ctx, 1)); // r e_θ
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, rot(ctx, c.chart, v)),
        steps::canonicalize(
            ctx,
            mul(ctx, make_scalar(ctx, Rational{2}), fb.direction(ctx, 2)))));
}
