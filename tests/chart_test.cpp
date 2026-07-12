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

// vibe 000073: a symmetric rank-2 field folds T_θr → T_rθ after expansion (the
// component inherits the slot-swap symmetry); symmetric is rank-2 only.
TEST(Chart, SymmetricFieldFoldsComponents)
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
    auto* T = make_field(ctx, make_tensor_name("T"), 2, {}, /*symmetric=*/true);
    auto* Tc = steps::canonicalize(
        ctx,
        steps::unroll_sums(
            ctx, expand_in_basis(ctx, T, fb, Variance::Covariant)));
    // T_θr and T_rθ are the same component: e_θ·T·e_r = e_r·T·e_θ.
    auto* c_tr = steps::simplify_scalars(
        ctx,
        simplify_basis_dot(
            ctx,
            steps::expand_products(
                ctx,
                make_dot(
                    ctx,
                    make_dot(ctx, fb.direction(ctx, 1), Tc),
                    fb.direction(ctx, 0))),
            fb));
    auto* c_rt = steps::simplify_scalars(
        ctx,
        simplify_basis_dot(
            ctx,
            steps::expand_products(
                ctx,
                make_dot(
                    ctx,
                    make_dot(ctx, fb.direction(ctx, 0), Tc),
                    fb.direction(ctx, 1))),
            fb));
    EXPECT_TRUE(eq(ctx, c_tr, c_rt));

    // symmetric is only defined for rank 2.
    EXPECT_THROW(
        (void)make_field(ctx, make_tensor_name("A"), 3, {}, true),
        std::invalid_argument);
}

// vibe 000073 Route A: chart.expand materializes an abstract field into frame
// components (a field derivative via the connection), the operators
// expand-first on an abstract field, and chart.components surfaces the scalar
// components.
TEST(Chart, RouteAExpandAndComponents)
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
    auto* T = make_field(
        ctx,
        make_tensor_name("T"),
        2,
        {CoordinateRef{2, 0, true}, CoordinateRef{2, 1, false}});

    // expand of a plain field yields a rank-2 dyadic (non-scalar, non-zero).
    auto* ex = expand(ctx, chart, T);
    EXPECT_EQ(infer_rank(ex), std::optional<int>{2});

    // The operator expands-first: div of the abstract field, surfaced as
    // components, agrees with div of the explicitly expanded field.
    auto comps_A = components(ctx, chart, divergence(ctx, chart, T));
    ASSERT_EQ(comps_A.size(), 3u);
    auto comps_B = components(ctx, chart, divergence(ctx, chart, ex));
    for (int i = 0; i < 3; ++i)
        EXPECT_TRUE(
            eq(ctx,
               steps::simplify_scalars(
                   ctx, make_difference(ctx, comps_A[i], comps_B[i])),
               make_scalar(ctx, Rational{0})));
    // The radial component is not constant (it carries ∂_r T_rr etc.).
    EXPECT_FALSE(eq(ctx, comps_A[0], make_scalar(ctx, Rational{0})));
}

// vibe 000074: component_matrix surfaces a rank-2 tensor as the physical
// component matrix e_i·T·e_j; components refuses a rank-2 input loudly.
TEST(Chart, ComponentMatrixOfSymmetricField)
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
    auto* T = make_field(ctx, make_tensor_name("T"), 2, {}, /*symmetric=*/true);

    auto m = component_matrix(ctx, chart, T);
    ASSERT_EQ(m.size(), 3u);
    for (auto const& row: m)
        ASSERT_EQ(row.size(), 3u);
    // Every entry is a scalar (rank 0) and nonzero — the minted component.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(infer_rank(m[i][j]), std::optional<int>{0});
            EXPECT_FALSE(eq(ctx, m[i][j], make_scalar(ctx, Rational{0})));
        }
    // Symmetry folds: the matrix itself is symmetric, entry by entry.
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            EXPECT_TRUE(eq(ctx, m[i][j], m[j][i]));
    // Off-diagonal entries are distinct components.
    EXPECT_FALSE(eq(ctx, m[0][1], m[0][2]));
    EXPECT_FALSE(eq(ctx, m[0][0], m[1][1]));

    // components refuses the rank-2 input, pointing at component_matrix.
    EXPECT_THROW((void)components(ctx, chart, T), std::invalid_argument);
}

// vibe 000075 gap A: an invariant wrapper (tr) around an abstract field must
// self-prepare — grad(tr ε) previously tripped canonicalize's encapsulate on
// the un-opened Trace(Σ ε_ij e_i⊗e_j).  It must agree with the manual
// workaround grad(expand(tr ε)).
TEST(Chart, GradOfTraceOfField)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 7, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 7, 2);
    CoordinateChart chart{ref, {x, y, z}, {x, y, z}};
    auto* T = make_field(ctx, make_tensor_name("T"), 2, {}, /*symmetric=*/true);
    auto* theta = make_trace(ctx, T);

    Expr const* direct = nullptr;
    ASSERT_NO_THROW(direct = gradient(ctx, chart, theta));
    Expr const* via_expand = gradient(ctx, chart, expand(ctx, chart, theta));
    auto cd = components(ctx, chart, direct);
    auto ce = components(ctx, chart, via_expand);
    for (int i = 0; i < 3; ++i)
        EXPECT_TRUE(
            eq(ctx,
               steps::simplify_scalars(ctx, make_difference(ctx, cd[i], ce[i])),
               make_scalar(ctx, Rational{0})));
    EXPECT_FALSE(eq(ctx, cd[0], make_scalar(ctx, Rational{0})));
}

// vibe 000075 gaps B+C: projection sees through an atomic identity tensor
// (θ·I would leave (e_i·I)·e_j unreduced) and through a scalar quotient
// fence, including one nested inside a product (2·(X/2)).
TEST(Chart, ComponentMatrixSeesThroughIdentityAndQuotient)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 7, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 7, 2);
    CoordinateChart chart{ref, {x, y, z}, {x, y, z}};
    auto fb = physical_frame(ctx, chart);

    // 5·I → 5 δ_ij.
    auto* five_I = make_tensor_product(
        ctx, make_scalar(ctx, Rational{5}), make_identity(ctx));
    auto mI = component_matrix(ctx, chart, five_I);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_TRUE(
                eq(ctx, mI[i][j], make_scalar(ctx, Rational{i == j ? 5 : 0})));

    // (e_x⊗e_x + e_y⊗e_y)/2 → diag(1/2, 1/2, 0).
    auto* dyads = make_sum(
        ctx,
        make_tensor_product(ctx, fb.direction(ctx, 0), fb.direction(ctx, 0)),
        make_tensor_product(ctx, fb.direction(ctx, 1), fb.direction(ctx, 1)));
    auto* half = make_scalar_div(ctx, dyads, make_scalar(ctx, Rational{2}));
    auto mQ = component_matrix(ctx, chart, half);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_TRUE(
                eq(ctx,
                   mQ[i][j],
                   make_scalar(
                       ctx, i == j && i < 2 ? Rational{1, 2} : Rational{0})));

    // The quotient nested inside a product — 2·(X/2) — needs the fence
    // distributor to peel the ScalarDiv it exposes.
    auto* nested =
        make_tensor_product(ctx, make_scalar(ctx, Rational{2}), half);
    auto mN = component_matrix(ctx, chart, nested);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_TRUE(eq(
                ctx,
                mN[i][j],
                make_scalar(ctx, i == j && i < 2 ? Rational{1} : Rational{0})));
}

// vibe 000074: component_matrix of an explicit dyad picks out the single
// entry — no abstract field involved, so this exercises the pure projection.
TEST(Chart, ComponentMatrixOfDyad)
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
    auto* dyad = make_tensor_product(
        ctx, fb.direction(ctx, 1), fb.direction(ctx, 1)); // e_θ ⊗ e_θ
    auto m = component_matrix(ctx, chart, dyad);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_TRUE(
                eq(ctx,
                   m[i][j],
                   make_scalar(ctx, Rational{i == 1 && j == 1 ? 1 : 0})));
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

// ---- first-class ∇ operator (vibe 000077, step C) ----------------------

// del(chart) is the invariant operator Σ_i (1/h_i) e_i ∂_{q^i}: a rank-1
// expression (the e_i part), built without applying it to anything.
TEST(Chart, DelOperatorIsRankOneAndInspectable)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 7, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 7, 2);
    CoordinateChart chart{ref, {x, y, z}, {x, y, z}};

    auto* nabla = del(ctx, chart);
    EXPECT_EQ(infer_rank(nabla), std::optional{1}); // ∇ is a vector operator
    // It renders with ∂ operators (it is unapplied, not a value).
    IndexNameMap map;
    auto s = render_latex(*nabla, map, &ctx);
    EXPECT_NE(s.find("\\partial_{"), std::string::npos);
}

// Applying the first-class ∇ with ⊗ reproduces the gradient: apply_operators
// of (∇ ⊗ f) equals chart.gradient(f) for a scalar field.
TEST(Chart, DelAppliedWithTensorProductIsGradient)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 7, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 7, 2);
    CoordinateChart chart{ref, {x, y, z}, {x, y, z}};
    auto* f = make_field(ctx, make_tensor_name("f"), 0, {});

    auto* via_operator = steps::apply_operators(
        ctx, make_tensor_product(ctx, del(ctx, chart), f));
    EXPECT_TRUE(eq(ctx, via_operator, gradient(ctx, chart, f)));
}

// The cylindrical ∇ carries the 1/r scale on the θ term: its θ-direction
// derivative is (1/r) ∂_θ, so applying ∇ to a scalar field picks up the 1/r.
TEST(Chart, DelCylindricalHasInverseScaleFactor)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* r =
        make_coordinate(ctx, make_tensor_name("r"), 2, 0, /*nonneg=*/true);
    auto* th = make_coordinate(ctx, make_tensor_name("\\theta"), 2, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 2, 2);
    CoordinateChart chart{
        ref,
        {r, th, z},
        {mul(ctx, r, cos_(ctx, th)), mul(ctx, r, sin_(ctx, th)), z}};
    auto* f = make_field(ctx, make_tensor_name("f"), 0, {});

    // ∇f via the operator equals chart.gradient(f) (which carries the 1/r).
    auto* via_operator = steps::apply_operators(
        ctx, make_tensor_product(ctx, del(ctx, chart), f));
    EXPECT_TRUE(eq(ctx, via_operator, gradient(ctx, chart, f)));
    // And the rendered operator shows a 1/r (the θ scale factor).
    IndexNameMap map;
    auto s = render_latex(*del(ctx, chart), map, &ctx);
    EXPECT_NE(s.find("frac{1}{r}"), std::string::npos);
}

// ---- free-index ∇ expansion (vibe 000078, increment 2) -----------------

// A Cartesian (WCS) chart with an identity embedding — the constant unit-scale
// frame `expand_nabla` targets.
namespace
{
auto cartesian_chart(Context& ctx, Basis const& ref) -> CoordinateChart
{
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 9, 0);
    auto* y = make_coordinate(ctx, make_tensor_name("y"), 9, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 9, 2);
    return CoordinateChart{ref, {x, y, z}, {x, y, z}};
}
} // namespace

// expand_nabla lowers the chart-free inc ε = ∇×(∇×ε)ᵀ to the free-index
// interior e_i × (e_j × ∂_i∂_j ε)ᵀ — ε stays abstract (no components), only
// second derivatives ∂_i∂_j appear, and the result is still rank 2.
TEST(Chart, ExpandNablaKeepsEpsilonAbstract)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* eps =
        make_field(ctx, make_tensor_name("\\varepsilon"), 2, {}, /*sym=*/true);
    auto* nab = make_nabla(ctx);
    auto* inc =
        make_cross(ctx, nab, make_transpose(ctx, make_cross(ctx, nab, eps)));

    auto* ex = expand_nabla(ctx, chart, inc);
    EXPECT_EQ(infer_rank(ex), std::optional{2});
    IndexNameMap map;
    auto s = render_latex(*ex, map, &ctx);
    EXPECT_EQ(s.find("varepsilon_{"), std::string::npos); // ε never
                                                          // componentised
    EXPECT_NE(s.find("partial"), std::string::npos);      // ∂'s present
}

// The abstract free-index inc ε, once componentised, matches brute-force
// ∇×(∇×ε)ᵀ term by term — the free-index path computes the same tensor.
TEST(Chart, ExpandNablaComponentsMatchBruteForce)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* eps =
        make_field(ctx, make_tensor_name("\\varepsilon"), 2, {}, /*sym=*/true);
    auto* nab = make_nabla(ctx);
    auto* inc =
        make_cross(ctx, nab, make_transpose(ctx, make_cross(ctx, nab, eps)));

    auto* free_form =
        componentize_nabla(ctx, chart, expand_nabla(ctx, chart, inc));
    auto* brute = rot(ctx, chart, make_transpose(ctx, rot(ctx, chart, eps)));

    auto a = component_matrix(ctx, chart, free_form);
    auto b = component_matrix(ctx, chart, brute);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < a[i].size(); ++j)
            EXPECT_TRUE(eq(ctx, expand(ctx, chart, a[i][j]), b[i][j]))
                << "component (" << i << "," << j << ")";
}

// A single ∇ operator reproduces the chart operator after componentisation:
// ∇·v == div, ∇×ε == rot, componentwise.
TEST(Chart, ExpandNablaSingleOperatorsMatchChart)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* v = make_field(ctx, make_tensor_name("v"), 1, {});
    auto* nab = make_nabla(ctx);

    auto* div_free = componentize_nabla(
        ctx, chart, expand_nabla(ctx, chart, make_dot(ctx, nab, v)));
    EXPECT_TRUE(
        eq(ctx, expand(ctx, chart, div_free), divergence(ctx, chart, v)));
}

// ∇·(∇·ε) via expand_nabla equals the chart operators' div(div ε): the two
// independent ∂-summation indices must stay distinct through operator
// application — a premature intermediate canonicalize once aliased them to
// ∂_i∂_i (vibe 000078 bug 3a), collapsing the double divergence.
TEST(Chart, ExpandNablaDivDivKeepsIndicesDistinct)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* eps =
        make_field(ctx, make_tensor_name("\\varepsilon"), 2, {}, /*sym=*/true);
    auto* nab = make_nabla(ctx);
    auto* dd = make_dot(ctx, nab, make_dot(ctx, nab, eps));
    auto* free = componentize_nabla(ctx, chart, expand_nabla(ctx, chart, dd));
    EXPECT_TRUE(
        eq(ctx,
           expand(ctx, chart, free),
           divergence(ctx, chart, divergence(ctx, chart, eps))));
}

// (∇⊗(∇·ε))ᵀ via expand_nabla stays rank 2 and equals (grad(div ε))ᵀ.  A
// transposed grad-div's Leibniz term ∂_i e_j = 0 leaves a rank-inflating
// zero-product `(e_i ⊗ 0 ⊗ ∂_jε)ᵀ` that the transpose fence hides from
// canonicalize; folding the algebraic zero law in the cleanup keeps the true
// rank.  Needed for the reassembly target's 2(∇∇·ε)ˢ term (vibe 000078).
TEST(Chart, ExpandNablaTransposedGradDivKeepsRank)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* eps =
        make_field(ctx, make_tensor_name("\\varepsilon"), 2, {}, /*sym=*/true);
    auto* nab = make_nabla(ctx);
    // (∇⊗(∇·ε))ᵀ
    auto* gd = make_tensor_product(ctx, nab, make_dot(ctx, nab, eps));
    auto* e = make_transpose(ctx, gd);
    EXPECT_EQ(infer_rank(expand_nabla(ctx, chart, e)), std::optional{2});
    auto* free = componentize_nabla(ctx, chart, expand_nabla(ctx, chart, e));
    EXPECT_TRUE(
        eq(ctx,
           expand(ctx, chart, free),
           make_transpose(
               ctx, gradient(ctx, chart, divergence(ctx, chart, eps)))));
}

// reassemble_nabla (vibe 000078 increment 4) is the inverse of expand_nabla for
// single operators: expanding ∇⊗ε / ∇·ε to the free-index form and reassembling
// recovers the original operator expression.
TEST(Chart, ReassembleNablaRoundTripsSingleOperators)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* eps =
        make_field(ctx, make_tensor_name("\\varepsilon"), 2, {}, /*sym=*/true);
    auto* nab = make_nabla(ctx);
    auto* grad = make_tensor_product(ctx, nab, eps); // ∇⊗ε
    auto* div = make_dot(ctx, nab, eps);             // ∇·ε
    EXPECT_TRUE(
        eq(ctx,
           reassemble_nabla(ctx, chart, expand_nabla(ctx, chart, grad)),
           grad));
    EXPECT_TRUE(eq(
        ctx, reassemble_nabla(ctx, chart, expand_nabla(ctx, chart, div)), div));
}

// reassemble_nabla folds the double divergence ∇·(∇·ε) — the (1,1,2) fan-in —
// back from its free-index reduction to the operator form.
TEST(Chart, ReassembleNablaRecoversDoubleDivergence)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* eps =
        make_field(ctx, make_tensor_name("\\varepsilon"), 2, {}, /*sym=*/true);
    auto* nab = make_nabla(ctx);
    auto* divdiv = make_dot(ctx, nab, make_dot(ctx, nab, eps)); // ∇·(∇·ε)
    EXPECT_TRUE(
        eq(ctx,
           reassemble_nabla(ctx, chart, expand_nabla(ctx, chart, divdiv)),
           divdiv));
}

// reassemble_nabla folds the composite operator shapes whose term structure the
// single-operator round-trips do not reach: a transposed gradient (grad-leg to
// the right of the operand ⇒ the transpose branch), a gradient of a trace (the
// `tr` fold in fold_divergences), a grad-div ∇(∇·v) (a gradient leg over a
// divergence-folded operand), and a transposed grad-div (∇(∇·ε))ᵀ.  Each must
// round-trip from its free-index expansion back to the operator form.
TEST(Chart, ReassembleNablaRoundTripsCompositeOperators)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* eps =
        make_field(ctx, make_tensor_name("\\varepsilon"), 2, {}, /*sym=*/true);
    auto* v = make_field(ctx, make_tensor_name("v"), 1, {});
    auto* nab = make_nabla(ctx);

    auto roundtrips = [&](Expr const* op)
    {
        auto* free = steps::canonicalize(ctx, expand_nabla(ctx, chart, op));
        return eq(ctx, reassemble_nabla(ctx, chart, free), op);
    };

    // (∇⊗ε)ᵀ — a gradient leg to the right of the operand ⇒ transpose branch.
    EXPECT_TRUE(
        roundtrips(make_transpose(ctx, make_tensor_product(ctx, nab, eps))));
    // ∇⊗(tr ε) — the trace fold plus a gradient leg over a scalar operand.
    EXPECT_TRUE(
        roundtrips(make_tensor_product(ctx, nab, make_trace(ctx, eps))));
    // ∇⊗(∇·v) — a gradient leg over a divergence-folded operand.
    EXPECT_TRUE(
        roundtrips(make_tensor_product(ctx, nab, make_dot(ctx, nab, v))));
    // (∇⊗(∇·ε))ᵀ — transpose over a grad-div.
    EXPECT_TRUE(roundtrips(make_transpose(
        ctx, make_tensor_product(ctx, nab, make_dot(ctx, nab, eps)))));
}

// reassemble_nabla folds a *bilinear* cross term — two ∂-marked operands joined
// by an inter-gradient dot — into a dot of two gradients (vibe 000087).  The
// second-order Leibniz rule Δ(u e) = (Δu)e + 2(∇u)·(∇⊗e) + u Δe exercises it:
// the two middle copies are the cross term the single-operand classifier used
// to mis-fold to Δe (dropping ∇u).
TEST(Chart, ReassembleNablaFoldsBilinearCrossTerm)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* u = make_field(ctx, make_tensor_name("u"), 0, {}); // scalar field
    auto* w = make_field(ctx, make_tensor_name("w"), 1, {}); // vector field
    auto* nab = make_nabla(ctx);

    // Δ(u w) = ∇·(∇⊗(u w))
    auto* lap = make_dot(
        ctx,
        nab,
        make_tensor_product(ctx, nab, make_tensor_product(ctx, u, w)));
    auto* free = steps::canonicalize(ctx, expand_nabla(ctx, chart, lap));
    auto* reass = reassemble_nabla(ctx, chart, free);

    // (Δu) w + 2 (∇u)·(∇⊗w) + u (Δw)
    auto* lap_u = make_dot(ctx, nab, make_tensor_product(ctx, nab, u));
    auto* lap_w = make_dot(ctx, nab, make_tensor_product(ctx, nab, w));
    auto* cross = make_dot(
        ctx,
        make_tensor_product(ctx, nab, u),
        make_tensor_product(ctx, nab, w));
    auto* rhs = make_sum(
        ctx,
        make_sum(
            ctx,
            make_tensor_product(ctx, lap_u, w),
            make_tensor_product(ctx, make_scalar(ctx, Rational{2}), cross)),
        make_tensor_product(ctx, u, lap_w));
    EXPECT_TRUE(eq(ctx, reass, rhs));
}

// reassemble_nabla folds an identity-scaled invariant term — the `Δθ·I` /
// `(∇∇··ε)I` shapes the strain identity produces — reading the ⊗-adjacent I as
// a standalone identity factor and the δ-pair as a Laplacian on the trace
// scalar. (The strain example exercises this end-to-end; here it is isolated so
// the C++ suite covers the identity-factor branch of reassemble_term directly.)
TEST(Chart, ReassembleNablaFoldsIdentityScaledInvariant)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* eps =
        make_field(ctx, make_tensor_name("\\varepsilon"), 2, {}, /*sym=*/true);
    auto* nab = make_nabla(ctx);
    auto* id = make_identity(ctx);

    // Δθ = ∇·∇(tr ε), a scalar; ⊗ I makes the invariant term (Δθ) I.
    auto* dtheta = expand_nabla(
        ctx,
        chart,
        make_dot(ctx, nab, make_tensor_product(ctx, nab, make_trace(ctx, eps))));
    auto* term = steps::canonicalize(ctx, make_tensor_product(ctx, dtheta, id));
    auto* reass = reassemble_nabla(ctx, chart, term);
    // reassemble_nabla dimensions its identity for tr(I)=n, but dimension-
    // awareness is identity-neutral (vibe 000081), so a plain I compares equal.
    auto* want = make_tensor_product(
        ctx,
        make_dot(ctx, nab, make_tensor_product(ctx, nab, make_trace(ctx, eps))),
        id);
    EXPECT_TRUE(eq(ctx, reass, want));
}

// ∇·((∇·v) I) = ∇(∇·v): the outer ∂ must reach the inner divergence ∇·v (a ∂
// operator hidden inside a ⊗-factor, resolved by apply_operators' structured-
// factor recursion), then e_i·I folds to e_i and the term reassembles to a
// grad-div.  A scalar coefficient (a Lamé constant λ) rides through — the
// endpoint's ∇·(λ(∇·u)I) piece (vibe 000080 Increment 8).
TEST(Chart, ReassembleNablaDivOfScalarTimesIdentity)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* v = make_field(ctx, make_tensor_name("v"), 1, {});
    auto* nab = make_nabla(ctx);
    auto* id = make_identity(ctx);

    auto reduce = [&](Expr const* e)
    {
        auto* x = steps::canonicalize(ctx, expand_nabla(ctx, chart, e));
        x = steps::contract_identity(ctx, x);
        return reassemble_nabla(ctx, chart, steps::canonicalize(ctx, x));
    };

    // ∇·((∇·v) I) → ∇(∇·v).
    auto* divv_I = make_tensor_product(ctx, make_dot(ctx, nab, v), id);
    auto* graddiv = make_tensor_product(ctx, nab, make_dot(ctx, nab, v));
    EXPECT_TRUE(eq(ctx, reduce(make_dot(ctx, nab, divv_I)), graddiv));

    // ∇·(λ (∇·v) I) → λ ∇(∇·v): the scalar coefficient λ is preserved, not
    // dropped as (or overwritten by) the operand blob.
    auto* lam = make_tensor_object(ctx, make_tensor_name("\\lambda"), {}, 0);
    auto* lam_divv_I = make_tensor_product(ctx, lam, divv_I);
    auto* want = make_tensor_product(ctx, lam, graddiv);
    EXPECT_TRUE(eq(ctx, reduce(make_dot(ctx, nab, lam_divv_I)), want));
}

// A scalar-halved operator operand — the symmetric gradient sym(∇u) =
// (∇u + (∇u)ᵀ)/2, the standard elasticity strain — reduces cleanly: the
// constant /2 rides out and the ∂-mark direction indices stay linked to their
// frame vectors.  Before the constant-denominator diff rule the full quotient
// rule dragged an un-differentiated numerator copy through canonicalize, whose
// inconsistent alpha-renaming orphaned the ∂ marks and dropped the second
// derivatives entirely (vibe 000080, sym-form (b)).
TEST(Chart, ReassembleNablaDivOfSymmetricGradient)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* u = make_field(ctx, make_tensor_name("u"), 1, {});
    auto* nab = make_nabla(ctx);
    auto* two = make_scalar(ctx, Rational{2});

    auto reduce = [&](Expr const* e)
    {
        auto* x = steps::canonicalize(ctx, expand_nabla(ctx, chart, e));
        x = steps::contract_identity(ctx, x);
        return reassemble_nabla(ctx, chart, steps::canonicalize(ctx, x));
    };

    auto* gradu = make_tensor_product(ctx, nab, u);
    auto* sym = make_scalar_div(
        ctx, make_sum(ctx, gradu, make_transpose(ctx, gradu)), two);
    // ∇·(sym ∇u) = ½(∇(∇·u) + ∇·∇u) — the grad-div from (∇u)ᵀ, the Laplacian
    // from ∇u, each halved.
    auto* graddiv = make_tensor_product(ctx, nab, make_dot(ctx, nab, u));
    auto* lap = make_dot(ctx, nab, make_tensor_product(ctx, nab, u));
    auto* want = make_sum(
        ctx,
        make_scalar_div(ctx, graddiv, two),
        make_scalar_div(ctx, lap, two));
    EXPECT_TRUE(eq(ctx, reduce(make_dot(ctx, nab, sym)), want));
}

// expand_nabla(∇·∇f) for a SCALAR field keeps its true rank 0 (the Laplacian
// Δf) instead of degrading to a rank-2 dyad (vibe 000079).  The inner grad ∇f
// is the scalar-scaled frame vector (∂_i f) e_i; the outer ∇· must contract e_ℓ
// with that e_i.  Differentiating the constant e_i emits a Leibniz connection
// term `0 ⊗ ∂_i f` whose rank (0) differs from the real term's (1); left in the
// Sum it made `infer_rank` misread the operand as a scalar, so `make_dot`
// silently turned the `·` into `⊗`.  Folding forced zeros in the deferred
// derivative drops that term before it can mislead the contraction.
TEST(Chart, ExpandNablaScalarDivGradIsLaplacian)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto chart = cartesian_chart(ctx, ref);
    auto* f = make_field(ctx, make_tensor_name("f"), 0, {});
    auto* nab = make_nabla(ctx);
    // ∇·(∇f) — grad of a scalar is ∇⊗f, div contracts it back to a scalar.
    auto* divgrad = make_dot(ctx, nab, make_tensor_product(ctx, nab, f));
    auto* expanded = expand_nabla(ctx, chart, divgrad);
    EXPECT_EQ(infer_rank(expanded), std::optional{0});
    // Componentized, it is the chart Laplacian Δf (a scalar).
    auto* free = componentize_nabla(ctx, chart, expanded);
    EXPECT_TRUE(eq(ctx, expand(ctx, chart, free), laplacian(ctx, chart, f)));
    // And it round-trips back to the operator form ∇·∇f.
    EXPECT_TRUE(eq(ctx, reassemble_nabla(ctx, chart, expanded), divgrad));
}

// expand_nabla refuses a curvilinear (non-unit-scale) chart: the free-index
// ∂_i cannot carry the moving frame's per-direction scale factors.
TEST(Chart, ExpandNablaRejectsCurvilinear)
{
    Context ctx;
    auto ref = wcs(ctx);
    auto* r =
        make_coordinate(ctx, make_tensor_name("r"), 3, 0, /*nonneg=*/true);
    auto* th = make_coordinate(ctx, make_tensor_name("\\theta"), 3, 1);
    auto* z = make_coordinate(ctx, make_tensor_name("z"), 3, 2);
    CoordinateChart cyl{
        ref,
        {r, th, z},
        {mul(ctx, r, cos_(ctx, th)), mul(ctx, r, sin_(ctx, th)), z}};
    auto* f = make_field(ctx, make_tensor_name("f"), 0, {});
    EXPECT_THROW(
        (void)expand_nabla(
            ctx, cyl, make_tensor_product(ctx, make_nabla(ctx), f)),
        std::invalid_argument);
}
