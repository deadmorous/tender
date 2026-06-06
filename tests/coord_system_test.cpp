#include <tender/coord_system.hpp>

#include <gtest/gtest.h>

using namespace tender;

static ResourceList make_rl()
{
    return ResourceList{};
}

// ===========================================================================
// WCS
// ===========================================================================

TEST(WCS, Dim) { EXPECT_EQ(wcs().dim(), 3); }

TEST(WCS, IsOrthonormal) { EXPECT_TRUE(wcs().is_orthonormal()); }

TEST(WCS, CoordsAreParameters)
{
    auto const& cs = wcs();
    for (int i = 0; i < 3; ++i)
    {
        ASSERT_NE(cs.coord(i), nullptr);
        EXPECT_NE(dynamic_cast<Parameter*>(cs.coord(i)), nullptr);
    }
}

TEST(WCS, CoordSymbols)
{
    auto const& cs = wcs();
    EXPECT_EQ(cs.coord(0)->symbol(), "x");
    EXPECT_EQ(cs.coord(1)->symbol(), "y");
    EXPECT_EQ(cs.coord(2)->symbol(), "z");
}

TEST(WCS, BasisIsRank1)
{
    auto const& cs = wcs();
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(cs.basis(i)->rank(), 1);
}

TEST(WCS, CobasisEqualsBasic)
{
    auto const& cs = wcs();
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(cs.cobasis(i), cs.basis(i));
}

TEST(WCS, DiagonalMetric)
{
    auto const& cs = wcs();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            auto* m = cs.metric(i, j);
            ASSERT_NE(dynamic_cast<RationalConst*>(m), nullptr);
            Rational expected{i == j ? 1 : 0};
            EXPECT_EQ(dynamic_cast<RationalConst*>(m)->value(), expected);
        }
}

// ===========================================================================
// DirectBasisCS — cobasis derivation
// ===========================================================================

TEST(DirectBasisCS, Dim)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* b = make_named_tensor(rl, "b", 1, {});
    auto* c = make_named_tensor(rl, "c", 1, {});
    auto cs = make_direct_basis_cs(a, b, c);
    EXPECT_EQ(cs->dim(), 3);
}

TEST(DirectBasisCS, NotOrthonormal)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* b = make_named_tensor(rl, "b", 1, {});
    auto* c = make_named_tensor(rl, "c", 1, {});
    auto cs = make_direct_basis_cs(a, b, c);
    EXPECT_FALSE(cs->is_orthonormal());
}

TEST(DirectBasisCS, NoCoords)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* b = make_named_tensor(rl, "b", 1, {});
    auto* c = make_named_tensor(rl, "c", 1, {});
    auto cs = make_direct_basis_cs(a, b, c);
    EXPECT_EQ(cs->coord(0), nullptr);
}

TEST(DirectBasisCS, BasisVectorsPreserved)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* b = make_named_tensor(rl, "b", 1, {});
    auto* c = make_named_tensor(rl, "c", 1, {});
    auto cs = make_direct_basis_cs(a, b, c);
    EXPECT_EQ(cs->basis(0), a);
    EXPECT_EQ(cs->basis(1), b);
    EXPECT_EQ(cs->basis(2), c);
}

TEST(DirectBasisCS, MetricIsContractOfBasisVectors)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* b = make_named_tensor(rl, "b", 1, {});
    auto* c = make_named_tensor(rl, "c", 1, {});
    auto cs = make_direct_basis_cs(a, b, c);
    // metric(i, j) = g_i · g_j  — should be a Contract
    auto* m00 = cs->metric(0, 0);
    auto* co = dynamic_cast<Contract*>(m00);
    ASSERT_NE(co, nullptr);
    EXPECT_EQ(co->lhs(), a);
    EXPECT_EQ(co->rhs(), a);
    EXPECT_EQ(m00->rank(), 0);
}

TEST(DirectBasisCS, CobasisRank1)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* b = make_named_tensor(rl, "b", 1, {});
    auto* c = make_named_tensor(rl, "c", 1, {});
    auto cs = make_direct_basis_cs(a, b, c);
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(cs->cobasis(i)->rank(), 1);
}

// Exit criterion: cobasis is structurally correct — each g^i is
// TensorProduct(Pow(volume, -1), cross_product) where the cross product
// involves the other two basis vectors.
TEST(DirectBasisCS, CobasisStructure)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* b = make_named_tensor(rl, "b", 1, {});
    auto* c = make_named_tensor(rl, "c", 1, {});
    auto cs = make_direct_basis_cs(a, b, c);

    // g^0 = (b×c) / V  →  TensorProduct(Pow(V,-1), CrossProduct(b,c))
    auto* g0 = cs->cobasis(0);
    auto* tp = dynamic_cast<TensorProduct*>(g0);
    ASSERT_NE(tp, nullptr);

    auto* inv_vol = dynamic_cast<Pow*>(tp->lhs());
    ASSERT_NE(inv_vol, nullptr);
    EXPECT_EQ(inv_vol->exponent(), Rational{-1});

    // The volume is a Contract(a, CrossProduct(b,c))
    auto* vol = dynamic_cast<Contract*>(inv_vol->base());
    ASSERT_NE(vol, nullptr);
    EXPECT_EQ(vol->lhs(), a);
    auto* bxc_vol = dynamic_cast<CrossProduct*>(vol->rhs());
    ASSERT_NE(bxc_vol, nullptr);
    EXPECT_EQ(bxc_vol->lhs(), b);
    EXPECT_EQ(bxc_vol->rhs(), c);

    // The cross product in the numerator is also b×c
    auto* bxc = dynamic_cast<CrossProduct*>(tp->rhs());
    ASSERT_NE(bxc, nullptr);
    EXPECT_EQ(bxc->lhs(), b);
    EXPECT_EQ(bxc->rhs(), c);
}

TEST(DirectBasisCS, CobasisPermutations)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* b = make_named_tensor(rl, "b", 1, {});
    auto* c = make_named_tensor(rl, "c", 1, {});
    auto cs = make_direct_basis_cs(a, b, c);

    // g^1 = TensorProduct(inv_vol, c×a)
    auto* g1 = dynamic_cast<TensorProduct*>(cs->cobasis(1));
    ASSERT_NE(g1, nullptr);
    auto* cxa = dynamic_cast<CrossProduct*>(g1->rhs());
    ASSERT_NE(cxa, nullptr);
    EXPECT_EQ(cxa->lhs(), c);
    EXPECT_EQ(cxa->rhs(), a);

    // g^2 = TensorProduct(inv_vol, a×b)
    auto* g2 = dynamic_cast<TensorProduct*>(cs->cobasis(2));
    ASSERT_NE(g2, nullptr);
    auto* axb = dynamic_cast<CrossProduct*>(g2->rhs());
    ASSERT_NE(axb, nullptr);
    EXPECT_EQ(axb->lhs(), a);
    EXPECT_EQ(axb->rhs(), b);
}

// ===========================================================================
// CylindricalCS
// ===========================================================================

TEST(CylindricalCS, Dim) { EXPECT_EQ(cylindrical_cs().dim(), 3); }

TEST(CylindricalCS, NotOrthonormal)
{
    EXPECT_FALSE(cylindrical_cs().is_orthonormal());
}

TEST(CylindricalCS, CoordSymbols)
{
    auto const& cs = cylindrical_cs();
    EXPECT_EQ(cs.coord(0)->symbol(), "r");
    EXPECT_EQ(cs.coord(1)->symbol(), "theta");
    EXPECT_EQ(cs.coord(2)->symbol(), "z");
}

TEST(CylindricalCS, BasisRanks)
{
    auto const& cs = cylindrical_cs();
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(cs.basis(i)->rank(), 1);
}

// g_theta = r e_theta — TensorProduct(r_param, e_theta)
TEST(CylindricalCS, BasisTheta)
{
    auto const& cs = cylindrical_cs();
    auto* g_theta = dynamic_cast<TensorProduct*>(cs.basis(1));
    ASSERT_NE(g_theta, nullptr);
    EXPECT_EQ(g_theta->lhs(), cs.coord(0));   // r
    EXPECT_EQ(g_theta->rank(), 1);
}

// g^theta = (1/r) e_theta — TensorProduct(Pow(r,-1), e_theta)
TEST(CylindricalCS, CobasisTheta)
{
    auto const& cs = cylindrical_cs();
    auto* cob = dynamic_cast<TensorProduct*>(cs.cobasis(1));
    ASSERT_NE(cob, nullptr);
    auto* inv_r = dynamic_cast<Pow*>(cob->lhs());
    ASSERT_NE(inv_r, nullptr);
    EXPECT_EQ(inv_r->base(), cs.coord(0));
    EXPECT_EQ(inv_r->exponent(), Rational{-1});
    EXPECT_EQ(cob->rank(), 1);
}

TEST(CylindricalCS, CobasisRAndZAreNamedTensors)
{
    auto const& cs = cylindrical_cs();
    EXPECT_NE(dynamic_cast<NamedTensor*>(cs.cobasis(0)), nullptr);
    EXPECT_NE(dynamic_cast<NamedTensor*>(cs.cobasis(2)), nullptr);
}

TEST(CylindricalCS, MetricDiag)
{
    auto const& cs = cylindrical_cs();
    // g_00 = 1, g_22 = 1
    EXPECT_EQ(dynamic_cast<RationalConst*>(cs.metric(0, 0))->value(), Rational{1});
    EXPECT_EQ(dynamic_cast<RationalConst*>(cs.metric(2, 2))->value(), Rational{1});
    // g_11 = r^2
    auto* m11 = dynamic_cast<Pow*>(cs.metric(1, 1));
    ASSERT_NE(m11, nullptr);
    EXPECT_EQ(m11->base(), cs.coord(0));
    EXPECT_EQ(m11->exponent(), Rational{2});
    // off-diagonal = 0
    EXPECT_EQ(dynamic_cast<RationalConst*>(cs.metric(0, 1))->value(), Rational{0});
}

// ===========================================================================
// SphericalCS
// ===========================================================================

TEST(SphericalCS, Dim) { EXPECT_EQ(spherical_cs().dim(), 3); }

TEST(SphericalCS, CoordSymbols)
{
    auto const& cs = spherical_cs();
    EXPECT_EQ(cs.coord(0)->symbol(), "r");
    EXPECT_EQ(cs.coord(1)->symbol(), "theta");
    EXPECT_EQ(cs.coord(2)->symbol(), "phi");
}

// g^phi = (1/(r sin(theta))) e_phi — TensorProduct(Product(Pow(r,-1),
//                                                          Pow(sin(theta),-1)), e_phi)
TEST(SphericalCS, CobasisPhi)
{
    auto const& cs = spherical_cs();
    auto* cob = dynamic_cast<TensorProduct*>(cs.cobasis(2));
    ASSERT_NE(cob, nullptr);
    EXPECT_EQ(cob->rank(), 1);
    // Scalar factor is a Product
    EXPECT_NE(dynamic_cast<Product*>(cob->lhs()), nullptr);
}

TEST(SphericalCS, MetricPhiPhi)
{
    auto const& cs = spherical_cs();
    // g_22 = r^2 sin^2(theta) — Product(Pow(r,2), Pow(sin(theta),2))
    auto* m22 = dynamic_cast<Product*>(cs.metric(2, 2));
    ASSERT_NE(m22, nullptr);
    EXPECT_EQ(m22->rank(), 0);
}

// ===========================================================================
// grad — cylindrical gradient exit criterion
// ===========================================================================

// For f = r*theta + z (depends on all three coordinates), the gradient
// produces exactly 3 non-zero terms — matching the cylindrical formula.
// make_tensor_product optimises TensorProduct(0, v) → 0, so a field that
// does not depend on a coordinate contributes no term to the sum.
TEST(Grad, CylindricalGradStructure)
{
    auto rl = make_rl();
    auto const& cs = cylindrical_cs();

    auto* r     = cs.coord(0);
    auto* theta = cs.coord(1);
    auto* z     = cs.coord(2);

    // f = r*theta + z
    auto* f = make_sum(rl, {make_product(rl, r, theta), z});
    auto* g = grad(rl, f, cs);

    ASSERT_EQ(g->rank(), 1);
    auto* s = dynamic_cast<Sum*>(g);
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(static_cast<int>(s->terms().size()), 3);
}

// ∂(r*theta)/∂r = theta  →  first term is TensorProduct(theta, e_r)
TEST(Grad, CylindricalFirstTerm)
{
    auto rl = make_rl();
    auto const& cs = cylindrical_cs();

    auto* r     = cs.coord(0);
    auto* theta = cs.coord(1);
    auto* f     = make_product(rl, r, theta);
    auto* g     = grad(rl, f, cs);
    auto* s     = dynamic_cast<Sum*>(g);
    ASSERT_NE(s, nullptr);

    // First term: TensorProduct(theta_param, e_r)
    auto* t0 = dynamic_cast<TensorProduct*>(s->terms()[0]);
    ASSERT_NE(t0, nullptr);
    // The scalar coefficient is the theta parameter
    EXPECT_EQ(t0->lhs(), theta);
    // The vector part is e_r (the cobasis for r-direction)
    EXPECT_EQ(t0->rhs(), cs.cobasis(0));
}

// ∂(r*theta)/∂theta = r  →  second term is TensorProduct(r, cobasis(1))
// where cobasis(1) = TensorProduct(Pow(r,-1), e_theta)
TEST(Grad, CylindricalSecondTerm)
{
    auto rl = make_rl();
    auto const& cs = cylindrical_cs();

    auto* r     = cs.coord(0);
    auto* theta = cs.coord(1);
    auto* f     = make_product(rl, r, theta);
    auto* g     = grad(rl, f, cs);
    auto* s     = dynamic_cast<Sum*>(g);
    ASSERT_NE(s, nullptr);

    // Second term: TensorProduct(r_param, cobasis(1))
    auto* t1 = dynamic_cast<TensorProduct*>(s->terms()[1]);
    ASSERT_NE(t1, nullptr);
    EXPECT_EQ(t1->lhs(), r);
    EXPECT_EQ(t1->rhs(), cs.cobasis(1));
}

// grad(x) in WCS = e_x.  Zero-coefficient terms collapse so the result is
// just TensorProduct(1, e_x), not a Sum.
TEST(Grad, WCSGradOfX)
{
    auto rl = make_rl();
    auto const& cs = wcs();
    auto* x = cs.coord(0);
    auto* g = grad(rl, x, cs);
    EXPECT_EQ(g->rank(), 1);
    // Zero-coefficient terms are optimised away; single non-zero term is
    // returned unwrapped from Sum.
    auto* tp = dynamic_cast<TensorProduct*>(g);
    ASSERT_NE(tp, nullptr);
    auto* coeff = dynamic_cast<RationalConst*>(tp->lhs());
    ASSERT_NE(coeff, nullptr);
    EXPECT_EQ(coeff->value(), Rational{1});
    EXPECT_EQ(tp->rhs(), cs.cobasis(0));  // e_x in WCS
}

// grad(x+y+z) in WCS = e_x + e_y + e_z (all three non-zero terms → Sum)
TEST(Grad, WCSGradOfXYZ)
{
    auto rl = make_rl();
    auto const& cs = wcs();
    auto* f = make_sum(rl, {cs.coord(0), cs.coord(1), cs.coord(2)});
    auto* g = grad(rl, f, cs);
    EXPECT_EQ(g->rank(), 1);
    auto* s = dynamic_cast<Sum*>(g);
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(static_cast<int>(s->terms().size()), 3);
}

TEST(Grad, NoCoordThrows)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* b = make_named_tensor(rl, "b", 1, {});
    auto* c = make_named_tensor(rl, "c", 1, {});
    auto cs = make_direct_basis_cs(a, b, c);
    auto* f = make_rational(rl, Rational{1});
    EXPECT_THROW(grad(rl, f, *cs), std::invalid_argument);
}
