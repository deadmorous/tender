#include <tender/integral.hpp>

#include <gtest/gtest.h>

using namespace tender;

static ResourceList make_rl()
{
    return ResourceList{};
}

// ===========================================================================
// Domain types
// ===========================================================================

TEST(SurfaceDomain, NameAndNormal)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* s = make_surface_domain(rl, "S", n);
    EXPECT_EQ(s->name(), "S");
    EXPECT_EQ(s->normal(), n);
}

TEST(SurfaceDomain, BoundaryIsNull)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* s = make_surface_domain(rl, "S", n);
    EXPECT_EQ(s->boundary(), nullptr);
}

TEST(SurfaceDomain, MeasureLatex)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* s = make_surface_domain(rl, "S", n);
    EXPECT_EQ(s->measure_latex(), "\\mathrm{d}S");
}

TEST(VolumeDomain, NameAndNormal)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    EXPECT_EQ(V->name(), "V");
    EXPECT_EQ(V->outward_normal(), n);
}

TEST(VolumeDomain, BoundaryIsSurface)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* bdy = dynamic_cast<SurfaceDomain*>(V->boundary());
    ASSERT_NE(bdy, nullptr);
    EXPECT_EQ(bdy->normal(), n);
}

TEST(VolumeDomain, BoundaryNameIsPartialV)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    EXPECT_EQ(V->boundary()->name(), "\\partial V");
}

TEST(VolumeDomain, MeasureLatex)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    EXPECT_EQ(V->measure_latex(), "\\mathrm{d}V");
}

// ===========================================================================
// Gradient node
// ===========================================================================

TEST(GradientNode, RankIsArgPlusOne)
{
    auto rl = make_rl();
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* g = make_gradient(rl, f);
    EXPECT_EQ(g->rank(), 1);
}

TEST(GradientNode, RankForVectorArg)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* g = make_gradient(rl, v);
    EXPECT_EQ(g->rank(), 2);
}

TEST(GradientNode, ArgAccessor)
{
    auto rl = make_rl();
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* g = dynamic_cast<Gradient*>(make_gradient(rl, f));
    ASSERT_NE(g, nullptr);
    EXPECT_EQ(g->arg(), f);
}

TEST(GradientNode, LatexScalar)
{
    auto rl = make_rl();
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* g = make_gradient(rl, f);
    EXPECT_EQ(g->latex(), "\\nabla f");
}

TEST(GradientNode, Python)
{
    auto rl = make_rl();
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* g = make_gradient(rl, f);
    EXPECT_NE(g->python().find("gradient"), std::string::npos);
}

// ===========================================================================
// Divergence node
// ===========================================================================

TEST(DivergenceNode, RankIsArgMinusOne)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* d = make_divergence(rl, v);
    EXPECT_EQ(d->rank(), 0);
}

TEST(DivergenceNode, RankForTensor)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* d = make_divergence(rl, A);
    EXPECT_EQ(d->rank(), 1);
}

TEST(DivergenceNode, ScalarArgThrows)
{
    auto rl = make_rl();
    auto* f = make_named_tensor(rl, "f", 0, {});
    EXPECT_THROW(make_divergence(rl, f), std::invalid_argument);
}

TEST(DivergenceNode, ArgAccessor)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* d = dynamic_cast<Divergence*>(make_divergence(rl, v));
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->arg(), v);
}

TEST(DivergenceNode, Latex)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* d = make_divergence(rl, v);
    EXPECT_NE(d->latex().find("\\nabla \\cdot"), std::string::npos);
}

TEST(DivergenceNode, Python)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* d = make_divergence(rl, v);
    EXPECT_NE(d->python().find("divergence"), std::string::npos);
}

// ===========================================================================
// Rotor node
// ===========================================================================

TEST(RotorNode, RankMatchesArg)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* r = make_rotor(rl, v);
    EXPECT_EQ(r->rank(), 1);
}

TEST(RotorNode, RankForTensor)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* r = make_rotor(rl, A);
    EXPECT_EQ(r->rank(), 2);
}

TEST(RotorNode, ScalarArgThrows)
{
    auto rl = make_rl();
    auto* f = make_named_tensor(rl, "f", 0, {});
    EXPECT_THROW(make_rotor(rl, f), std::invalid_argument);
}

TEST(RotorNode, ArgAccessor)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* r = make_rotor(rl, v);
    auto* rr = dynamic_cast<Rotor*>(r);
    ASSERT_NE(rr, nullptr);
    EXPECT_EQ(rr->arg(), v);
}

TEST(RotorNode, Latex)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* r = make_rotor(rl, v);
    EXPECT_NE(r->latex().find("times"), std::string::npos);
}

TEST(RotorNode, Python)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* r = make_rotor(rl, v);
    EXPECT_NE(r->python().find("rotor"), std::string::npos);
}

// ===========================================================================
// Integral node
// ===========================================================================

TEST(IntegralNode, RankMatchesIntegrand)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* I = make_integral(rl, V, f);
    EXPECT_EQ(I->rank(), 0);
}

TEST(IntegralNode, VectorIntegrandRank)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* g = make_named_tensor(rl, "g", 1, {});
    auto* I = make_integral(rl, V, g);
    EXPECT_EQ(I->rank(), 1);
}

TEST(IntegralNode, DomainAndIntegrandAccessors)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* I = dynamic_cast<Integral*>(make_integral(rl, V, f));
    ASSERT_NE(I, nullptr);
    EXPECT_EQ(I->domain(), V);
    EXPECT_EQ(I->integrand(), f);
}

TEST(IntegralNode, LatexContainsDomainName)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* I = make_integral(rl, V, f);
    auto tex = I->latex();
    EXPECT_NE(tex.find("\\int_{V}"), std::string::npos);
    EXPECT_NE(tex.find("\\mathrm{d}V"), std::string::npos);
}

TEST(IntegralNode, SurfaceLatex)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* dV = V->surface_boundary();
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* I = make_integral(rl, dV, f);
    auto tex = I->latex();
    EXPECT_NE(tex.find("\\partial V"), std::string::npos);
    EXPECT_NE(tex.find("\\mathrm{d}S"), std::string::npos);
}

TEST(IntegralNode, Python)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* I = make_integral(rl, V, f);
    EXPECT_NE(I->python().find("integral"), std::string::npos);
}

// ===========================================================================
// apply_integration_by_parts_step
// ===========================================================================

// Pattern: Integral(V, A:(∇B)) → Sum(Integral(∂V,(An)·B), -Integral(V,(∇·A)·B))
TEST(IBP, SingleIntegralMatches)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* dV = V->surface_boundary();

    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* B = make_named_tensor(rl, "B", 1, {});
    auto* grad_B = make_gradient(rl, B);
    auto* integrand = make_double_contract(rl, A, grad_B);
    auto* integ = make_integral(rl, V, integrand);

    auto step = apply_integration_by_parts_step(V);
    auto* result = step.apply(rl, State{integ}).expr();

    // Result is a Sum of two terms
    auto* s = dynamic_cast<Sum*>(result);
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(static_cast<int>(s->terms().size()), 2);

    // First term: Integral(∂V, Contract(Contract(A,n), B))
    auto* t0 = dynamic_cast<Integral*>(s->terms()[0]);
    ASSERT_NE(t0, nullptr);
    EXPECT_EQ(t0->domain(), dV);
    auto* inner0 = dynamic_cast<Contract*>(t0->integrand());
    ASSERT_NE(inner0, nullptr);
    EXPECT_EQ(inner0->rank(), 0);

    // Second term: Scale(-1, Integral(V, Contract(Divergence(A), B)))
    auto* sc = dynamic_cast<Scale*>(s->terms()[1]);
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->coeff(), Rational{-1});
    auto* t1 = dynamic_cast<Integral*>(sc->expr());
    ASSERT_NE(t1, nullptr);
    EXPECT_EQ(t1->domain(), V);
    // Integrand contains Divergence(A)
    auto* inner1 = dynamic_cast<Contract*>(t1->integrand());
    ASSERT_NE(inner1, nullptr);
    EXPECT_NE(dynamic_cast<Divergence*>(inner1->lhs()), nullptr);
}

TEST(IBP, DivergenceArgIsA)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* B = make_named_tensor(rl, "B", 1, {});
    auto* integ =
        make_integral(rl, V, make_double_contract(rl, A, make_gradient(rl, B)));

    auto step = apply_integration_by_parts_step(V);
    auto* result = step.apply(rl, State{integ}).expr();

    auto* s = dynamic_cast<Sum*>(result);
    ASSERT_NE(s, nullptr);
    auto* sc = dynamic_cast<Scale*>(s->terms()[1]);
    ASSERT_NE(sc, nullptr);
    auto* vol_integ = dynamic_cast<Integral*>(sc->expr());
    ASSERT_NE(vol_integ, nullptr);
    auto* co = dynamic_cast<Contract*>(vol_integ->integrand());
    ASSERT_NE(co, nullptr);
    auto* div_A = dynamic_cast<Divergence*>(co->lhs());
    ASSERT_NE(div_A, nullptr);
    EXPECT_EQ(div_A->arg(), A);
}

TEST(IBP, NonMatchingIntegralIsUnchanged)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* integ = make_integral(rl, V, f); // plain integrand, not DC(A,
                                           // Grad(B))

    auto step = apply_integration_by_parts_step(V);
    auto* result = step.apply(rl, State{integ}).expr();
    EXPECT_EQ(result, integ); // unchanged
}

TEST(IBP, AppliesToMatchingTermsInSum)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* B = make_named_tensor(rl, "B", 1, {});
    auto* f = make_named_tensor(rl, "f", 0, {});

    // Sum of IBP-applicable and non-applicable integrals
    auto* ibp_term =
        make_integral(rl, V, make_double_contract(rl, A, make_gradient(rl, B)));
    auto* plain_term = make_integral(rl, V, f);
    auto* sum_expr = make_sum(rl, {ibp_term, plain_term});

    auto step = apply_integration_by_parts_step(V);
    auto* result = step.apply(rl, State{sum_expr}).expr();

    // Result is a Sum; the IBP term expanded to 2 sub-terms + 1 plain term = 3
    auto* s = dynamic_cast<Sum*>(result);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(static_cast<int>(s->terms().size()), 3);
}

// ===========================================================================
// apply_divergence_theorem_step
// ===========================================================================

TEST(DivThm, RewritesDivergenceIntegral)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* dV = V->surface_boundary();

    auto* A = make_named_tensor(rl, "A", 1, {});
    auto* integ = make_integral(rl, V, make_divergence(rl, A));

    auto step = apply_divergence_theorem_step(V);
    auto* result = step.apply(rl, State{integ}).expr();

    // Result: Integral(∂V, A·n)
    auto* surf_integ = dynamic_cast<Integral*>(result);
    ASSERT_NE(surf_integ, nullptr);
    EXPECT_EQ(surf_integ->domain(), dV);
    auto* co = dynamic_cast<Contract*>(surf_integ->integrand());
    ASSERT_NE(co, nullptr);
    EXPECT_EQ(co->lhs(), A);
    EXPECT_EQ(co->rhs(), n);
}

TEST(DivThm, NonDivergenceIntegralUnchanged)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* integ = make_integral(rl, V, f);

    auto step = apply_divergence_theorem_step(V);
    auto* result = step.apply(rl, State{integ}).expr();
    EXPECT_EQ(result, integ);
}

// ===========================================================================
// localize_step
// ===========================================================================

TEST(Localize, StripsIntegralWrapper)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* integ = make_integral(rl, V, f);

    auto step = localize_step(V);
    auto* result = step.apply(rl, State{integ}).expr();
    EXPECT_EQ(result, f);
}

TEST(Localize, OnlyStripsTargetDomain)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* dV = V->surface_boundary();
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* g = make_named_tensor(rl, "g", 0, {});

    // Sum of Integral over V and Integral over ∂V
    auto* vol_term = make_integral(rl, V, f);
    auto* surf_term = make_integral(rl, dV, g);
    auto* sum_expr = make_sum(rl, {vol_term, surf_term});

    // Localize over V: strips vol_term, leaves surf_term
    auto step = localize_step(V);
    auto* result = step.apply(rl, State{sum_expr}).expr();

    // Result is Sum(f, Integral(∂V, g))
    auto* s = dynamic_cast<Sum*>(result);
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(static_cast<int>(s->terms().size()), 2);
    EXPECT_EQ(s->terms()[0], f); // f extracted from Integral(V, f)
    EXPECT_NE(
        dynamic_cast<Integral*>(s->terms()[1]),
        nullptr); // Integral(∂V, g) untouched
}

TEST(Localize, StepName)
{
    auto rl = make_rl();
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto step = localize_step(V);
    EXPECT_EQ(step.name(), "localize(V)");
}
