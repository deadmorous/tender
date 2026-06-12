// PVW feasibility example — Phase 11 exit criterion.
//
// Demonstrates a complete symbolic derivation of the strong-form equilibrium
// equation from the Principle of Virtual Work (d'Alembert form):
//
//   ∫_V σ:∇δu dV  =  ∫_V f·δu dV  +  ∫_∂V t·δu dS  −  ∫_V ρü·δu dV
//
// Steps:
//   1. Write the PVW as a single expression (LHS − RHS = 0).
//   2. Apply integration by parts to ∫_V σ:∇δu:
//        → ∫_∂V (σn)·δu dS  −  ∫_V (∇·σ)·δu dV
//   3. Rearrange (grouping surface and volume terms via expand + simplify).
//   4. Localize over V to extract the volume integrand → equilibrium equation.
//   5. Localize over ∂V to extract the surface integrand → traction BC.

#include <tender/integral.hpp>

#include <gtest/gtest.h>

using namespace tender;

static ResourceList make_rl()
{
    return ResourceList{};
}

// Helper: find the first Integral node in a Sum that is over `domain`.
static Integral* first_integral_over(Expr* e, Domain* domain)
{
    if (auto* integ = dynamic_cast<Integral*>(e))
        return integ->domain() == domain ? integ : nullptr;
    if (auto* s = dynamic_cast<Sum*>(e))
        for (auto* t: s->terms())
            if (auto* found = first_integral_over(t, domain))
                return found;
    if (auto* sc = dynamic_cast<Scale*>(e))
        return first_integral_over(sc->expr(), domain);
    return nullptr;
}

// ===========================================================================
// End-to-end derivation
// ===========================================================================

TEST(PVW, IBPTransformationStructure)
{
    auto rl = make_rl();

    // Named tensors
    auto* sigma = make_named_tensor(rl, "sigma", 2, {});
    auto* delta_u = make_named_tensor(rl, "delta_u", 1, {});
    auto* f_body = make_named_tensor(rl, "f", 1, {});
    auto* t_trac = make_named_tensor(rl, "t", 1, {});
    auto* rho_udd = make_named_tensor(rl, "rho_udd", 1, {});
    auto* n = make_named_tensor(rl, "n", 1, {});

    // Domains
    auto* V = make_volume_domain(rl, "V", n);
    auto* dV = V->surface_boundary();

    // PVW (rearranged so that LHS = 0 is implied):
    //   ∫_V σ:∇δu dV  −  ∫_V f·δu dV  −  ∫_∂V t·δu dS  +  ∫_V ρü·δu dV
    auto* pvw = make_sum(
        rl,
        {make_integral(
             rl, V, make_double_contract(rl, sigma, make_gradient(rl, delta_u))),
         make_scale(
             rl,
             Rational{-1},
             make_integral(rl, V, make_contract(rl, f_body, delta_u))),
         make_scale(
             rl,
             Rational{-1},
             make_integral(rl, dV, make_contract(rl, t_trac, delta_u))),
         make_integral(rl, V, make_contract(rl, rho_udd, delta_u))});

    State initial{pvw};

    // Step 1: integration by parts on the ∫_V σ:∇δu term.
    auto history =
        Derivation{{apply_integration_by_parts_step(V)}}.apply(rl, initial);

    auto* after_ibp = history[1].expr();
    ASSERT_EQ(static_cast<int>(history.size()), 2);
    EXPECT_EQ(after_ibp->rank(), 0);

    // The result must be a Sum.
    auto* s = dynamic_cast<Sum*>(after_ibp);
    ASSERT_NE(s, nullptr);

    // After IBP, the expression contains at least one Integral over ∂V
    // (the surface term (σn)·δu) and at least one Integral over V
    // containing Divergence(σ).
    auto* surf_integ = first_integral_over(after_ibp, dV);
    ASSERT_NE(surf_integ, nullptr) << "no surface integral found after IBP";
    EXPECT_EQ(surf_integ->domain(), dV);

    auto* vol_ibp = first_integral_over(after_ibp, V);
    ASSERT_NE(vol_ibp, nullptr) << "no volume integral found after IBP";

    // The IBP volume integral's integrand contains Divergence(sigma).
    // It is Contract(Divergence(sigma), delta_u), rank-0.
    EXPECT_EQ(vol_ibp->integrand()->rank(), 0);
}

TEST(PVW, IBPSurfaceTermContainsNormal)
{
    auto rl = make_rl();

    auto* sigma = make_named_tensor(rl, "sigma", 2, {});
    auto* delta_u = make_named_tensor(rl, "delta_u", 1, {});
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* dV = V->surface_boundary();

    // Just the IBP-applicable term: ∫_V σ:∇δu
    auto* ibp_term = make_integral(
        rl, V, make_double_contract(rl, sigma, make_gradient(rl, delta_u)));

    auto step = apply_integration_by_parts_step(V);
    auto* result = step.apply(rl, State{ibp_term}).expr();

    // Surface integral: ∫_∂V (σn)·δu — its integrand is Contract(Contract(σ,n),
    // δu)
    auto* surf = first_integral_over(result, dV);
    ASSERT_NE(surf, nullptr);

    auto* outer_co = dynamic_cast<Contract*>(surf->integrand());
    ASSERT_NE(outer_co, nullptr) << "surface integrand is not a Contract";
    EXPECT_EQ(outer_co->rhs(), delta_u);

    auto* inner_co = dynamic_cast<Contract*>(outer_co->lhs());
    ASSERT_NE(inner_co, nullptr);
    EXPECT_EQ(inner_co->lhs(), sigma);
    EXPECT_EQ(inner_co->rhs(), n);
}

TEST(PVW, LocalizeVolumeGivesEquilibrium)
{
    auto rl = make_rl();

    auto* sigma = make_named_tensor(rl, "sigma", 2, {});
    auto* delta_u = make_named_tensor(rl, "delta_u", 1, {});
    auto* f_body = make_named_tensor(rl, "f", 1, {});
    auto* rho_udd = make_named_tensor(rl, "rho_udd", 1, {});
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* dV = V->surface_boundary();

    // Construct what the expression looks like after IBP: a sum of volume
    // and surface integrals, all with δu as the test function.
    //
    // Volume part:  −∫_V (∇·σ)·δu dV − ∫_V f·δu dV + ∫_V ρü·δu dV
    //             = ∫_V (−∇·σ − f + ρü)·δu dV
    //
    // We represent the un-simplified version (three separate volume integrals)
    // and apply localize to each.
    auto* post_ibp = make_sum(
        rl,
        {// surface term
         make_integral(
             rl, dV, make_contract(rl, make_contract(rl, sigma, n), delta_u)),
         // volume terms from IBP
         make_scale(
             rl,
             Rational{-1},
             make_integral(
                 rl, V, make_contract(rl, make_divergence(rl, sigma), delta_u))),
         make_scale(
             rl,
             Rational{-1},
             make_integral(rl, V, make_contract(rl, f_body, delta_u))),
         make_integral(rl, V, make_contract(rl, rho_udd, delta_u))});

    // Localize over V: extracts volume integrands and discards the ∂V term
    // (test functions with support inside V make the surface integral vanish).
    auto step = localize_step(V);
    auto* localized = step.apply(rl, State{post_ibp}).expr();

    // Result is a Sum of the three volume integrands only — no surface term.
    auto* s = dynamic_cast<Sum*>(localized);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(static_cast<int>(s->terms().size()), 3);

    // The surface integral over ∂V must NOT be present.
    EXPECT_EQ(first_integral_over(localized, dV), nullptr);

    // One of the exposed terms must contain Divergence(sigma).
    bool found_div_sigma = false;
    for (auto* term: s->terms())
    {
        Expr* inner = term;
        if (auto* sc = dynamic_cast<Scale*>(inner))
            inner = sc->expr();
        if (auto* co = dynamic_cast<Contract*>(inner))
            if (dynamic_cast<Divergence*>(co->lhs()))
                found_div_sigma = true;
    }
    EXPECT_TRUE(found_div_sigma)
        << "expected Divergence(sigma) in localized volume integrands";
}

TEST(PVW, LocalizeSurfaceGivesTractionBC)
{
    auto rl = make_rl();

    auto* sigma = make_named_tensor(rl, "sigma", 2, {});
    auto* delta_u = make_named_tensor(rl, "delta_u", 1, {});
    auto* t_trac = make_named_tensor(rl, "t", 1, {});
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* dV = V->surface_boundary();

    // Surface part of PVW after IBP (and traction term moved to RHS):
    //   ∫_∂V (σn − t)·δu dS  [should equal 0]
    auto* sigma_n = make_contract(rl, sigma, n);
    auto* sigma_n_minus_t =
        make_sum(rl, {sigma_n, make_scale(rl, Rational{-1}, t_trac)});
    auto* surf_expr =
        make_integral(rl, dV, make_contract(rl, sigma_n_minus_t, delta_u));

    // Localize over ∂V: gives the traction BC integrand
    auto step = localize_step(dV);
    auto* bc = step.apply(rl, State{surf_expr}).expr();

    // Result: Contract(Sum(σn, -t), δu)  — the traction BC coefficient
    EXPECT_EQ(bc->rank(), 0);
    auto* co = dynamic_cast<Contract*>(bc);
    ASSERT_NE(co, nullptr);
    EXPECT_EQ(co->rhs(), delta_u);
    // LHS of outer contract is the σn − t expression
    EXPECT_NE(dynamic_cast<Sum*>(co->lhs()), nullptr);
}

// ===========================================================================
// Derivation history captures all steps
// ===========================================================================

TEST(PVW, DerivationHistoryLength)
{
    auto rl = make_rl();

    auto* sigma = make_named_tensor(rl, "sigma", 2, {});
    auto* delta_u = make_named_tensor(rl, "delta_u", 1, {});
    auto* f_body = make_named_tensor(rl, "f", 1, {});
    auto* t_trac = make_named_tensor(rl, "t", 1, {});
    auto* rho_udd = make_named_tensor(rl, "rho_udd", 1, {});
    auto* n = make_named_tensor(rl, "n", 1, {});
    auto* V = make_volume_domain(rl, "V", n);
    auto* dV = V->surface_boundary();

    auto* pvw = make_sum(
        rl,
        {make_integral(
             rl, V, make_double_contract(rl, sigma, make_gradient(rl, delta_u))),
         make_scale(
             rl,
             Rational{-1},
             make_integral(rl, V, make_contract(rl, f_body, delta_u))),
         make_scale(
             rl,
             Rational{-1},
             make_integral(rl, dV, make_contract(rl, t_trac, delta_u))),
         make_integral(rl, V, make_contract(rl, rho_udd, delta_u))});

    // Full derivation: IBP → localize V
    auto history =
        Derivation{{apply_integration_by_parts_step(V), localize_step(V)}}
            .apply(rl, State{pvw});

    // 3 states: initial, after IBP, after localize
    ASSERT_EQ(static_cast<int>(history.size()), 3);

    // Step names are recorded correctly.
    EXPECT_EQ(history[1].label(), "ibp(V)");
    EXPECT_EQ(history[2].label(), "localize(V)");

    // Final expression rank is 0 (scalar equation)
    EXPECT_EQ(history.back().expr()->rank(), 0);
}
