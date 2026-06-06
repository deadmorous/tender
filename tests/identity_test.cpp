#include <tender/identity.hpp>

#include <gtest/gtest.h>

using namespace tender;

static ResourceList make_rl()
{
    return ResourceList{};
}

// ===========================================================================
// PatternVar — construction and accessors
// ===========================================================================

TEST(PatternVar, SymbolAccessor)
{
    auto rl = make_rl();
    auto* pv = make_pattern_var(rl, "alpha");
    EXPECT_EQ(pv->symbol(), "alpha");
}

TEST(PatternVar, DefaultRankIsMinusOne)
{
    auto rl = make_rl();
    auto* pv = make_pattern_var(rl, "x");
    EXPECT_EQ(pv->rank(), -1);
}

TEST(PatternVar, ConstrainRankFluent)
{
    auto rl = make_rl();
    auto* pv = make_pattern_var(rl, "v")->constrain_rank(1);
    EXPECT_EQ(pv->rank(), 1);
    EXPECT_EQ(pv->constraints().required_rank, 1);
}

TEST(PatternVar, ConstrainSymmetricFluent)
{
    auto rl = make_rl();
    auto* pv = make_pattern_var(rl, "S")->constrain_rank(2)->constrain_symmetric();
    EXPECT_TRUE(pv->constraints().symmetric);
    EXPECT_EQ(pv->rank(), 2);
}

TEST(PatternVar, ConstrainSkewSymmetric)
{
    auto rl = make_rl();
    auto* pv = make_pattern_var(rl, "W")->constrain_rank(2)->constrain_skew_symmetric();
    EXPECT_TRUE(pv->constraints().skew_symmetric);
}

TEST(PatternVar, LatexIsSymbol)
{
    auto rl = make_rl();
    auto* pv = make_pattern_var(rl, "beta");
    EXPECT_EQ(pv->latex(), "beta");
}

TEST(PatternVar, PythonRendering)
{
    auto rl = make_rl();
    auto* pv = make_pattern_var(rl, "gamma");
    EXPECT_EQ(pv->python(), "pattern_var('gamma')");
}

// ===========================================================================
// Identity — construction and accessors
// ===========================================================================

TEST(IdentityClass, NameAccessor)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* b = make_named_tensor(rl, "b", 1, {});
    Identity id{"my_id", a, b};
    EXPECT_EQ(id.name(), "my_id");
}

TEST(IdentityClass, LhsRhsAccessors)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* b = make_named_tensor(rl, "b", 1, {});
    Identity id{"test", a, b};
    EXPECT_EQ(id.lhs(), a);
    EXPECT_EQ(id.rhs(), b);
}

TEST(IdentityClass, FromDerivation)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* e = make_sin(rl, t);
    State initial{e};
    auto history = Derivation{{diff_step(t)}}.apply(rl, initial);

    auto id = Identity::from_derivation("sin_diff", history);
    EXPECT_EQ(id.name(), "sin_diff");
    EXPECT_EQ(id.lhs(), history.front().expr());
    EXPECT_EQ(id.rhs(), history.back().expr());
}

TEST(IdentityClass, FromDerivationTooShortThrows)
{
    auto rl = make_rl();
    std::vector<State> history{State{make_rational(rl, Rational{1})}};
    EXPECT_THROW(Identity::from_derivation("x", history), std::invalid_argument);
}

// ===========================================================================
// apply_identity — substitution correctness
// ===========================================================================

TEST(ApplyIdentity, SimpleScalarSubstitution)
{
    // Identity: pv → pv + 1  (trivial renaming test)
    auto rl = make_rl();
    auto* pv = make_pattern_var(rl, "x")->constrain_rank(0);
    auto* rhs_pat = make_sum(rl, {pv, make_rational(rl, Rational{1})});
    Identity id{"shift", pv, rhs_pat};

    auto* actual = make_symbolic_var(rl, "t");
    PatternMapping mapping = {{pv, actual}};
    auto step = apply_identity(id, mapping);
    EXPECT_EQ(step.name(), "apply(shift)");

    auto* any_expr = make_rational(rl, Rational{0});
    auto result = step.apply(rl, State{any_expr});
    // result.expr() should be Sum(t, 1)
    EXPECT_NE(dynamic_cast<Sum*>(result.expr()), nullptr);
    EXPECT_EQ(result.expr()->rank(), 0);
}

TEST(ApplyIdentity, RankConstraintViolationThrows)
{
    auto rl = make_rl();
    auto* pv = make_pattern_var(rl, "v")->constrain_rank(1);
    Identity id{"id", pv, pv};

    auto* scalar = make_rational(rl, Rational{3});  // rank-0
    PatternMapping mapping = {{pv, scalar}};
    auto step = apply_identity(id, mapping);

    auto* any_expr = make_rational(rl, Rational{0});
    State dummy{any_expr};
    EXPECT_THROW([[maybe_unused]] auto _ = step.apply(rl, dummy), std::invalid_argument);
}

TEST(ApplyIdentity, SubstitutionInsideContract)
{
    // Identity: contract(a, b) → scalar
    // Substituted: a→u, b→v  ⟹ result is Contract(u, v)
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a")->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b")->constrain_rank(1);
    auto* rhs_pat = make_contract(rl, a, b);
    Identity id{"dot", a, rhs_pat};

    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    PatternMapping mapping = {{a, u}, {b, v}};
    auto step = apply_identity(id, mapping);

    auto* any_expr = make_rational(rl, Rational{0});
    auto result = step.apply(rl, State{any_expr});
    EXPECT_NE(dynamic_cast<Contract*>(result.expr()), nullptr);
    EXPECT_EQ(result.expr()->rank(), 0);
}

// ===========================================================================
// BAC-CAB identity — exit criterion for Phase 8
// ===========================================================================

// a × (b × c) = b(a·c) − c(a·b)   for rank-1 a, b, c

TEST(BacCab, IdentityCanBeStated)
{
    auto rl = make_rl();

    auto* a = make_pattern_var(rl, "a")->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b")->constrain_rank(1);
    auto* c = make_pattern_var(rl, "c")->constrain_rank(1);

    // LHS: a × (b × c) — nested CrossProduct, bypass factory guard
    auto* bc = rl.make<CrossProduct>(b, c);
    auto* lhs_pat = rl.make<CrossProduct>(a, bc);
    EXPECT_EQ(lhs_pat->rank(), 1);

    // RHS: b(a·c) − c(a·b)
    auto* b_ac = make_tensor_product(rl, make_contract(rl, a, c), b);
    auto* c_ab = make_tensor_product(rl, make_contract(rl, a, b), c);
    auto* rhs_pat = make_sum(rl, {b_ac, make_scale(rl, Rational{-1}, c_ab)});
    EXPECT_EQ(rhs_pat->rank(), 1);

    Identity bac_cab{"bac_cab", lhs_pat, rhs_pat};
    EXPECT_EQ(bac_cab.name(), "bac_cab");
    EXPECT_EQ(bac_cab.lhs()->rank(), 1);
    EXPECT_EQ(bac_cab.rhs()->rank(), 1);
}

TEST(BacCab, ManualApplication)
{
    auto rl = make_rl();

    // Build identity
    auto* a = make_pattern_var(rl, "a")->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b")->constrain_rank(1);
    auto* c = make_pattern_var(rl, "c")->constrain_rank(1);

    auto* bc_pat = rl.make<CrossProduct>(b, c);
    auto* lhs_pat = rl.make<CrossProduct>(a, bc_pat);
    auto* b_ac = make_tensor_product(rl, make_contract(rl, a, c), b);
    auto* c_ab = make_tensor_product(rl, make_contract(rl, a, b), c);
    auto* rhs_pat = make_sum(rl, {b_ac, make_scale(rl, Rational{-1}, c_ab)});

    Identity bac_cab{"bac_cab", lhs_pat, rhs_pat};

    // Actual rank-1 vectors
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});

    // Apply with manual mapping: a→u, b→v, c→w
    PatternMapping mapping = {{a, u}, {b, v}, {c, w}};
    auto step = apply_identity(bac_cab, mapping);
    EXPECT_EQ(step.name(), "apply(bac_cab)");

    // Initial state: u × (v × w) — bypass factory guard in test code
    auto* vw = rl.make<CrossProduct>(v, w);
    auto* uvw = rl.make<CrossProduct>(u, vw);
    State initial{uvw};

    auto history = Derivation{{step}}.apply(rl, initial);
    ASSERT_EQ(history.size(), 2u);

    // Result: v(u·w) − w(u·v) — a Sum of TensorProduct terms, rank 1
    auto* result = history[1].expr();
    EXPECT_EQ(result->rank(), 1);
    EXPECT_EQ(history[1].label(), "apply(bac_cab)");

    auto* s = dynamic_cast<Sum*>(result);
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(s->terms().size(), 2u);

    // First term: (u·w) v  — TensorProduct with rank-0 lhs
    auto* t0 = dynamic_cast<TensorProduct*>(s->terms()[0]);
    ASSERT_NE(t0, nullptr);
    EXPECT_EQ(t0->lhs()->rank(), 0);
    EXPECT_EQ(t0->rhs()->rank(), 1);

    // Second term: Scale(-1, (u·v) w)
    auto* t1 = dynamic_cast<Scale*>(s->terms()[1]);
    ASSERT_NE(t1, nullptr);
    EXPECT_EQ(t1->coeff(), Rational{-1});
    EXPECT_NE(dynamic_cast<TensorProduct*>(t1->expr()), nullptr);
}

TEST(BacCab, LatexOutput)
{
    auto rl = make_rl();

    auto* a = make_pattern_var(rl, "a")->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b")->constrain_rank(1);
    auto* c = make_pattern_var(rl, "c")->constrain_rank(1);

    auto* b_ac = make_tensor_product(rl, make_contract(rl, a, c), b);
    auto* c_ab = make_tensor_product(rl, make_contract(rl, a, b), c);
    auto* rhs_pat = make_sum(rl, {b_ac, make_scale(rl, Rational{-1}, c_ab)});

    // RHS LaTeX should contain the pattern var symbols
    std::string ltx = rhs_pat->latex();
    EXPECT_NE(ltx.find('a'), std::string::npos);
    EXPECT_NE(ltx.find('b'), std::string::npos);
    EXPECT_NE(ltx.find('c'), std::string::npos);
    // Scalar mult: no \otimes in output
    EXPECT_EQ(ltx.find("\\otimes"), std::string::npos);
}

TEST(BacCab, FromDerivationRoundtrip)
{
    // Build the identity via from_derivation and verify lhs/rhs
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a")->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b")->constrain_rank(1);
    auto* c = make_pattern_var(rl, "c")->constrain_rank(1);

    auto* bc_pat = rl.make<CrossProduct>(b, c);
    auto* lhs_pat = rl.make<CrossProduct>(a, bc_pat);
    auto* b_ac = make_tensor_product(rl, make_contract(rl, a, c), b);
    auto* c_ab = make_tensor_product(rl, make_contract(rl, a, b), c);
    auto* rhs_pat = make_sum(rl, {b_ac, make_scale(rl, Rational{-1}, c_ab)});

    State s0{lhs_pat};
    State s1{rhs_pat, "bac_cab"};
    auto id = Identity::from_derivation("bac_cab", {s0, s1});

    EXPECT_EQ(id.lhs(), lhs_pat);
    EXPECT_EQ(id.rhs(), rhs_pat);
}

// ===========================================================================
// apply_identity — substitute_pattern coverage for additional node types
// ===========================================================================

TEST(SubstitutePattern, DoubleContractInRhs)
{
    // Identity: A:B → A:B  (trivial, tests DoubleContract traversal)
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "A")->constrain_rank(2);
    auto* b = make_pattern_var(rl, "B")->constrain_rank(2);
    auto* rhs_pat = make_double_contract(rl, a, b);
    Identity id{"dc_id", a, rhs_pat};

    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* B = make_named_tensor(rl, "B", 2, {});
    PatternMapping mapping = {{a, A}, {b, B}};
    auto step = apply_identity(id, mapping);

    auto history = Derivation{{step}}.apply(rl, State{A});
    auto* dc = dynamic_cast<DoubleContract*>(history[1].expr());
    ASSERT_NE(dc, nullptr);
    EXPECT_EQ(dc->lhs(), A);
    EXPECT_EQ(dc->rhs(), B);
}

TEST(SubstitutePattern, CrossProductInRhs)
{
    // Identity: a → a×b  (tests CrossProduct path in substitute_pattern)
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a")->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b")->constrain_rank(1);
    // RHS: a×b — build with rl.make to bypass guard (a and b are PatternVars, not
    // CrossProducts, so make_cross_product would also work here)
    auto* rhs_pat = make_cross_product(rl, a, b);
    Identity id{"cp_id", a, rhs_pat};

    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    PatternMapping mapping = {{a, u}, {b, v}};
    auto step = apply_identity(id, mapping);

    auto history = Derivation{{step}}.apply(rl, State{u});
    auto* cp = dynamic_cast<CrossProduct*>(history[1].expr());
    ASSERT_NE(cp, nullptr);
    EXPECT_EQ(cp->lhs(), u);
    EXPECT_EQ(cp->rhs(), v);
}

TEST(SubstitutePattern, FunctionApplyInRhs)
{
    auto rl = make_rl();
    auto* x = make_pattern_var(rl, "x")->constrain_rank(0);
    auto* rhs_pat = make_sin(rl, x);
    Identity id{"sin_id", x, rhs_pat};

    auto* t = make_parameter(rl, "t");
    PatternMapping mapping = {{x, t}};
    auto step = apply_identity(id, mapping);

    auto history = Derivation{{step}}.apply(rl, State{t});
    auto* fa = dynamic_cast<FunctionApply*>(history[1].expr());
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->arg(), t);
}

TEST(SubstitutePattern, PowInRhs)
{
    auto rl = make_rl();
    auto* x = make_pattern_var(rl, "x")->constrain_rank(0);
    auto* rhs_pat = make_pow(rl, x, Rational{2});
    Identity id{"sq_id", x, rhs_pat};

    auto* t = make_parameter(rl, "t");
    PatternMapping mapping = {{x, t}};
    auto step = apply_identity(id, mapping);

    auto history = Derivation{{step}}.apply(rl, State{t});
    auto* pw = dynamic_cast<Pow*>(history[1].expr());
    ASSERT_NE(pw, nullptr);
    EXPECT_EQ(pw->base(), t);
}

TEST(SubstitutePattern, ProductInRhs)
{
    auto rl = make_rl();
    auto* x = make_pattern_var(rl, "x")->constrain_rank(0);
    auto* y = make_pattern_var(rl, "y")->constrain_rank(0);
    auto* rhs_pat = make_product(rl, x, y);
    Identity id{"prod_id", x, rhs_pat};

    auto* s = make_symbolic_var(rl, "s");
    auto* t = make_parameter(rl, "t");
    PatternMapping mapping = {{x, s}, {y, t}};
    auto step = apply_identity(id, mapping);

    auto history = Derivation{{step}}.apply(rl, State{s});
    auto* pr = dynamic_cast<Product*>(history[1].expr());
    ASSERT_NE(pr, nullptr);
    EXPECT_EQ(pr->lhs(), s);
    EXPECT_EQ(pr->rhs(), t);
}
