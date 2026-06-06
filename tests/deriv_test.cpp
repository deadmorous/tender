#include <tender/expr.hpp>

#include <gtest/gtest.h>

using namespace tender;

static ResourceList make_rl()
{
    return ResourceList{};
}

// ===========================================================================
// Parameter — construction and rendering
// ===========================================================================

TEST(Parameter, RankIsZero)
{
    auto rl = make_rl();
    auto* p = make_parameter(rl, "t");
    EXPECT_EQ(p->rank(), 0);
}

TEST(Parameter, LatexSingleChar)
{
    auto rl = make_rl();
    EXPECT_EQ(make_parameter(rl, "t")->latex(), "t");
}

TEST(Parameter, LatexMultiChar)
{
    auto rl = make_rl();
    EXPECT_EQ(make_parameter(rl, "lam")->latex(), "\\text{lam}");
}

TEST(Parameter, Python)
{
    auto rl = make_rl();
    EXPECT_EQ(make_parameter(rl, "t")->python(), "parameter('t')");
}

TEST(Parameter, IsSubtypeOfExpr)
{
    auto rl = make_rl();
    Expr* e = make_parameter(rl, "t");
    EXPECT_NE(dynamic_cast<Parameter*>(e), nullptr);
}

// ===========================================================================
// time_parameter
// ===========================================================================

TEST(TimeParameter, SymbolIsT)
{
    EXPECT_EQ(time_parameter()->symbol(), "t");
}

// ===========================================================================
// Product — construction and rendering
// ===========================================================================

TEST(Product, RankIsZero)
{
    auto rl = make_rl();
    auto* x = make_parameter(rl, "x");
    auto* y = make_parameter(rl, "y");
    auto* pr = make_product(rl, x, y);
    EXPECT_EQ(pr->rank(), 0);
}

TEST(Product, NonScalarLhsThrows)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* s = make_parameter(rl, "x");
    EXPECT_THROW(make_product(rl, v, s), std::invalid_argument);
}

TEST(Product, NonScalarRhsThrows)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* s = make_parameter(rl, "x");
    EXPECT_THROW(make_product(rl, s, v), std::invalid_argument);
}

TEST(Product, Latex)
{
    auto rl = make_rl();
    auto* x = make_parameter(rl, "x");
    auto* y = make_parameter(rl, "y");
    auto* pr = make_product(rl, x, y);
    EXPECT_EQ(pr->latex(), "x \\cdot y");
}

TEST(Product, Python)
{
    auto rl = make_rl();
    auto* x = make_parameter(rl, "x");
    auto* y = make_parameter(rl, "y");
    auto* pr = make_product(rl, x, y);
    EXPECT_EQ(pr->python(), "prod(parameter('x'), parameter('y'))");
}

TEST(Product, RationalLhsCollapsesToScale)
{
    auto rl = make_rl();
    auto* two = make_rational(rl, Rational{2});
    auto* x = make_parameter(rl, "x");
    auto* pr = make_product(rl, two, x);
    // make_scale(2, x) = Scale(2, x)
    auto* sc = dynamic_cast<Scale*>(pr);
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->coeff(), Rational{2});
}

TEST(Product, RationalRhsCollapsesToScale)
{
    auto rl = make_rl();
    auto* x = make_parameter(rl, "x");
    auto* three = make_rational(rl, Rational{3});
    auto* pr = make_product(rl, x, three);
    auto* sc = dynamic_cast<Scale*>(pr);
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->coeff(), Rational{3});
}

TEST(Product, ZeroLhsYieldsZero)
{
    auto rl = make_rl();
    auto* zero = make_rational(rl, Rational{0});
    auto* x = make_parameter(rl, "x");
    auto* pr = make_product(rl, zero, x);
    auto* rc = dynamic_cast<RationalConst*>(pr);
    ASSERT_NE(rc, nullptr);
    EXPECT_TRUE(rc->value().is_zero());
}

TEST(Product, OneRhsPassthrough)
{
    auto rl = make_rl();
    auto* x = make_parameter(rl, "x");
    auto* one = make_rational(rl, Rational{1});
    EXPECT_EQ(make_product(rl, x, one), x);
}

// ===========================================================================
// depends_on
// ===========================================================================

TEST(DependsOn, ParameterDependsOnItself)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    EXPECT_TRUE(depends_on(t, t));
}

TEST(DependsOn, ParameterSameNameOtherAlloc)
{
    auto rl = make_rl();
    auto* t1 = make_parameter(rl, "t");
    auto* t2 = make_parameter(rl, "t");
    EXPECT_TRUE(depends_on(t1, t2));
    EXPECT_TRUE(depends_on(t2, t1));
}

TEST(DependsOn, ParameterDifferentName)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* s = make_parameter(rl, "s");
    EXPECT_FALSE(depends_on(t, s));
}

TEST(DependsOn, RationalConstIndependent)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* c = make_rational(rl, Rational{5});
    EXPECT_FALSE(depends_on(t, c));
}

TEST(DependsOn, NamedTensorIndependent)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* A = make_named_tensor(rl, "A", 2, {});
    EXPECT_FALSE(depends_on(t, A));
}

TEST(DependsOn, ScaleInheritsFromInner)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* e = make_scale(rl, Rational{3}, t);
    EXPECT_TRUE(depends_on(t, e));
}

TEST(DependsOn, SumDependsIfAnyTermDoes)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* c = make_rational(rl, Rational{1});
    auto* s = make_sum(rl, {c, t});
    EXPECT_TRUE(depends_on(t, s));
}

TEST(DependsOn, FunctionApplyInheritsFromArg)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* e = make_sin(rl, t);
    EXPECT_TRUE(depends_on(t, e));
}

TEST(DependsOn, PowInheritsFromBase)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* e = make_pow(rl, t, Rational{2});
    EXPECT_TRUE(depends_on(t, e));
}

TEST(DependsOn, ProductInheritsFromOperands)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* s = make_parameter(rl, "s");
    auto* pr = make_product(rl, t, s);
    EXPECT_TRUE(depends_on(t, pr));
    EXPECT_TRUE(depends_on(s, pr));
    auto* r = make_parameter(rl, "r");
    EXPECT_FALSE(depends_on(r, pr));
}

TEST(DependsOn, IdentityTensorIndependent)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* I = make_identity(rl);
    EXPECT_FALSE(depends_on(t, I));
}

// ===========================================================================
// deriv — zero for independent expressions
// ===========================================================================

TEST(Deriv, ZeroForIndependentRationalConst)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* c = make_rational(rl, Rational{7});
    auto* d = deriv(rl, t, c);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_TRUE(rc->value().is_zero());
}

TEST(Deriv, ZeroForIndependentNamedTensor)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* d = deriv(rl, t, A);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_TRUE(rc->value().is_zero());
}

TEST(Deriv, ZeroForDifferentParameter)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* s = make_parameter(rl, "s");
    auto* d = deriv(rl, t, s);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_TRUE(rc->value().is_zero());
}

// ===========================================================================
// deriv — parameter itself
// ===========================================================================

TEST(Deriv, ParameterWrtItself)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* d = deriv(rl, t, t);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{1});
}

TEST(Deriv, ParameterWrtSameNameOtherAlloc)
{
    auto rl = make_rl();
    auto* t1 = make_parameter(rl, "t");
    auto* t2 = make_parameter(rl, "t");
    auto* d = deriv(rl, t1, t2);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{1});
}

// ===========================================================================
// deriv — Scale rule
// ===========================================================================

TEST(Deriv, ScaleRule)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // d/dt (3*t) = 3
    auto* e = make_scale(rl, Rational{3}, t);
    auto* d = deriv(rl, t, e);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{3});
}

// ===========================================================================
// deriv — Sum rule
// ===========================================================================

TEST(Deriv, SumRule)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* s = make_parameter(rl, "s");
    // d/dt (t + s) = 1 + 0 = 1
    auto* e = make_sum(rl, {t, s});
    auto* d = deriv(rl, t, e);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{1});
}

TEST(Deriv, SumOfTwoDependentTerms)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // d/dt (t + t) = d/dt (2t) = 2
    auto* e = make_sum(rl, {t, t}); // make_sum collects: Scale(2, t)
    auto* d = deriv(rl, t, e);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{2});
}

// ===========================================================================
// deriv — chain rule through FunctionApply
// ===========================================================================

TEST(Deriv, ChainRuleSinT)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // d/dt sin(t) = cos(t) * 1 = cos(t)
    auto* e = make_sin(rl, t);
    auto* d = deriv(rl, t, e);
    auto* fa = dynamic_cast<FunctionApply*>(d);
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Cos);
    EXPECT_EQ(fa->arg(), t);
}

TEST(Deriv, ChainRuleExpT)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // d/dt exp(t) = exp(t)
    auto* e = make_exp(rl, t);
    auto* d = deriv(rl, t, e);
    auto* fa = dynamic_cast<FunctionApply*>(d);
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Exp);
}

TEST(Deriv, ChainRuleCosT)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // d/dt cos(t) = -sin(t)
    auto* e = make_cos(rl, t);
    auto* d = deriv(rl, t, e);
    auto* sc = dynamic_cast<Scale*>(d);
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->coeff(), Rational{-1});
    auto* fa = dynamic_cast<FunctionApply*>(sc->expr());
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Sin);
}

TEST(Deriv, ChainRuleSinOf2T)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // d/dt sin(2*t) = cos(2*t) * 2 = Scale(2, cos(2t))
    auto* arg = make_scale(rl, Rational{2}, t);
    auto* e = make_sin(rl, arg);
    auto* d = deriv(rl, t, e);
    // deriv(t, 2t) = 2 (RationalConst)
    // make_product(cos(2t), 2) => make_scale(2, cos(2t))
    auto* sc = dynamic_cast<Scale*>(d);
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->coeff(), Rational{2});
    auto* fa = dynamic_cast<FunctionApply*>(sc->expr());
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Cos);
}

// ===========================================================================
// deriv — chain rule through Pow
// ===========================================================================

TEST(Deriv, ChainRulePowT)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // d/dt t^3 = 3*t^2
    auto* e = make_pow(rl, t, Rational{3});
    auto* d = deriv(rl, t, e);
    auto* sc = dynamic_cast<Scale*>(d);
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->coeff(), Rational{3});
    auto* pw = dynamic_cast<Pow*>(sc->expr());
    ASSERT_NE(pw, nullptr);
    EXPECT_EQ(pw->exponent(), Rational{2});
}

// ===========================================================================
// dt / ddt
// ===========================================================================

TEST(DtDdt, DtOfT)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* d = dt(rl, t);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{1});
}

TEST(DtDdt, DtOfIndependentTensor)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* d = dt(rl, A);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_TRUE(rc->value().is_zero());
}

TEST(DtDdt, DdtOfT)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // d²/dt² t = d/dt 1 = 0
    auto* d = ddt(rl, t);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_TRUE(rc->value().is_zero());
}

TEST(DtDdt, DdtOfT2)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // d²/dt² t^2 = d/dt (2t) = 2
    auto* t2 = make_pow(rl, t, Rational{2});
    auto* d = ddt(rl, t2);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{2});
}

TEST(DtDdt, DtUsesBuiltinT)
{
    // dt(sin(t)) where t is a fresh make_parameter(rl,"t") should still work
    // because dependency tracking is name-based.
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* e = make_sin(rl, t);
    auto* d = dt(rl, e);
    auto* fa = dynamic_cast<FunctionApply*>(d);
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Cos);
}

// ===========================================================================
// MaterialDeriv
// ===========================================================================

TEST(MaterialDeriv, RankMatchesField)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* md = make_material_deriv(rl, v, f);
    EXPECT_EQ(md->rank(), 0);
}

TEST(MaterialDeriv, RankMatchesVectorField)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* md = make_material_deriv(rl, v, u);
    EXPECT_EQ(md->rank(), 1);
}

TEST(MaterialDeriv, NonRank1VelocityThrows)
{
    auto rl = make_rl();
    auto* s = make_named_tensor(rl, "s", 0, {});
    auto* f = make_named_tensor(rl, "f", 0, {});
    EXPECT_THROW(make_material_deriv(rl, s, f), std::invalid_argument);
}

TEST(MaterialDeriv, Latex)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* md = make_material_deriv(rl, v, f);
    EXPECT_EQ(md->latex(), "\\frac{\\mathrm{D}}{\\mathrm{D}t}\\left(f\\right)");
}

TEST(MaterialDeriv, Python)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* md = make_material_deriv(rl, v, f);
    EXPECT_EQ(md->python(), "material_deriv(tensor('v', 1), tensor('f', 0))");
}

TEST(MaterialDeriv, VelocityAndFieldAccessors)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* f = make_named_tensor(rl, "f", 0, {});
    auto* md = make_material_deriv(rl, v, f);
    auto* casted = dynamic_cast<MaterialDeriv*>(md);
    ASSERT_NE(casted, nullptr);
    EXPECT_EQ(casted->velocity(), v);
    EXPECT_EQ(casted->field(), f);
}

// ===========================================================================
// depends_on — coverage for tensor node types
// ===========================================================================

TEST(DependsOn, SumReturnsFalseWhenNoneDepend)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* a = make_symbolic_var(rl, "a");
    auto* b = make_symbolic_var(rl, "b");
    EXPECT_FALSE(depends_on(t, make_sum(rl, {a, b})));
}

TEST(DependsOn, TensorProductContainingParameter)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* tv = make_tensor_product(rl, t, v); // rank-1, contains t
    EXPECT_TRUE(depends_on(t, tv));
}

TEST(DependsOn, TraceContainingParameter)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* tA = make_tensor_product(rl, t, A); // rank-2, contains t
    EXPECT_TRUE(depends_on(t, make_trace(rl, tA)));
}

TEST(DependsOn, ContractContainingParameter)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});
    auto* tv = make_tensor_product(rl, t, v); // rank-1, contains t
    EXPECT_TRUE(depends_on(t, make_contract(rl, tv, w)));
}

TEST(DependsOn, DoubleContractContainingParameter)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* B = make_named_tensor(rl, "B", 2, {});
    auto* tA = make_tensor_product(rl, t, A); // rank-2, contains t
    // make_double_contract(tA, B): tA is not IdentityTensor → DoubleContract
    // node
    EXPECT_TRUE(depends_on(t, make_double_contract(rl, tA, B)));
}

TEST(DependsOn, DoubleContractReversedContainingParameter)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* B = make_named_tensor(rl, "B", 2, {});
    auto* tA = make_tensor_product(rl, t, A);
    EXPECT_TRUE(depends_on(t, make_double_contract_reversed(rl, tA, B)));
}

TEST(DependsOn, CrossProductContainingParameter)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});
    auto* tv = make_tensor_product(rl, t, v); // rank-1, contains t
    EXPECT_TRUE(depends_on(t, make_cross_product(rl, tv, w)));
}

TEST(DependsOn, ATan2ContainingParameter)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* x = make_named_const(rl, "pi");
    EXPECT_TRUE(depends_on(t, make_atan2(rl, t, x)));
}

TEST(DependsOn, MaterialDerivContainingParameter)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* f = make_named_tensor(rl, "f", 0, {});
    // velocity = t⊗v is rank-1 and depends on t
    auto* tv = make_tensor_product(rl, t, v);
    EXPECT_TRUE(depends_on(t, make_material_deriv(rl, tv, f)));
}

// ===========================================================================
// deriv — tensor Leibniz rules
// ===========================================================================

TEST(DerivTensor, TensorProductOneSide)
{
    // d/dt [t ⊗ v] = 1 ⊗ v   (only lhs depends on t)
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* d = deriv(rl, t, make_tensor_product(rl, t, v));
    auto* tp = dynamic_cast<TensorProduct*>(d);
    ASSERT_NE(tp, nullptr);
    EXPECT_EQ(tp->rhs(), v);
    EXPECT_EQ(d->rank(), 1);
}

TEST(DerivTensor, TensorProductBothSides)
{
    // d/dt [t ⊗ t] = 1⊗t + t⊗1  (both sides depend on t)
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* d = deriv(rl, t, make_tensor_product(rl, t, t));
    EXPECT_NE(dynamic_cast<Sum*>(d), nullptr);
    EXPECT_EQ(d->rank(), 0);
}

TEST(DerivTensor, TraceRule)
{
    // d/dt Tr(t ⊗ A) = Tr(1 ⊗ A)
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* tA = make_tensor_product(rl, t, A); // rank-2
    auto* d = deriv(rl, t, make_trace(rl, tA));
    auto* tr = dynamic_cast<Trace*>(d);
    ASSERT_NE(tr, nullptr);
    EXPECT_EQ(d->rank(), 0);
}

TEST(DerivTensor, ContractOneSide)
{
    // d/dt [(t⊗v) · w] = d/dt(t⊗v) · w = (1⊗v) · w
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});
    auto* tv = make_tensor_product(rl, t, v); // rank-1, depends on t
    auto* d = deriv(rl, t, make_contract(rl, tv, w));
    // Result: Contract(deriv(tv), w) = Contract(TensorProduct(1,v), w)
    EXPECT_NE(dynamic_cast<Contract*>(d), nullptr);
    EXPECT_EQ(d->rank(), 0);
}

TEST(DerivTensor, ContractBothSides)
{
    // d/dt [(t⊗v) · (t⊗v)] = both sides depend on t → Sum of two Contracts
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* tv = make_tensor_product(rl, t, v);
    auto* d = deriv(rl, t, make_contract(rl, tv, tv));
    EXPECT_NE(dynamic_cast<Sum*>(d), nullptr);
}

TEST(DerivTensor, DoubleContractRule)
{
    // d/dt [(t⊗A) : B] where only lhs depends on t
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* B = make_named_tensor(rl, "B", 2, {});
    auto* tA = make_tensor_product(rl, t, A); // rank-2, contains t
    auto* dc = make_double_contract(rl, tA, B);
    auto* d = deriv(rl, t, dc);
    // Result: DoubleContract(deriv(tA), B) = DoubleContract(TensorProduct(1,A),
    // B)
    EXPECT_NE(dynamic_cast<DoubleContract*>(d), nullptr);
}

TEST(DerivTensor, DoubleContractReversedRule)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* B = make_named_tensor(rl, "B", 2, {});
    auto* tA = make_tensor_product(rl, t, A);
    auto* dcr = make_double_contract_reversed(rl, tA, B);
    auto* d = deriv(rl, t, dcr);
    EXPECT_NE(dynamic_cast<DoubleContractReversed*>(d), nullptr);
}

TEST(DerivTensor, CrossProductRule)
{
    // d/dt [(t⊗v) × w] where only lhs depends on t
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});
    auto* tv = make_tensor_product(rl, t, v); // rank-1
    auto* d = deriv(rl, t, make_cross_product(rl, tv, w));
    EXPECT_NE(dynamic_cast<CrossProduct*>(d), nullptr);
    EXPECT_EQ(d->rank(), 1);
}
