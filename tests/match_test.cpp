#include <tender/match.hpp>

#include <gtest/gtest.h>

#include <tender/expr.hpp>
#include <tender/integral.hpp>

using namespace tender;

static ResourceList make_rl()
{
    return ResourceList{};
}

// ===========================================================================
// match_pattern — basic cases
// ===========================================================================

TEST(MatchPattern, PatternVarBindsToAnyExpr)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* v = make_named_tensor(rl, "v", 1, {});

    auto bindings = match_pattern(a, v, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
    EXPECT_EQ(bindings[0].at(a), v);
}

TEST(MatchPattern, PatternVarRankMismatchFails)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* s = make_named_tensor(rl, "s", 0, {}); // scalar

    EXPECT_TRUE(match_pattern(a, s, {}).empty());
}

TEST(MatchPattern, SamePatternVarMustBindConsistently)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});

    // Pattern: a·a  against u·v — should fail (a can't be both u and v)
    auto* pattern = rl.make<Contract>(a, a);
    auto* expr = rl.make<Contract>(u, v);
    EXPECT_TRUE(match_pattern(pattern, expr, {}).empty());
}

TEST(MatchPattern, SamePatternVarBindsConsistentlyOk)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* v = make_named_tensor(rl, "v", 1, {});

    // Pattern: a·a  against v·v — should succeed
    auto* pattern = rl.make<Contract>(a, a);
    auto* expr = rl.make<Contract>(v, v);
    auto bindings = match_pattern(pattern, expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
    EXPECT_EQ(bindings[0].at(a), v);
}

TEST(MatchPattern, ConcreteLeafMismatchFails)
{
    auto rl = make_rl();
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    EXPECT_TRUE(match_pattern(u, v, {}).empty());
}

TEST(MatchPattern, CrossProductStructure)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    auto* b = make_pattern_var(rl, "b");
    a->constrain_rank(1);
    b->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});

    auto* pattern = rl.make<CrossProduct>(a, b);
    auto* expr = rl.make<CrossProduct>(u, v);
    auto bindings = match_pattern(pattern, expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
    EXPECT_EQ(bindings[0].at(a), u);
    EXPECT_EQ(bindings[0].at(b), v);
}

// BAC-CAB pattern:  a×(b×c)
TEST(MatchPattern, BacCabLhsPattern)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(1);
    auto* c = make_pattern_var(rl, "c");
    c->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});

    auto* pattern = rl.make<CrossProduct>(a, rl.make<CrossProduct>(b, c));
    auto* expr = rl.make<CrossProduct>(u, rl.make<CrossProduct>(v, w));
    auto bindings = match_pattern(pattern, expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
    EXPECT_EQ(bindings[0].at(a), u);
    EXPECT_EQ(bindings[0].at(b), v);
    EXPECT_EQ(bindings[0].at(c), w);
}

// ===========================================================================
// find_matches — tree search
// ===========================================================================

TEST(FindMatches, MatchAtRoot)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* v = make_named_tensor(rl, "v", 1, {});

    Identity id{"test", a, a};
    auto matches = find_matches(id, v);
    ASSERT_EQ(static_cast<int>(matches.size()), 1);
    EXPECT_EQ(matches[0].at(a), v);
}

TEST(FindMatches, MatchDeepInTree)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});

    // Pattern: a×(b×a)  — requires a==u in both positions
    auto* pattern_lhs = rl.make<CrossProduct>(a, rl.make<CrossProduct>(b, a));
    Identity id{"test", pattern_lhs, a};

    // Expr: u×(v×u)
    auto* expr = rl.make<CrossProduct>(u, rl.make<CrossProduct>(v, u));

    auto matches = find_matches(id, expr);
    // Should find a match at the root with a→u, b→v
    ASSERT_FALSE(matches.empty());
    bool found = false;
    for (auto const& m: matches)
        if (m.count(a) && m.at(a) == u && m.count(b) && m.at(b) == v)
            found = true;
    EXPECT_TRUE(found);
}

TEST(FindMatches, MatchInsideSum)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {}); // rank-1 placeholder

    // Pattern: a×(b×a)
    auto* pattern_lhs = rl.make<CrossProduct>(a, rl.make<CrossProduct>(b, a));
    // Make a rank-1 identity RHS (a itself)
    Identity id{"test", pattern_lhs, a};

    // u×(v×u) is buried inside a sum with an unrelated term
    auto* uvu = rl.make<CrossProduct>(u, rl.make<CrossProduct>(v, u));
    // Sum needs same rank — CrossProduct rank 1, so use another rank-1 term
    auto* expr = make_sum(rl, {uvu, w});

    auto matches = find_matches(id, expr);
    bool found = false;
    for (auto const& m: matches)
        if (m.count(a) && m.at(a) == u)
            found = true;
    EXPECT_TRUE(found);
}

TEST(FindMatches, BudgetExceeded)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* v = make_named_tensor(rl, "v", 1, {});

    Identity id{"test", a, a};
    EXPECT_THROW(find_matches(id, v, 0), std::runtime_error);
}

// ===========================================================================
// apply_identity_auto
// ===========================================================================

TEST(ApplyIdentityAuto, FindsAndApplies)
{
    auto rl = make_rl();
    // Pattern: a×(b×c) = b(a·c) − c(a·b)  — but just test that it finds
    // something
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(1);
    auto* c = make_pattern_var(rl, "c");
    c->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});

    // lhs = a×(b×c), rhs = b (just use b as the result for simplicity)
    auto* lhs = rl.make<CrossProduct>(a, rl.make<CrossProduct>(b, c));
    Identity id{"bac_cab_simplified", lhs, b};

    auto* expr = rl.make<CrossProduct>(u, rl.make<CrossProduct>(v, w));
    auto step = apply_identity_auto(id, expr);
    // Step applies: result should be v (the b slot)
    auto* result = step.apply(rl, State{expr}).expr();
    EXPECT_EQ(result, v);
}

TEST(ApplyIdentityAuto, NoMatchThrows)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* s = make_named_tensor(rl, "s", 0, {}); // scalar — won't match rank-1

    Identity id{"test", a, a};
    // s is rank 0, a requires rank 1 — no match
    EXPECT_THROW(apply_identity_auto(id, s), std::invalid_argument);
}

// ===========================================================================
// match_pattern — structural branches not yet covered
// ===========================================================================

TEST(MatchPattern, SumPatternMatchesSum)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});

    auto* pattern = make_sum(rl, {a, b});
    auto* expr = make_sum(rl, {u, v});
    auto bindings = match_pattern(pattern, expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
    EXPECT_EQ(bindings[0].at(a), u);
    EXPECT_EQ(bindings[0].at(b), v);
}

TEST(MatchPattern, SumPatternSizeMismatch)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(1);
    auto* c = make_pattern_var(rl, "c");
    c->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});

    auto* pattern = make_sum(rl, {a, b, c}); // 3 terms
    auto* expr = make_sum(rl, {u, v});       // 2 terms
    EXPECT_TRUE(match_pattern(pattern, expr, {}).empty());
}

TEST(MatchPattern, SumNonSumExprFails)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});

    auto* pattern = make_sum(rl, {a, b});
    EXPECT_TRUE(match_pattern(pattern, u, {}).empty());
}

TEST(MatchPattern, ScalePatternMatches)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});

    auto* pattern = make_scale(rl, Rational{2}, a);
    auto* expr = make_scale(rl, Rational{2}, u);
    auto bindings = match_pattern(pattern, expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
    EXPECT_EQ(bindings[0].at(a), u);
}

TEST(MatchPattern, ScaleCoeffMismatch)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});

    auto* pattern = make_scale(rl, Rational{2}, a);
    auto* expr = make_scale(rl, Rational{3}, u);
    EXPECT_TRUE(match_pattern(pattern, expr, {}).empty());
}

TEST(MatchPattern, ScaleNonScaleFails)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});

    auto* pattern = make_scale(rl, Rational{2}, a);
    EXPECT_TRUE(match_pattern(pattern, u, {}).empty());
}

TEST(MatchPattern, TensorProductPattern)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(0);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* s = make_named_tensor(rl, "s", 0, {});

    auto* pattern = make_tensor_product(rl, a, b);
    auto* expr = make_tensor_product(rl, u, s);
    auto bindings = match_pattern(pattern, expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
    EXPECT_EQ(bindings[0].at(a), u);
    EXPECT_EQ(bindings[0].at(b), s);
}

TEST(MatchPattern, DoubleContractPattern)
{
    auto rl = make_rl();
    auto* A = make_pattern_var(rl, "A");
    A->constrain_rank(2);
    auto* B = make_pattern_var(rl, "B");
    B->constrain_rank(2);
    auto* M = make_named_tensor(rl, "M", 2, {});
    auto* N = make_named_tensor(rl, "N", 2, {});

    auto* pattern = make_double_contract(rl, A, B);
    auto* expr = make_double_contract(rl, M, N);
    auto bindings = match_pattern(pattern, expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
    EXPECT_EQ(bindings[0].at(A), M);
    EXPECT_EQ(bindings[0].at(B), N);
}

TEST(MatchPattern, DoubleContractReversedPattern)
{
    auto rl = make_rl();
    auto* A = make_pattern_var(rl, "A");
    A->constrain_rank(2);
    auto* B = make_pattern_var(rl, "B");
    B->constrain_rank(2);
    auto* M = make_named_tensor(rl, "M", 2, {});
    auto* N = make_named_tensor(rl, "N", 2, {});

    auto* pattern = make_double_contract_reversed(rl, A, B);
    auto* expr = make_double_contract_reversed(rl, M, N);
    auto bindings = match_pattern(pattern, expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
    EXPECT_EQ(bindings[0].at(A), M);
    EXPECT_EQ(bindings[0].at(B), N);
}

TEST(MatchPattern, TracePattern)
{
    auto rl = make_rl();
    auto* A = make_pattern_var(rl, "A");
    A->constrain_rank(2);
    auto* M = make_named_tensor(rl, "M", 2, {});

    auto* pattern = make_trace(rl, A);
    auto* expr = make_trace(rl, M);
    auto bindings = match_pattern(pattern, expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
    EXPECT_EQ(bindings[0].at(A), M);
}

TEST(MatchPattern, TraceTypeMismatch)
{
    auto rl = make_rl();
    auto* A = make_pattern_var(rl, "A");
    A->constrain_rank(2);
    auto* M = make_named_tensor(rl, "M", 2, {});

    auto* pattern = make_trace(rl, A);
    // M itself is not a Trace node
    EXPECT_TRUE(match_pattern(pattern, M, {}).empty());
}

TEST(MatchPattern, FunctionApplySameKind)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(0);
    auto* s = make_named_tensor(rl, "s", 0, {});

    auto* pattern = make_function(rl, FunctionKind::Sin, a);
    auto* expr = make_function(rl, FunctionKind::Sin, s);
    auto bindings = match_pattern(pattern, expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
    EXPECT_EQ(bindings[0].at(a), s);
}

TEST(MatchPattern, FunctionApplyDifferentKind)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(0);
    auto* s = make_named_tensor(rl, "s", 0, {});

    auto* pattern = make_function(rl, FunctionKind::Sin, a);
    auto* expr = make_function(rl, FunctionKind::Cos, s);
    EXPECT_TRUE(match_pattern(pattern, expr, {}).empty());
}

TEST(MatchPattern, PowPatternMatches)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(0);
    auto* s = make_named_tensor(rl, "s", 0, {});

    auto* pattern = make_pow(rl, a, Rational{2});
    auto* expr = make_pow(rl, s, Rational{2});
    auto bindings = match_pattern(pattern, expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
    EXPECT_EQ(bindings[0].at(a), s);
}

TEST(MatchPattern, PowExponentMismatch)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(0);
    auto* s = make_named_tensor(rl, "s", 0, {});

    auto* pattern = make_pow(rl, a, Rational{2});
    auto* expr = make_pow(rl, s, Rational{3});
    EXPECT_TRUE(match_pattern(pattern, expr, {}).empty());
}

TEST(MatchPattern, IdentityTensorMatchesSelf)
{
    auto rl = make_rl();
    auto* I = make_identity(rl);
    ASSERT_EQ(static_cast<int>(match_pattern(I, I, {}).size()), 1);
}

TEST(MatchPattern, IdentityTensorMismatch)
{
    auto rl = make_rl();
    auto* I = make_identity(rl);
    auto* M = make_named_tensor(rl, "M", 2, {});
    EXPECT_TRUE(match_pattern(I, M, {}).empty());
}

TEST(MatchPattern, LeviCivitaMatchesSelf)
{
    auto rl = make_rl();
    auto* eps = make_levi_civita(rl);
    ASSERT_EQ(static_cast<int>(match_pattern(eps, eps, {}).size()), 1);
}

TEST(MatchPattern, RationalConstMatches)
{
    auto rl = make_rl();
    auto* two = rl.make<RationalConst>(Rational{2});
    auto* two2 = rl.make<RationalConst>(Rational{2});
    ASSERT_EQ(static_cast<int>(match_pattern(two, two2, {}).size()), 1);
}

TEST(MatchPattern, RationalConstMismatch)
{
    auto rl = make_rl();
    auto* two = rl.make<RationalConst>(Rational{2});
    auto* three = rl.make<RationalConst>(Rational{3});
    EXPECT_TRUE(match_pattern(two, three, {}).empty());
}

TEST(MatchPattern, SymbolicVarMatchesSelf)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    ASSERT_EQ(static_cast<int>(match_pattern(x, x, {}).size()), 1);
}

TEST(MatchPattern, SymbolicVarMismatch)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* y = make_symbolic_var(rl, "y");
    EXPECT_TRUE(match_pattern(x, y, {}).empty());
}

TEST(MatchPattern, SymmetricConstraintMatches)
{
    auto rl = make_rl();
    auto* A = make_pattern_var(rl, "A");
    A->constrain_rank(2);
    A->constrain_symmetric();
    auto* M_expr = make_named_tensor(rl, "M", 2, {});
    auto* M = dynamic_cast<NamedTensor*>(M_expr);
    ASSERT_NE(M, nullptr);
    M->declare_symmetric();

    auto bindings = match_pattern(A, M_expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
}

TEST(MatchPattern, SymmetricConstraintFailsOnNonSymmetric)
{
    auto rl = make_rl();
    auto* A = make_pattern_var(rl, "A");
    A->constrain_rank(2);
    A->constrain_symmetric();
    auto* M = make_named_tensor(rl, "M", 2, {}); // not declared symmetric

    EXPECT_TRUE(match_pattern(A, M, {}).empty());
}

TEST(MatchPattern, SkewConstraintMatches)
{
    auto rl = make_rl();
    auto* A = make_pattern_var(rl, "A");
    A->constrain_rank(2);
    A->constrain_skew_symmetric();
    auto* M_expr = make_named_tensor(rl, "M", 2, {});
    auto* M = dynamic_cast<NamedTensor*>(M_expr);
    ASSERT_NE(M, nullptr);
    M->declare_skew_symmetric();

    auto bindings = match_pattern(A, M_expr, {});
    ASSERT_EQ(static_cast<int>(bindings.size()), 1);
}

TEST(MatchPattern, SkewConstraintFailsOnNonSkew)
{
    auto rl = make_rl();
    auto* A = make_pattern_var(rl, "A");
    A->constrain_rank(2);
    A->constrain_skew_symmetric();
    auto* M = make_named_tensor(rl, "M", 2, {}); // not declared skew

    EXPECT_TRUE(match_pattern(A, M, {}).empty());
}

// ===========================================================================
// find_and_rewrite_all
// ===========================================================================

TEST(FindAndRewriteAll, RootMatch)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});

    // identity: a×b → b (trivial RHS for testing)
    auto* lhs = rl.make<CrossProduct>(a, b);
    Identity id{"test", lhs, b};

    auto* expr = rl.make<CrossProduct>(u, v);
    auto results = find_and_rewrite_all(id, rl, expr);

    ASSERT_EQ(static_cast<int>(results.size()), 1);
    EXPECT_EQ(results[0].first, v);
    EXPECT_EQ(results[0].second, "apply(test)");
}

TEST(FindAndRewriteAll, SubExprInsideScale)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});

    // identity: a×b → b
    auto* lhs = rl.make<CrossProduct>(a, b);
    Identity id{"test", lhs, b};

    // Scale(-1, u×v): the inner cross matches; new root is Scale(-1, v)
    auto* inner = rl.make<CrossProduct>(u, v);
    auto* expr = make_scale(rl, Rational{-1}, inner);

    auto results = find_and_rewrite_all(id, rl, expr);

    // One match: inside the Scale
    bool found = false;
    for (auto const& [new_root, name]: results)
    {
        auto* sc = dynamic_cast<Scale*>(new_root);
        if (sc && sc->coeff() == Rational{-1} && sc->expr() == v)
            found = true;
    }
    EXPECT_TRUE(found);
}

TEST(FindAndRewriteAll, SubExprInsideSum)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});

    // identity: a×b → b
    auto* lhs = rl.make<CrossProduct>(a, b);
    Identity id{"test", lhs, b};

    // Sum(u×v, w): one term matches, one doesn't
    auto* cross_uv = rl.make<CrossProduct>(u, v);
    auto* expr = make_sum(rl, {cross_uv, w});

    auto results = find_and_rewrite_all(id, rl, expr);

    // Should find match at the cross_uv term, producing Sum(v, w)
    bool found = false;
    for (auto const& [new_root, name]: results)
    {
        auto* s = dynamic_cast<Sum*>(new_root);
        if (!s || s->terms().size() != 2)
            continue;
        if (s->terms()[0] == v && s->terms()[1] == w)
            found = true;
    }
    EXPECT_TRUE(found);
}

TEST(FindAndRewriteAll, SubExprInsideTensorProduct)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* s = make_named_tensor(rl, "s", 0, {});

    // identity: a → u (trivial: replace any rank-1 with u itself)
    Identity id{"test", a, u};

    // TensorProduct(u, s): lhs matches a → u; new root is TensorProduct(u, s)
    auto* expr = make_tensor_product(rl, u, s);
    auto results = find_and_rewrite_all(id, rl, expr);

    EXPECT_FALSE(results.empty());
}

TEST(FindAndRewriteAll, SubExprInsideContract)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});

    // identity: a → u
    Identity id{"test", a, u};

    // Contract(u, v): both sides match a
    auto* expr = rl.make<Contract>(u, v);
    auto results = find_and_rewrite_all(id, rl, expr);
    EXPECT_FALSE(results.empty());
}

TEST(FindAndRewriteAll, SubExprInsideDoubleContract)
{
    auto rl = make_rl();
    auto* A = make_pattern_var(rl, "A");
    A->constrain_rank(2);
    auto* M = make_named_tensor(rl, "M", 2, {});
    auto* N = make_named_tensor(rl, "N", 2, {});

    Identity id{"test", A, M}; // replace any rank-2 with M

    auto* expr = make_double_contract(rl, M, N);
    auto results = find_and_rewrite_all(id, rl, expr);
    EXPECT_FALSE(results.empty());
}

TEST(FindAndRewriteAll, SubExprInsideDoubleContractReversed)
{
    auto rl = make_rl();
    auto* A = make_pattern_var(rl, "A");
    A->constrain_rank(2);
    auto* M = make_named_tensor(rl, "M", 2, {});
    auto* N = make_named_tensor(rl, "N", 2, {});

    Identity id{"test", A, M};

    auto* expr = make_double_contract_reversed(rl, M, N);
    auto results = find_and_rewrite_all(id, rl, expr);
    EXPECT_FALSE(results.empty());
}

TEST(FindAndRewriteAll, SubExprInsideTrace)
{
    auto rl = make_rl();
    auto* A = make_pattern_var(rl, "A");
    A->constrain_rank(2);
    auto* M = make_named_tensor(rl, "M", 2, {});
    auto* N = make_named_tensor(rl, "N", 2, {});

    Identity id{"test", A, N}; // replace any rank-2 with N

    auto* expr = make_trace(rl, M);
    auto results = find_and_rewrite_all(id, rl, expr);
    ASSERT_EQ(static_cast<int>(results.size()), 1);
    auto* tr = dynamic_cast<Trace*>(results[0].first);
    ASSERT_NE(tr, nullptr);
    EXPECT_EQ(tr->arg(), N);
}

TEST(FindAndRewriteAll, SubExprInsideFunctionApply)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(0);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(0);
    auto* s = make_named_tensor(rl, "s", 0, {});
    auto* t = make_named_tensor(rl, "t", 0, {});

    // lhs = Sin(a): only matches FunctionApply(Sin, _), not bare scalars
    auto* lhs = make_function(rl, FunctionKind::Sin, a);
    // rhs = b: but we need it to map a→s. Use a simpler identity: Sin(a) → t
    auto* rhs = t;
    Identity id{"test", lhs, rhs};

    // expr = Sin(s): should match at root, giving t
    auto* expr = make_function(rl, FunctionKind::Sin, s);
    auto results = find_and_rewrite_all(id, rl, expr);
    ASSERT_EQ(static_cast<int>(results.size()), 1);
    EXPECT_EQ(results[0].first, t);
}

TEST(FindAndRewriteAll, SubExprInsidePow)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(0);
    auto* s = make_named_tensor(rl, "s", 0, {});
    auto* t = make_named_tensor(rl, "t", 0, {});

    // lhs = a^2: only matches Pow(_, 2)
    auto* lhs = make_pow(rl, a, Rational{2});
    Identity id{"test", lhs, t};

    auto* expr = make_pow(rl, s, Rational{2});
    auto results = find_and_rewrite_all(id, rl, expr);
    ASSERT_EQ(static_cast<int>(results.size()), 1);
    EXPECT_EQ(results[0].first, t);
}

TEST(FindAndRewriteAll, NoMatch)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* s = make_named_tensor(rl, "s", 0, {});

    // identity: a×b (vector cross) — no cross in expr
    auto* lhs = rl.make<CrossProduct>(a, b);
    Identity id{"test", lhs, a};

    auto results = find_and_rewrite_all(id, rl, u);
    EXPECT_TRUE(results.empty());
}

TEST(FindAndRewriteAll, InsideIntegrand)
{
    auto rl = make_rl();
    auto* a = make_pattern_var(rl, "a");
    a->constrain_rank(1);
    auto* b = make_pattern_var(rl, "b");
    b->constrain_rank(1);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* n = make_named_tensor(rl, "n", 1, {});

    // lhs = a×b: only matches CrossProduct nodes, not bare rank-1 tensors
    auto* lhs = rl.make<CrossProduct>(a, b);
    Identity id{"test", lhs, v};

    // integrand = u×n: match inside the integral
    auto* cross_un = rl.make<CrossProduct>(u, n);
    auto* domain = make_volume_domain(rl, "V", n);
    auto* expr = make_integral(rl, domain, cross_un);
    auto results = find_and_rewrite_all(id, rl, expr);

    bool found = false;
    for (auto const& [new_root, name]: results)
    {
        auto* integ = dynamic_cast<Integral*>(new_root);
        if (integ && integ->integrand() == v)
            found = true;
    }
    EXPECT_TRUE(found);
}
