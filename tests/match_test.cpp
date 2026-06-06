#include <tender/match.hpp>

#include <gtest/gtest.h>

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
