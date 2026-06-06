#include <tender/derivation.hpp>
#include <tender/expr.hpp>

#include <gtest/gtest.h>

using namespace tender;

static ResourceList make_rl()
{
    return ResourceList{};
}

// ===========================================================================
// simplify_identity_step
// ===========================================================================

// Exit criterion: I · v  →  v
TEST(SimplifyIdentity, ContractIdentityLeft)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* I = make_identity(rl);
    auto* expr = make_contract(rl, I, v);  // I · v

    auto history = Derivation{{simplify_identity_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[1].expr(), v);
    EXPECT_EQ(history[1].label(), "simplify_identity");
}

TEST(SimplifyIdentity, ContractIdentityRight)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* I = make_identity(rl);
    auto* expr = make_contract(rl, v, I);  // v · I

    auto history = Derivation{{simplify_identity_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[1].expr(), v);
}

TEST(SimplifyIdentity, ContractIdentityWithRank2Left)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* I = make_identity(rl);
    auto* expr = make_contract(rl, I, A);  // I · A

    auto history = Derivation{{simplify_identity_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[1].expr(), A);
}

TEST(SimplifyIdentity, ContractIdentityWithRank2Right)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* I = make_identity(rl);
    auto* expr = make_contract(rl, A, I);  // A · I

    auto history = Derivation{{simplify_identity_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[1].expr(), A);
}

TEST(SimplifyIdentity, DoubleContractIdentityLeft)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* I = make_identity(rl);
    auto* expr = make_double_contract(rl, I, A);  // I : A = Tr(A)

    auto history = Derivation{{simplify_identity_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    auto* tr = dynamic_cast<Trace*>(history[1].expr());
    ASSERT_NE(tr, nullptr);
    EXPECT_EQ(tr->arg(), A);
    EXPECT_EQ(history[1].expr()->rank(), 0);
}

TEST(SimplifyIdentity, DoubleContractIdentityRight)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* I = make_identity(rl);
    auto* expr = make_double_contract(rl, A, I);  // A : I = Tr(A)

    auto history = Derivation{{simplify_identity_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    EXPECT_NE(dynamic_cast<Trace*>(history[1].expr()), nullptr);
}

TEST(SimplifyIdentity, DoubleContractReversedIdentityLeft)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* I = make_identity(rl);
    auto* expr = make_double_contract_reversed(rl, I, A);

    auto history = Derivation{{simplify_identity_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    EXPECT_NE(dynamic_cast<Trace*>(history[1].expr()), nullptr);
}

TEST(SimplifyIdentity, RecursiveInsideScale)
{
    // 3 * (I · v)  →  3 * v
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* I = make_identity(rl);
    auto* iv = make_contract(rl, I, v);
    auto* expr = make_scale(rl, Rational{3}, iv);

    auto history = Derivation{{simplify_identity_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    auto* sc = dynamic_cast<Scale*>(history[1].expr());
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->coeff(), Rational{3});
    EXPECT_EQ(sc->expr(), v);
}

TEST(SimplifyIdentity, RecursiveInsideSum)
{
    // (I · u) + (I · v)  →  u + v
    auto rl = make_rl();
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* I = make_identity(rl);
    auto* expr = make_sum(rl, {make_contract(rl, I, u), make_contract(rl, I, v)});

    auto history = Derivation{{simplify_identity_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    auto* s = dynamic_cast<Sum*>(history[1].expr());
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(s->terms().size(), 2u);
    EXPECT_EQ(s->terms()[0], u);
    EXPECT_EQ(s->terms()[1], v);
}

TEST(SimplifyIdentity, LeafPassthrough)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto history = Derivation{{simplify_identity_step()}}.apply(rl, State{v});
    EXPECT_EQ(history[1].expr(), v);
}

TEST(SimplifyIdentity, NonIdentityContractUnchanged)
{
    auto rl = make_rl();
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* expr = make_contract(rl, u, v);

    auto history = Derivation{{simplify_identity_step()}}.apply(rl, State{expr});
    EXPECT_NE(dynamic_cast<Contract*>(history[1].expr()), nullptr);
}

// ===========================================================================
// substitute_step
// ===========================================================================

TEST(Substitute, ReplaceLeafNode)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});
    // Expression: v — replace v with w
    auto history =
        Derivation{{substitute_step(v, w)}}.apply(rl, State{v});
    EXPECT_EQ(history[1].expr(), w);
    EXPECT_EQ(history[1].label(), "substitute");
}

TEST(Substitute, ReplaceInsideScale)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});
    auto* expr = make_scale(rl, Rational{2}, v);  // 2v
    // Replace v → w: should give 2w
    auto history = Derivation{{substitute_step(v, w)}}.apply(rl, State{expr});
    auto* sc = dynamic_cast<Scale*>(history[1].expr());
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->expr(), w);
}

TEST(Substitute, ReplaceInsideSum)
{
    auto rl = make_rl();
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});
    auto* expr = make_sum(rl, {u, v});  // u + v
    // Replace v → w: should give u + w
    auto history = Derivation{{substitute_step(v, w)}}.apply(rl, State{expr});
    auto* s = dynamic_cast<Sum*>(history[1].expr());
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(s->terms().size(), 2u);
    EXPECT_EQ(s->terms()[0], u);
    EXPECT_EQ(s->terms()[1], w);
}

TEST(Substitute, NoMatchReturnsIdentical)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});
    auto* x = make_named_tensor(rl, "x", 1, {});
    // Replace v → w in expression x: no match → x unchanged
    auto history = Derivation{{substitute_step(v, w)}}.apply(rl, State{x});
    EXPECT_EQ(history[1].expr(), x);
}

TEST(Substitute, ReplaceSubtree)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* I = make_identity(rl);
    auto* iv = make_contract(rl, I, v);   // I · v
    auto* w = make_named_tensor(rl, "w", 1, {});
    // Replace (I·v) with w in expression (I·v)
    auto history = Derivation{{substitute_step(iv, w)}}.apply(rl, State{iv});
    EXPECT_EQ(history[1].expr(), w);
}

TEST(Substitute, ReplaceInsideContract)
{
    auto rl = make_rl();
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});
    auto* expr = make_contract(rl, u, v);
    // Replace v → w in u · v → u · w
    auto history = Derivation{{substitute_step(v, w)}}.apply(rl, State{expr});
    auto* co = dynamic_cast<Contract*>(history[1].expr());
    ASSERT_NE(co, nullptr);
    EXPECT_EQ(co->rhs(), w);
}

// ===========================================================================
// expand_step
// ===========================================================================

TEST(Expand, ScaleOverSum)
{
    // 3 * (u + v)  →  3u + 3v
    auto rl = make_rl();
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* expr = make_scale(rl, Rational{3}, make_sum(rl, {u, v}));

    auto history = Derivation{{expand_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[1].label(), "expand");
    auto* s = dynamic_cast<Sum*>(history[1].expr());
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(s->terms().size(), 2u);
    auto* sc0 = dynamic_cast<Scale*>(s->terms()[0]);
    auto* sc1 = dynamic_cast<Scale*>(s->terms()[1]);
    ASSERT_NE(sc0, nullptr);
    ASSERT_NE(sc1, nullptr);
    EXPECT_EQ(sc0->coeff(), Rational{3});
    EXPECT_EQ(sc1->coeff(), Rational{3});
    EXPECT_EQ(sc0->expr(), u);
    EXPECT_EQ(sc1->expr(), v);
}

TEST(Expand, ContractLhsOverSum)
{
    // A · (u + v)  →  A·u + A·v
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* expr = make_contract(rl, A, make_sum(rl, {u, v}));

    auto history = Derivation{{expand_step()}}.apply(rl, State{expr});
    auto* s = dynamic_cast<Sum*>(history[1].expr());
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(s->terms().size(), 2u);
    EXPECT_NE(dynamic_cast<Contract*>(s->terms()[0]), nullptr);
    EXPECT_NE(dynamic_cast<Contract*>(s->terms()[1]), nullptr);
}

TEST(Expand, ContractRhsOverSum)
{
    // (u + v) · w  →  u·w + v·w
    auto rl = make_rl();
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* w = make_named_tensor(rl, "w", 1, {});
    auto* expr = make_contract(rl, make_sum(rl, {u, v}), w);

    auto history = Derivation{{expand_step()}}.apply(rl, State{expr});
    auto* s = dynamic_cast<Sum*>(history[1].expr());
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(s->terms().size(), 2u);
    EXPECT_NE(dynamic_cast<Contract*>(s->terms()[0]), nullptr);
    EXPECT_NE(dynamic_cast<Contract*>(s->terms()[1]), nullptr);
}

TEST(Expand, TensorProductOverSum)
{
    // a ⊗ (u + v)  →  a⊗u + a⊗v
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* expr = make_tensor_product(rl, a, make_sum(rl, {u, v}));

    auto history = Derivation{{expand_step()}}.apply(rl, State{expr});
    auto* s = dynamic_cast<Sum*>(history[1].expr());
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(s->terms().size(), 2u);
}

TEST(Expand, LeafPassthrough)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto history = Derivation{{expand_step()}}.apply(rl, State{v});
    EXPECT_EQ(history[1].expr(), v);
}

TEST(Expand, NestedExpansion)
{
    // 2 * (A · (u + v))  →  2*(A·u) + 2*(A·v)  (two levels)
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* inner = make_contract(rl, A, make_sum(rl, {u, v}));
    auto* expr = make_scale(rl, Rational{2}, inner);

    auto history = Derivation{{expand_step()}}.apply(rl, State{expr});
    // After expand: Sum(Scale(2, A·u), Scale(2, A·v))
    auto* s = dynamic_cast<Sum*>(history[1].expr());
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->terms().size(), 2u);
    EXPECT_NE(dynamic_cast<Scale*>(s->terms()[0]), nullptr);
    EXPECT_NE(dynamic_cast<Scale*>(s->terms()[1]), nullptr);
}

// ===========================================================================
// Combined: expand + simplify_identity
// ===========================================================================

TEST(Combined, ExpandThenSimplifyIdentity)
{
    // I · (u + v)  →  expand → I·u + I·v  →  simplify_id → u + v
    auto rl = make_rl();
    auto* I = make_identity(rl);
    auto* u = make_named_tensor(rl, "u", 1, {});
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* expr = make_contract(rl, I, make_sum(rl, {u, v}));

    Derivation d{{expand_step(), simplify_identity_step()}};
    auto history = d.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 3u);
    auto* s = dynamic_cast<Sum*>(history[2].expr());
    ASSERT_NE(s, nullptr);
    ASSERT_EQ(s->terms().size(), 2u);
    EXPECT_EQ(s->terms()[0], u);
    EXPECT_EQ(s->terms()[1], v);
}
