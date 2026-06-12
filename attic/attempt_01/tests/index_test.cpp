#include <tender/expr.hpp>
#include <tender/index.hpp>

#include <gtest/gtest.h>

using namespace tender;

// ===========================================================================
// Helpers
// ===========================================================================

static ResourceList make_rl()
{
    return ResourceList{};
}

// ===========================================================================
// IndexSpace
// ===========================================================================

TEST(IndexSpace, IntegerRange)
{
    IndexSpace s{"spatial", int64_t{3}, true};
    EXPECT_EQ(s.name(), "spatial");
    EXPECT_EQ(std::get<int64_t>(s.range()), 3);
    EXPECT_TRUE(s.auto_sum());
}

TEST(IndexSpace, SymbolicRange)
{
    IndexSpace s{"fem", std::string{"N"}, false};
    EXPECT_EQ(s.name(), "fem");
    EXPECT_EQ(std::get<std::string>(s.range()), "N");
    EXPECT_FALSE(s.auto_sum());
}

TEST(IndexSpace, AutoSumFalse)
{
    IndexSpace s{"layers", int64_t{5}, false};
    EXPECT_FALSE(s.auto_sum());
}

// ===========================================================================
// Index
// ===========================================================================

TEST(Index, ConstructAndAccess)
{
    IndexSpace sp{"s3d", int64_t{3}, true};
    Index idx{"i", &sp};
    EXPECT_EQ(idx.letter(), "i");
    EXPECT_EQ(idx.space(), &sp);
}

TEST(Index, NullSpaceThrows)
{
    EXPECT_THROW(Index("i", nullptr), std::invalid_argument);
}

// ===========================================================================
// Built-in singleton spaces
// ===========================================================================

TEST(BuiltinSpaces, Spatial3dRange)
{
    auto* s = spatial_3d_space();
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(std::get<int64_t>(s->range()), 3);
    EXPECT_TRUE(s->auto_sum());
}

TEST(BuiltinSpaces, Spatial2dRange)
{
    auto* s = spatial_2d_space();
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(std::get<int64_t>(s->range()), 2);
    EXPECT_TRUE(s->auto_sum());
}

TEST(BuiltinSpaces, Spatial3dIsSingleton)
{
    EXPECT_EQ(spatial_3d_space(), spatial_3d_space());
}

TEST(BuiltinSpaces, Spatial2dIsSingleton)
{
    EXPECT_EQ(spatial_2d_space(), spatial_2d_space());
}

// ===========================================================================
// AutoSum shortcut constructors
// ===========================================================================

TEST(AutoSumIndex, AutoSumIndex3d)
{
    auto rl = make_rl();
    auto* idx = auto_sum_index_3d(rl, "i");
    EXPECT_EQ(idx->letter(), "i");
    EXPECT_EQ(idx->space(), spatial_3d_space());
    EXPECT_TRUE(idx->space()->auto_sum());
}

TEST(AutoSumIndex, AutoSumIndex2d)
{
    auto rl = make_rl();
    auto* idx = auto_sum_index_2d(rl, "alpha");
    EXPECT_EQ(idx->letter(), "alpha");
    EXPECT_EQ(idx->space(), spatial_2d_space());
}

TEST(AutoSumIndex, MakeIndexExplicit)
{
    auto rl = make_rl();
    IndexSpace sp{"fem", std::string{"N"}, false};
    auto* idx = make_index(rl, "k", &sp);
    EXPECT_EQ(idx->letter(), "k");
    EXPECT_EQ(idx->space(), &sp);
}

// ===========================================================================
// NamedTensor
// ===========================================================================

TEST(NamedTensor, Rank)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {{SlotLevel::Upper, "i"}});
    EXPECT_EQ(v->rank(), 1);
}

TEST(NamedTensor, Slots)
{
    auto rl = make_rl();
    SlotList sl = {{SlotLevel::Upper, "i"}, {SlotLevel::Lower, "j"}};
    auto* T = make_named_tensor(rl, "T", 2, sl);
    ASSERT_EQ(T->slots().size(), 2u);
    EXPECT_EQ(T->slots()[0].level, SlotLevel::Upper);
    EXPECT_EQ(T->slots()[0].display, "i");
    EXPECT_EQ(T->slots()[1].level, SlotLevel::Lower);
    EXPECT_EQ(T->slots()[1].display, "j");
}

TEST(NamedTensor, LatexNoSlots)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    EXPECT_EQ(A->latex(), "\\mathbf{A}");
}

TEST(NamedTensor, LatexUpperSlot)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {{SlotLevel::Upper, "i"}});
    EXPECT_EQ(v->latex(), "\\mathbf{v}^{i}");
}

TEST(NamedTensor, LatexLowerSlot)
{
    auto rl = make_rl();
    auto* w = make_named_tensor(rl, "w", 1, {{SlotLevel::Lower, "j"}});
    EXPECT_EQ(w->latex(), "\\mathbf{w}_{j}");
}

TEST(NamedTensor, LatexMixedSlots)
{
    auto rl = make_rl();
    SlotList sl = {{SlotLevel::Upper, "i"}, {SlotLevel::Lower, "j"}};
    auto* T = make_named_tensor(rl, "T", 2, sl);
    EXPECT_EQ(T->latex(), "\\mathbf{T}^{i}_{j}");
}

TEST(NamedTensor, Python)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {{SlotLevel::Upper, "i"}});
    EXPECT_EQ(v->python(), "tensor('v', 1)");
}

// ===========================================================================
// ExplicitSum
// ===========================================================================

TEST(ExplicitSum, ReducesRankByTwo)
{
    auto rl = make_rl();
    auto* idx = auto_sum_index_3d(rl, "i");
    SlotList sl = {
        {SlotLevel::Upper, "i", idx},
        {SlotLevel::Lower, "i", idx},
        {SlotLevel::Lower, "j"}};
    auto* T = make_named_tensor(rl, "T", 3, sl);
    auto* e = make_explicit_sum(rl, T, idx);
    EXPECT_EQ(e->rank(), 1);
}

TEST(ExplicitSum, ConsumesPairedSlots)
{
    auto rl = make_rl();
    auto* idx = auto_sum_index_3d(rl, "i");
    SlotList sl = {
        {SlotLevel::Upper, "i", idx},
        {SlotLevel::Lower, "i", idx},
        {SlotLevel::Lower, "j"}};
    auto* T = make_named_tensor(rl, "T", 3, sl);
    auto* e = make_explicit_sum(rl, T, idx);
    // Only the "j" lower slot should remain.
    ASSERT_EQ(e->slots().size(), 1u);
    EXPECT_EQ(e->slots()[0].display, "j");
}

TEST(ExplicitSum, NoPairThrows)
{
    auto rl = make_rl();
    auto* idx = auto_sum_index_3d(rl, "i");
    // Body has no slot bound to idx.
    auto* v = make_named_tensor(rl, "v", 1, {{SlotLevel::Upper, "i"}});
    EXPECT_THROW(make_explicit_sum(rl, v, idx), std::invalid_argument);
}

TEST(ExplicitSum, Latex)
{
    auto rl = make_rl();
    auto* idx = auto_sum_index_3d(rl, "i");
    SlotList sl = {{SlotLevel::Upper, "i", idx}, {SlotLevel::Lower, "i", idx}};
    auto* T = make_named_tensor(rl, "T", 2, sl);
    auto* e = make_explicit_sum(rl, T, idx);
    EXPECT_EQ(e->latex(), "\\sum_{i} \\mathbf{T}^{i}_{i}");
}

// ===========================================================================
// NoSum
// ===========================================================================

TEST(NoSum, PreservesRank)
{
    auto rl = make_rl();
    auto* idx = auto_sum_index_3d(rl, "i");
    SlotList sl = {{SlotLevel::Upper, "i", idx}, {SlotLevel::Lower, "i", idx}};
    auto* T = make_named_tensor(rl, "T", 2, sl);
    auto* e = make_no_sum(rl, T, idx);
    EXPECT_EQ(e->rank(), 2);
}

TEST(NoSum, PreservesSlots)
{
    auto rl = make_rl();
    auto* idx = auto_sum_index_3d(rl, "i");
    SlotList sl = {{SlotLevel::Upper, "i", idx}, {SlotLevel::Lower, "i", idx}};
    auto* T = make_named_tensor(rl, "T", 2, sl);
    auto* e = make_no_sum(rl, T, idx);
    EXPECT_EQ(e->slots().size(), 2u);
}

TEST(NoSum, Latex)
{
    auto rl = make_rl();
    auto* idx = auto_sum_index_3d(rl, "i");
    SlotList sl = {{SlotLevel::Upper, "i", idx}, {SlotLevel::Lower, "i", idx}};
    auto* T = make_named_tensor(rl, "T", 2, sl);
    auto* e = make_no_sum(rl, T, idx);
    // NoSum is transparent in LaTeX (just a semantic marker).
    EXPECT_EQ(e->latex(), T->latex());
}

// ===========================================================================
// convolve()
// ===========================================================================

TEST(Convolve, UpperWithLower)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {{SlotLevel::Upper, "i"}});
    auto* w = make_named_tensor(rl, "w", 1, {{SlotLevel::Lower, "i"}});
    auto* e = convolve(rl, v, 0, w, 0);
    EXPECT_EQ(e->rank(), 0);
    EXPECT_EQ(e->slots().size(), 0u);
}

TEST(Convolve, RemainingSlots)
{
    auto rl = make_rl();
    // T: rank-2 with upper i, lower j
    SlotList sl_t = {{SlotLevel::Upper, "i"}, {SlotLevel::Lower, "j"}};
    auto* T = make_named_tensor(rl, "T", 2, sl_t);
    // v: rank-1 with lower i
    auto* v = make_named_tensor(rl, "v", 1, {{SlotLevel::Lower, "i"}});
    // convolve T's upper slot (0) with v's lower slot (0): T^i v_i → rank 1
    auto* e = convolve(rl, T, 0, v, 0);
    EXPECT_EQ(e->rank(), 1);
    // One slot left: T's lower j.
    ASSERT_EQ(e->slots().size(), 1u);
    EXPECT_EQ(e->slots()[0].display, "j");
}

TEST(Convolve, SameLevelThrows)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {{SlotLevel::Upper, "i"}});
    auto* w = make_named_tensor(rl, "w", 1, {{SlotLevel::Upper, "i"}});
    EXPECT_THROW(convolve(rl, v, 0, w, 0), std::invalid_argument);
}

TEST(Convolve, OutOfBoundsSlotThrows)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {{SlotLevel::Upper, "i"}});
    auto* w = make_named_tensor(rl, "w", 1, {{SlotLevel::Lower, "i"}});
    EXPECT_THROW(convolve(rl, v, 5, w, 0), std::invalid_argument);
    EXPECT_THROW(convolve(rl, v, 0, w, 5), std::invalid_argument);
}

TEST(Convolve, Latex)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {{SlotLevel::Upper, "i"}});
    auto* w = make_named_tensor(rl, "w", 1, {{SlotLevel::Lower, "i"}});
    auto* e = convolve(rl, v, 0, w, 0);
    EXPECT_EQ(e->latex(), "\\mathbf{v}^{i} \\mathbf{w}_{i}");
}

TEST(Convolve, Python)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {{SlotLevel::Upper, "i"}});
    auto* w = make_named_tensor(rl, "w", 1, {{SlotLevel::Lower, "i"}});
    auto* e = convolve(rl, v, 0, w, 0);
    EXPECT_EQ(e->python(), "convolve(tensor('v', 1), 0, tensor('w', 1), 0)");
}

TEST(Convolve, RhsRemainingSlots)
{
    auto rl = make_rl();
    // v: rank-1 upper "i"
    auto* v = make_named_tensor(rl, "v", 1, {{SlotLevel::Upper, "i"}});
    // T: rank-2 lower "i", lower "j"
    SlotList sl = {{SlotLevel::Lower, "i"}, {SlotLevel::Lower, "j"}};
    auto* T = make_named_tensor(rl, "T", 2, sl);
    // convolve v's upper (0) with T's lower i (0): rank = 1+2-2 = 1
    auto* e = convolve(rl, v, 0, T, 0);
    EXPECT_EQ(e->rank(), 1);
    // Remaining: T's lower j (from rhs)
    ASSERT_EQ(e->slots().size(), 1u);
    EXPECT_EQ(e->slots()[0].display, "j");
    EXPECT_EQ(e->slots()[0].level, SlotLevel::Lower);
}

TEST(ExplicitSum, Python)
{
    auto rl = make_rl();
    auto* idx = auto_sum_index_3d(rl, "i");
    SlotList sl = {{SlotLevel::Upper, "i", idx}, {SlotLevel::Lower, "i", idx}};
    auto* T = make_named_tensor(rl, "T", 2, sl);
    auto* e = make_explicit_sum(rl, T, idx);
    EXPECT_EQ(e->python(), "explicit_sum(index=i, tensor('T', 2))");
}

TEST(NoSum, Python)
{
    auto rl = make_rl();
    auto* idx = auto_sum_index_3d(rl, "i");
    SlotList sl = {{SlotLevel::Upper, "i", idx}, {SlotLevel::Lower, "i", idx}};
    auto* T = make_named_tensor(rl, "T", 2, sl);
    auto* e = make_no_sum(rl, T, idx);
    EXPECT_EQ(e->python(), "no_sum(index=i, tensor('T', 2))");
}
