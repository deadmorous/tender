#include <tender/index.hpp>

#include <gtest/gtest.h>

using namespace tender;

// ---- IndexSpace construction -------------------------------------------

TEST(IndexSpace, ValidConstruction)
{
    Context ctx;
    auto* sp = make_index_space(
        ctx,
        {1, 2, 3},
        {make_index_name("i"), make_index_name("j"), make_index_name("k")});
    ASSERT_NE(sp, nullptr);
}

TEST(IndexSpace, EmptyValuesThrows)
{
    Context ctx;
    EXPECT_THROW(
        { (void)make_index_space(ctx, {}, {make_index_name("i")}); },
        std::invalid_argument);
}

TEST(IndexSpace, EmptySchemaThrows)
{
    Context ctx;
    EXPECT_THROW(
        { (void)make_index_space(ctx, {1, 2, 3}, {}); }, std::invalid_argument);
}

TEST(IndexSpace, NonContiguousValuesAccepted)
{
    Context ctx;
    auto* sp = make_index_space(
        ctx, {1, 3}, {make_index_name("i"), make_index_name("j")});
    ASSERT_NE(sp, nullptr);
    auto vals = sp->values();
    ASSERT_EQ(vals.size(), 2u);
    EXPECT_EQ(vals[0], 1);
    EXPECT_EQ(vals[1], 3);
}

// ---- IndexSpace::values() ----------------------------------------------

TEST(IndexSpace, ValuesSpanMatchesInput)
{
    Context ctx;
    auto* sp = make_index_space(
        ctx,
        {0, 1, 2, 3},
        {make_index_name("\\mu"),
         make_index_name("\\nu"),
         make_index_name("\\rho"),
         make_index_name("\\sigma")});
    auto vals = sp->values();
    ASSERT_EQ(vals.size(), 4u);
    EXPECT_EQ(vals[0], 0);
    EXPECT_EQ(vals[1], 1);
    EXPECT_EQ(vals[2], 2);
    EXPECT_EQ(vals[3], 3);
}

// ---- IndexSpace::dummy_name() ------------------------------------------

TEST(IndexSpace, DummyNameReturnsCorrectEntry)
{
    Context ctx;
    auto* sp = make_index_space(
        ctx,
        {1, 2},
        {make_index_name("i"), make_index_name("j"), make_index_name("k")});
    EXPECT_EQ(sp->dummy_name(0).v.view(), "i");
    EXPECT_EQ(sp->dummy_name(1).v.view(), "j");
    EXPECT_EQ(sp->dummy_name(2).v.view(), "k");
}

TEST(IndexSpace, DummyNameOutOfRangeThrows)
{
    Context ctx;
    auto* sp = make_index_space(
        ctx, {1, 2}, {make_index_name("i"), make_index_name("j")});
    EXPECT_THROW(sp->dummy_name(-1), std::out_of_range);
    EXPECT_THROW(sp->dummy_name(2), std::out_of_range);
}

// ---- Well-known singletons ---------------------------------------------

TEST(WellKnownSpaces, Space3dIsNonNull)
{
    EXPECT_NE(space_3d(), nullptr);
}

TEST(WellKnownSpaces, Space3dIsSingleton)
{
    EXPECT_EQ(space_3d(), space_3d());
}

TEST(WellKnownSpaces, Space3dValues)
{
    auto vals = space_3d()->values();
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_EQ(vals[0], 1);
    EXPECT_EQ(vals[1], 2);
    EXPECT_EQ(vals[2], 3);
}

TEST(WellKnownSpaces, Space3dFirstDummyName)
{
    EXPECT_EQ(space_3d()->dummy_name(0).v.view(), "i");
}

TEST(WellKnownSpaces, Space2dValues)
{
    auto vals = space_2d()->values();
    ASSERT_EQ(vals.size(), 2u);
    EXPECT_EQ(vals[0], 1);
    EXPECT_EQ(vals[1], 2);
}

TEST(WellKnownSpaces, Space2dFirstDummyName)
{
    EXPECT_EQ(space_2d()->dummy_name(0).v.view(), "\\alpha");
}

TEST(WellKnownSpaces, Space4dValues)
{
    auto vals = space_4d()->values();
    ASSERT_EQ(vals.size(), 4u);
    EXPECT_EQ(vals[0], 0);
    EXPECT_EQ(vals[1], 1);
    EXPECT_EQ(vals[2], 2);
    EXPECT_EQ(vals[3], 3);
}

TEST(WellKnownSpaces, Space4dFirstDummyName)
{
    EXPECT_EQ(space_4d()->dummy_name(0).v.view(), "\\mu");
}

TEST(WellKnownSpaces, DistinctInstancesAreDistinct)
{
    EXPECT_NE(space_2d(), space_3d());
    EXPECT_NE(space_3d(), space_4d());
    EXPECT_NE(space_2d(), space_4d());
}

// ---- Slot type ---------------------------------------------------------

TEST(IndexSlot, ConstructionWithSpace)
{
    IndexSlot s{Level::Upper, Realm::Orthonormal, space_3d()};
    EXPECT_EQ(s.level, Level::Upper);
    EXPECT_EQ(s.realm, Realm::Orthonormal);
    EXPECT_EQ(s.space, space_3d());
}

TEST(IndexSlot, NullSpaceForLabelRealm)
{
    IndexSlot s{Level::Lower, Realm::Label, nullptr};
    EXPECT_EQ(s.realm, Realm::Label);
    EXPECT_EQ(s.space, nullptr);
}

// ---- Index association types -------------------------------------------

TEST(CountableIndex, StoresId)
{
    CountableIndex idx{42};
    EXPECT_EQ(idx.id, 42);
}

TEST(ConcreteIndex, StoresValue)
{
    ConcreteIndex idx{3};
    EXPECT_EQ(idx.value, 3);
}

TEST(LabelIndex, StoresName)
{
    LabelIndex idx{make_index_name("vol")};
    EXPECT_EQ(idx.name.v.view(), "vol");
}

TEST(IndexAssoc, HoldsCountableIndex)
{
    IndexAssoc a = CountableIndex{7};
    EXPECT_TRUE(std::holds_alternative<CountableIndex>(a));
    EXPECT_EQ(std::get<CountableIndex>(a).id, 7);
}

TEST(IndexAssoc, HoldsConcreteIndex)
{
    IndexAssoc a = ConcreteIndex{2};
    EXPECT_TRUE(std::holds_alternative<ConcreteIndex>(a));
    EXPECT_EQ(std::get<ConcreteIndex>(a).value, 2);
}

TEST(IndexAssoc, HoldsLabelIndex)
{
    IndexAssoc a = LabelIndex{make_index_name("surf")};
    EXPECT_TRUE(std::holds_alternative<LabelIndex>(a));
    EXPECT_EQ(std::get<LabelIndex>(a).name.v.view(), "surf");
}
