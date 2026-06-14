#include <tender/permutation_spec.hpp>

#include <gtest/gtest.h>

using namespace tender;

// ---- Permutation<N> -------------------------------------------------------

TEST(Permutation, Equality)
{
    EXPECT_EQ((Permutation<3>{{1, 2, 0}}), (Permutation<3>{{1, 2, 0}}));
    EXPECT_NE((Permutation<3>{{1, 2, 0}}), (Permutation<3>{{0, 1, 2}}));
}

// ---- PermutationSpec: default construction --------------------------------

TEST(PermutationSpec, DefaultEmpty)
{
    PermutationSpec ps;
    EXPECT_TRUE(ps.empty());
    EXPECT_EQ(ps.size(), 0u);
    EXPECT_EQ(ps.rank(), 0u);
    EXPECT_EQ(ps.begin(), ps.end());
    EXPECT_EQ(ps.end() - ps.begin(), 0);
}

TEST(PermutationSpec, DefaultSameLevelOnly)
{
    // default same_level_only is true
    EXPECT_TRUE(PermutationSpec{}.same_level_only());
}

// ---- PermutationSpec: one generator ---------------------------------------

TEST(PermutationSpec, OneGenerator)
{
    Permutation<3> cyc{{1, 2, 0}};
    PermutationSpec ps(true, cyc);
    EXPECT_FALSE(ps.empty());
    EXPECT_EQ(ps.size(), 1u);
    EXPECT_EQ(ps.rank(), 3u);
    EXPECT_TRUE(ps.same_level_only());
    EXPECT_EQ(ps.end() - ps.begin(), 1);
}

TEST(PermutationSpec, OneGeneratorValues)
{
    Permutation<3> cyc{{1, 2, 0}};
    PermutationSpec ps(false, cyc);
    auto it = ps.begin();
    EXPECT_EQ((*it)[0], 1u);
    EXPECT_EQ((*it)[1], 2u);
    EXPECT_EQ((*it)[2], 0u);
    EXPECT_EQ((*it).size(), 3u);
    EXPECT_EQ((*it).rank(), 3u);
    EXPECT_FALSE(ps.same_level_only());
}

// ---- PermutationSpec: multiple generators ---------------------------------

TEST(PermutationSpec, ThreeGenerators)
{
    Permutation<4> swap01{{1, 0, 2, 3}};
    Permutation<4> swap23{{0, 1, 3, 2}};
    Permutation<4> swappairs{{2, 3, 0, 1}};
    PermutationSpec ps(true, swap01, swap23, swappairs);
    EXPECT_EQ(ps.size(), 3u);
    EXPECT_EQ(ps.rank(), 4u);

    auto it = ps.begin();
    EXPECT_EQ((*it)[0], 1u);
    EXPECT_EQ((*it)[1], 0u);
    ++it;
    EXPECT_EQ((*it)[0], 0u);
    EXPECT_EQ((*it)[2], 3u);
    ++it;
    EXPECT_EQ((*it)[0], 2u);
    EXPECT_EQ((*it)[2], 0u);
    ++it;
    EXPECT_EQ(it, ps.end());
}

// ---- PermutationSpec: random-access iterator ------------------------------

TEST(PermutationSpec, IndexOperator)
{
    Permutation<2> id{{0, 1}};
    Permutation<2> swap{{1, 0}};
    PermutationSpec ps(false, id, swap);

    EXPECT_EQ(ps.begin()[0][0], 0u);
    EXPECT_EQ(ps.begin()[0][1], 1u);
    EXPECT_EQ(ps.begin()[1][0], 1u);
    EXPECT_EQ(ps.begin()[1][1], 0u);
}

TEST(PermutationSpec, IteratorArithmetic)
{
    Permutation<3> p{{1, 2, 0}};
    Permutation<3> q{{1, 0, 2}};
    PermutationSpec ps(true, p, q);

    auto b = ps.begin();
    auto e = ps.end();
    EXPECT_EQ(e - b, 2);
    EXPECT_EQ(b + 2, e);
    EXPECT_EQ(2 + b, e);
    EXPECT_EQ(e - 2, b);
    EXPECT_EQ((b + 1) - b, 1);
}

TEST(PermutationSpec, IteratorOrdering)
{
    Permutation<2> p{{1, 0}};
    Permutation<2> q{{0, 1}};
    PermutationSpec ps(false, p, q);

    EXPECT_LT(ps.begin(), ps.end());
    EXPECT_GT(ps.end(), ps.begin());
    EXPECT_LE(ps.begin(), ps.begin());
}

TEST(PermutationSpec, PostIncrementDecrement)
{
    Permutation<2> p{{1, 0}};
    Permutation<2> q{{0, 1}};
    PermutationSpec ps(false, p, q);

    auto it = ps.begin();
    auto old = it++;
    EXPECT_EQ(old, ps.begin());
    EXPECT_EQ(it, ps.begin() + 1);
    auto old2 = it--;
    EXPECT_EQ(old2, ps.begin() + 1);
    EXPECT_EQ(it, ps.begin());
}

// ---- PermutationSpec: equality --------------------------------------------

TEST(PermutationSpec, Equality)
{
    Permutation<3> p{{1, 2, 0}};
    PermutationSpec ps1(true, p);
    PermutationSpec ps2(true, p);
    PermutationSpec ps3(false, p);
    EXPECT_EQ(ps1, ps2);
    EXPECT_NE(ps1, ps3);
    EXPECT_NE(ps1, PermutationSpec{});
}
