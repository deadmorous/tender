#include <tender/name.hpp>

#include <gtest/gtest.h>

using namespace tender;

// ---- make_tensor_name --------------------------------------------------

TEST(MakeTensorName, SingleLetterAccepted)
{
    auto n = make_tensor_name("A");
    EXPECT_EQ(n.v.view(), "A");
}

TEST(MakeTensorName, AllCasesAccepted)
{
    EXPECT_NO_THROW(make_tensor_name("a"));
    EXPECT_NO_THROW(make_tensor_name("Z"));
}

TEST(MakeTensorName, LatexCommandAccepted)
{
    auto n = make_tensor_name("\\sigma");
    EXPECT_EQ(n.v.view(), "\\sigma");
    EXPECT_NO_THROW(make_tensor_name("\\Phi"));
    EXPECT_NO_THROW(make_tensor_name("\\varepsilon"));
}

TEST(MakeTensorName, MultiLetterWordRejected)
{
    EXPECT_THROW(make_tensor_name("ab"), std::invalid_argument);
    EXPECT_THROW(make_tensor_name("sigma"), std::invalid_argument);
}

TEST(MakeTensorName, EmptyRejected)
{
    EXPECT_THROW(make_tensor_name(""), std::invalid_argument);
}

TEST(MakeTensorName, DigitRejected)
{
    EXPECT_THROW(make_tensor_name("1"), std::invalid_argument);
    EXPECT_THROW(make_tensor_name("A1"), std::invalid_argument);
}

TEST(MakeTensorName, BareBackslashRejected)
{
    EXPECT_THROW(make_tensor_name("\\"), std::invalid_argument);
}

TEST(MakeTensorName, LatexCommandWithDigitRejected)
{
    EXPECT_THROW(make_tensor_name("\\sig3ma"), std::invalid_argument);
}

// ---- make_index_name ---------------------------------------------------

TEST(MakeIndexName, SingleLetterAccepted)
{
    auto n = make_index_name("i");
    EXPECT_EQ(n.v.view(), "i");
}

TEST(MakeIndexName, LatexCommandAccepted)
{
    auto n = make_index_name("\\mu");
    EXPECT_EQ(n.v.view(), "\\mu");
    EXPECT_NO_THROW(make_index_name("\\alpha"));
}

TEST(MakeIndexName, MultiLetterWordAccepted)
{
    auto n = make_index_name("vol");
    EXPECT_EQ(n.v.view(), "vol");
    EXPECT_NO_THROW(make_index_name("surf"));
    EXPECT_NO_THROW(make_index_name("ref"));
}

TEST(MakeIndexName, EmptyRejected)
{
    EXPECT_THROW(make_index_name(""), std::invalid_argument);
}

TEST(MakeIndexName, WordWithDigitRejected)
{
    EXPECT_THROW(make_index_name("a1"), std::invalid_argument);
    EXPECT_THROW(make_index_name("1a"), std::invalid_argument);
}

TEST(MakeIndexName, BareBackslashRejected)
{
    EXPECT_THROW(make_index_name("\\"), std::invalid_argument);
}

// ---- Equality ----------------------------------------------------------

TEST(TensorName, EqualityBySameString)
{
    auto a = make_tensor_name("A");
    auto b = make_tensor_name("A");
    EXPECT_EQ(a, b);
}

TEST(TensorName, InequalityByDifferentString)
{
    auto a = make_tensor_name("A");
    auto b = make_tensor_name("B");
    EXPECT_NE(a, b);
}

TEST(IndexName, EqualityBySameString)
{
    auto a = make_index_name("vol");
    auto b = make_index_name("vol");
    EXPECT_EQ(a, b);
}

// ---- Type distinctness -------------------------------------------------
// TensorName and IndexName are different types; this is verified at compile
// time by checking they are not the same type.

TEST(NameTypes, TensorNameAndIndexNameAreDistinct)
{
    static_assert(!std::is_same_v<TensorName, IndexName>);
}
