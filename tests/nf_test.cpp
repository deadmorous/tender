#include <tender/nf.hpp>

#include <tender/index_space.hpp>

#include <gtest/gtest.h>

using namespace tender;
using namespace tender::nf;

namespace
{

// A bare rank-1 atom named `name` with a single lower oblique slot bound to a
// fresh dummy index — enough to exercise builders/equality/hashing.
auto vector_atom(Context& ctx, std::string_view name, int index_id)
    -> Factor const*
{
    std::vector<SlotBinding> slots = {SlotBinding{
        IndexSlot{Level::Lower, Realm::Oblique, space_3d()},
        CountableIndex{index_id}}};
    return make_atom(
        ctx,
        TensorObject{
            .name = make_tensor_name(name),
            .rank = 1,
            .traits = std::nullopt,
            .slots = std::move(slots)});
}

// An abstract (slot-less) atom of the given rank.
auto abstract_atom(Context& ctx, std::string_view name, int rank)
    -> Factor const*
{
    return make_atom(
        ctx,
        TensorObject{
            .name = make_tensor_name(name),
            .rank = rank,
            .traits = std::nullopt,
            .slots = {}});
}

} // namespace

// ---- builders ----------------------------------------------------------

TEST(NfBuilders, Atom)
{
    Context ctx;
    auto const* f = abstract_atom(ctx, "A", 2);
    ASSERT_NE(f, nullptr);
    ASSERT_TRUE(std::holds_alternative<Atom>(f->node));
    EXPECT_EQ(std::get<Atom>(f->node).obj.name.v.view(), "A");
}

TEST(NfBuilders, Contraction)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 2);
    auto const* b = abstract_atom(ctx, "B", 2);
    auto const* c = make_contraction(ctx, {a, b}, {COp::Dot});
    ASSERT_TRUE(std::holds_alternative<Contraction>(c->node));
    auto const& con = std::get<Contraction>(c->node);
    EXPECT_EQ(con.factors.size(), 2u);
    ASSERT_EQ(con.ops.size(), 1u);
    EXPECT_EQ(con.ops[0], COp::Dot);
}

TEST(NfBuilders, ContractionRejectsArityMismatch)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 2);
    auto const* b = abstract_atom(ctx, "B", 2);
    EXPECT_THROW((void)make_contraction(ctx, {a, b}, {}), std::invalid_argument);
    EXPECT_THROW(
        (void)make_contraction(ctx, {a, b}, {COp::Dot, COp::Dot}),
        std::invalid_argument);
    EXPECT_THROW((void)make_contraction(ctx, {}, {}), std::invalid_argument);
}

TEST(NfBuilders, CrossRejectsEmpty)
{
    Context ctx;
    EXPECT_THROW((void)make_cross(ctx, {}), std::invalid_argument);
}

TEST(NfBuilders, ParenRejectsNull)
{
    Context ctx;
    EXPECT_THROW((void)make_paren(ctx, nullptr), std::invalid_argument);
}

TEST(NfBuilders, Paren)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 1);
    auto const* b = abstract_atom(ctx, "B", 1);
    auto const* inner =
        make_nf(ctx, {Term{.tensors = {a}}, Term{.tensors = {b}}});
    auto const* p = make_paren(ctx, inner);
    ASSERT_TRUE(std::holds_alternative<Paren>(p->node));
    EXPECT_EQ(std::get<Paren>(p->node).body, inner);
}

// ---- Term::signed_coeff ------------------------------------------------

TEST(NfTerm, SignedCoeff)
{
    Term pos{.sign = Sign::Pos, .coeff = Rational{3, 2}};
    Term neg{.sign = Sign::Neg, .coeff = Rational{3, 2}};
    EXPECT_EQ(pos.signed_coeff(), (Rational{3, 2}));
    EXPECT_EQ(neg.signed_coeff(), (Rational{-3, 2}));
}

// ---- structural equality -----------------------------------------------

TEST(NfEqual, AtomsByContents)
{
    Context ctx;
    auto const* a1 = vector_atom(ctx, "a", 7);
    auto const* a2 = vector_atom(ctx, "a", 7);
    auto const* a3 = vector_atom(ctx, "a", 8); // different index id
    auto const* b = vector_atom(ctx, "b", 7);  // different name
    EXPECT_TRUE(equal(a1, a2));
    EXPECT_FALSE(equal(a1, a3));
    EXPECT_FALSE(equal(a1, b));
}

TEST(NfEqual, DistinctVariants)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 2);
    auto const* b = abstract_atom(ctx, "B", 2);
    auto const* con = make_contraction(ctx, {a, b}, {COp::Dot});
    auto const* cross = make_cross(ctx, {a, b});
    EXPECT_FALSE(equal(con, cross));
}

TEST(NfEqual, ContractionOpSensitive)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 2);
    auto const* b = abstract_atom(ctx, "B", 2);
    auto const* dot = make_contraction(ctx, {a, b}, {COp::Dot});
    auto const* ddot = make_contraction(ctx, {a, b}, {COp::DDot});
    auto const* dot2 = make_contraction(
        ctx,
        {abstract_atom(ctx, "A", 2), abstract_atom(ctx, "B", 2)},
        {COp::Dot});
    EXPECT_FALSE(equal(dot, ddot));
    EXPECT_TRUE(equal(dot, dot2));
}

TEST(NfEqual, ParenRecurses)
{
    Context ctx;
    auto build = [&](std::string_view n)
    {
        return make_paren(
            ctx, make_nf(ctx, {Term{.tensors = {abstract_atom(ctx, n, 1)}}}));
    };
    EXPECT_TRUE(equal(build("A"), build("A")));
    EXPECT_FALSE(equal(build("A"), build("B")));
}

TEST(NfEqual, TermSignAndCoeff)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 1);
    Term t1{.sign = Sign::Pos, .coeff = Rational{2}, .tensors = {a}};
    Term t2{.sign = Sign::Pos, .coeff = Rational{2}, .tensors = {a}};
    Term t3{.sign = Sign::Neg, .coeff = Rational{2}, .tensors = {a}};
    Term t4{.sign = Sign::Pos, .coeff = Rational{3}, .tensors = {a}};
    EXPECT_TRUE(equal(t1, t2));
    EXPECT_FALSE(equal(t1, t3));
    EXPECT_FALSE(equal(t1, t4));
}

TEST(NfEqual, TermBoundIndicesAndModes)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 1);
    Term t1{.bound = {{CountableIndex{1}, SumMode::Default}}, .tensors = {a}};
    Term t2{.bound = {{CountableIndex{1}, SumMode::Default}}, .tensors = {a}};
    Term t3{.bound = {{CountableIndex{1}, SumMode::NoSum}}, .tensors = {a}};
    Term t4{.bound = {{CountableIndex{2}, SumMode::Default}}, .tensors = {a}};
    EXPECT_TRUE(equal(t1, t2));
    EXPECT_FALSE(equal(t1, t3));
    EXPECT_FALSE(equal(t1, t4));
}

TEST(NfEqual, ScalarsArePositional)
{
    // equal() does not reorder; scalar sorting is canon's job, not equality's.
    Context ctx;
    auto const* p = abstract_atom(ctx, "A", 0);
    auto const* q = abstract_atom(ctx, "B", 0);
    Term t1{.scalars = {p, q}};
    Term t2{.scalars = {q, p}};
    EXPECT_FALSE(equal(t1, t2));
}

TEST(NfEqual, NfByTermSequence)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 1);
    auto const* b = abstract_atom(ctx, "B", 1);
    auto const* n1 = make_nf(ctx, {Term{.tensors = {a}}, Term{.tensors = {b}}});
    auto const* n2 = make_nf(ctx, {Term{.tensors = {a}}, Term{.tensors = {b}}});
    auto const* n3 = make_nf(ctx, {Term{.tensors = {b}}, Term{.tensors = {a}}});
    auto const* n4 = make_nf(ctx, {Term{.tensors = {a}}});
    EXPECT_TRUE(equal(n1, n2));
    EXPECT_FALSE(equal(n1, n3));
    EXPECT_FALSE(equal(n1, n4));
}

// ---- structural hashing ------------------------------------------------

TEST(NfHash, EqualStructuresHashEqual)
{
    Context ctx;
    auto const* a1 = vector_atom(ctx, "a", 7);
    auto const* a2 = vector_atom(ctx, "a", 7);
    EXPECT_EQ(hash(*a1), hash(*a2));

    auto const* c1 = make_contraction(
        ctx,
        {abstract_atom(ctx, "A", 2), abstract_atom(ctx, "B", 2)},
        {COp::Dot});
    auto const* c2 = make_contraction(
        ctx,
        {abstract_atom(ctx, "A", 2), abstract_atom(ctx, "B", 2)},
        {COp::Dot});
    EXPECT_EQ(hash(*c1), hash(*c2));

    Term t1{.sign = Sign::Neg, .coeff = Rational{2}, .tensors = {a1}};
    Term t2{.sign = Sign::Neg, .coeff = Rational{2}, .tensors = {a2}};
    EXPECT_EQ(hash(t1), hash(t2));

    auto const* n1 = make_nf(ctx, {t1});
    auto const* n2 = make_nf(ctx, {t2});
    EXPECT_EQ(hash(*n1), hash(*n2));
}

TEST(NfHash, DistinctStructuresHashDistinct)
{
    // Not contractually required, but a healthy hash should separate these.
    Context ctx;
    auto const* dot = make_contraction(
        ctx,
        {abstract_atom(ctx, "A", 2), abstract_atom(ctx, "B", 2)},
        {COp::Dot});
    auto const* ddot = make_contraction(
        ctx,
        {abstract_atom(ctx, "A", 2), abstract_atom(ctx, "B", 2)},
        {COp::DDot});
    EXPECT_NE(hash(*dot), hash(*ddot));

    Term pos{.sign = Sign::Pos, .coeff = Rational{2}};
    Term neg{.sign = Sign::Neg, .coeff = Rational{2}};
    EXPECT_NE(hash(pos), hash(neg));
}
