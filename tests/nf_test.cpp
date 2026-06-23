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

TEST(NfBuilders, Unary)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 2);
    auto const* tr = make_unary(ctx, UnaryOp::Trace, a);
    ASSERT_TRUE(std::holds_alternative<Unary>(tr->node));
    EXPECT_EQ(std::get<Unary>(tr->node).op, UnaryOp::Trace);
    EXPECT_EQ(std::get<Unary>(tr->node).operand, a);
    EXPECT_THROW(
        (void)make_unary(ctx, UnaryOp::Trace, nullptr), std::invalid_argument);
}

TEST(NfFactor, UnaryEqualityCompareHash)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 2);
    auto const* b = abstract_atom(ctx, "B", 2);
    auto const* tr_a = make_unary(ctx, UnaryOp::Trace, a);
    auto const* tr_a2 = make_unary(ctx, UnaryOp::Trace, a);
    auto const* tp_a = make_unary(ctx, UnaryOp::Transpose, a);
    auto const* tr_b = make_unary(ctx, UnaryOp::Trace, b);
    EXPECT_TRUE(equal(tr_a, tr_a2));
    EXPECT_FALSE(equal(tr_a, tp_a)); // different op
    EXPECT_FALSE(equal(tr_a, tr_b)); // different operand
    EXPECT_EQ(hash(*tr_a), hash(*tr_a2));
    // Unary sorts after Atom (later variant tag); consistent with equal.
    EXPECT_GT(compare(*tr_a, *a), 0);
    EXPECT_EQ(compare(*tr_a, *tr_a2), 0);
    EXPECT_NE(compare(*tr_a, *tp_a), 0);
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

TEST(NfEqual, TermCoeff)
{
    // The sign lives in the (signed) coeff; there is no separate Sign field.
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 1);
    Term t1{.coeff = Rational{2}, .tensors = {a}};
    Term t2{.coeff = Rational{2}, .tensors = {a}};
    Term t3{.coeff = Rational{-2}, .tensors = {a}};
    Term t4{.coeff = Rational{3}, .tensors = {a}};
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

// ---- total order -------------------------------------------------------

TEST(NfCompare, FactorsByVariantTag)
{
    // Atom < Contraction < Cross < Paren (variant declaration order).
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 2);
    auto const* b = abstract_atom(ctx, "B", 2);
    auto const* atom = abstract_atom(ctx, "A", 2);
    auto const* con = make_contraction(ctx, {a, b}, {COp::Dot});
    auto const* cross = make_cross(ctx, {a, b});
    auto const* paren = make_paren(ctx, make_nf(ctx, {Term{.tensors = {a}}}));
    EXPECT_LT(compare(*atom, *con), 0);
    EXPECT_LT(compare(*con, *cross), 0);
    EXPECT_LT(compare(*cross, *paren), 0);
    EXPECT_GT(compare(*paren, *atom), 0);
}

TEST(NfCompare, AtomsByName)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 1);
    auto const* b = abstract_atom(ctx, "B", 1);
    EXPECT_LT(compare(*a, *b), 0);
    EXPECT_GT(compare(*b, *a), 0);
    EXPECT_EQ(compare(*a, *abstract_atom(ctx, "A", 1)), 0);
}

TEST(NfCompare, ContractionByOpsWhenFactorsEqual)
{
    Context ctx;
    auto dot = [&]
    {
        return make_contraction(
            ctx,
            {abstract_atom(ctx, "A", 2), abstract_atom(ctx, "B", 2)},
            {COp::Dot});
    };
    auto ddot = [&]
    {
        return make_contraction(
            ctx,
            {abstract_atom(ctx, "A", 2), abstract_atom(ctx, "B", 2)},
            {COp::DDot});
    };
    EXPECT_LT(compare(*dot(), *ddot()), 0); // Dot < DDot
    EXPECT_EQ(compare(*dot(), *dot()), 0);
}

TEST(NfCompare, TermsByTensorShapeThenCoeff)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 1);
    auto const* b = abstract_atom(ctx, "B", 1);
    // Same tensor shape ⇒ ordered by coeff.
    Term t2{.coeff = Rational{2}, .tensors = {a}};
    Term t5{.coeff = Rational{5}, .tensors = {a}};
    EXPECT_LT(compare(t2, t5), 0);
    // Different tensor shape dominates the coeff.
    Term tb{.coeff = Rational{100}, .tensors = {b}};
    EXPECT_LT(compare(t5, tb), 0); // A-shape < B-shape regardless of coeff
}

TEST(NfCompare, NfByTermSequenceLength)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 1);
    auto const* n1 = make_nf(ctx, {Term{.tensors = {a}}});
    auto const* n2 = make_nf(ctx, {Term{.tensors = {a}}, Term{.tensors = {a}}});
    EXPECT_LT(compare(*n1, *n2), 0);
}

TEST(NfCompare, ConsistentWithEqualAndAntisymmetric)
{
    Context ctx;
    auto const* a = abstract_atom(ctx, "A", 2);
    auto const* b = abstract_atom(ctx, "B", 2);
    std::vector<Factor const*> fs = {
        a,
        b,
        make_contraction(ctx, {a, b}, {COp::Dot}),
        make_cross(ctx, {a, b}),
        make_paren(ctx, make_nf(ctx, {Term{.tensors = {a}}})),
    };
    for (auto const* x: fs)
        for (auto const* y: fs)
        {
            EXPECT_EQ(compare(*x, *y) == 0, equal(x, y));
            EXPECT_EQ(compare(*x, *y), -compare(*y, *x));
        }
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

    Term t1{.coeff = Rational{-2}, .tensors = {a1}};
    Term t2{.coeff = Rational{-2}, .tensors = {a2}};
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

    Term pos{.coeff = Rational{2}};
    Term neg{.coeff = Rational{-2}};
    EXPECT_NE(hash(pos), hash(neg));
}
