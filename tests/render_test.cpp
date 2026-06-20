#include <tender/render.hpp>

#include <gtest/gtest.h>

using namespace tender;

// ---- helpers -----------------------------------------------------------

// Make a named abstract tensor with optional rank.
static auto T(
    Context& ctx,
    std::string_view name,
    std::optional<int> rank = std::nullopt) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, rank);
}

// Shorthand for render_latex with a fresh map.
static auto latex(Expr const& e) -> std::string
{
    IndexNameMap map;
    return render_latex(e, map);
}

// ---- IndexNameMap ------------------------------------------------------

TEST(IndexNameMap, AssignAndLookup)
{
    IndexNameMap map;
    CountableIndex ci{42};
    map.assign(ci, make_index_name("k"));
    auto name = map.lookup(ci);
    ASSERT_TRUE(name.has_value());
    EXPECT_EQ(name->v.view(), "k");
}

TEST(IndexNameMap, IndexFor)
{
    IndexNameMap map;
    CountableIndex ci{7};
    map.assign(ci, make_index_name("m"));
    auto found = map.index_for(make_index_name("m"));
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->id, 7);
}

TEST(IndexNameMap, LookupAbsent)
{
    IndexNameMap map;
    EXPECT_FALSE(map.lookup(CountableIndex{99}).has_value());
}

TEST(IndexNameMap, IndexForAbsent)
{
    IndexNameMap map;
    EXPECT_FALSE(map.index_for(make_index_name("z")).has_value());
}

TEST(IndexNameMap, ReassignId)
{
    IndexNameMap map;
    CountableIndex ci{1};
    map.assign(ci, make_index_name("i"));
    map.assign(ci, make_index_name("j")); // replace
    EXPECT_EQ(map.lookup(ci)->v.view(), "j");
    EXPECT_FALSE(map.index_for(make_index_name("i")).has_value());
}

TEST(IndexNameMap, ReassignNameMovesOldId)
{
    IndexNameMap map;
    CountableIndex ci1{1};
    CountableIndex ci2{2};
    map.assign(ci1, make_index_name("i"));
    map.assign(ci2, make_index_name("i")); // steal name "i" from ci1
    EXPECT_EQ(map.lookup(ci2)->v.view(), "i");
    EXPECT_FALSE(map.lookup(ci1).has_value());
}

TEST(IndexNameMap, AutoAllocFromSchema)
{
    IndexNameMap map;
    Context ctx;
    CountableIndex ci{ctx.alloc_index_id()};
    // Allocate by rendering; uses space_3d schema starting from "i"
    auto name = map.name_for(ci, space_3d());
    EXPECT_EQ(name.v.view(), "i");
    // Same id → same name
    EXPECT_EQ(map.name_for(ci, space_3d()).v.view(), "i");
}

TEST(IndexNameMap, AutoAllocSkipsAssigned)
{
    IndexNameMap map;
    Context ctx;
    // Pre-assign the first schema name
    CountableIndex ci0{ctx.alloc_index_id()};
    CountableIndex ci1{ctx.alloc_index_id()};
    map.assign(ci0, make_index_name("i")); // "i" is taken
    // Next allocation should skip "i" and give "j"
    auto name = map.name_for(ci1, space_3d());
    EXPECT_EQ(name.v.view(), "j");
}

TEST(IndexNameMap, ReuseAcrossRenderCalls)
{
    Context ctx;
    CountableIndex ci{ctx.alloc_index_id()};
    std::vector<SlotBinding> slots = {SlotBinding{
        IndexSlot{Level::Upper, Realm::Orthonormal, space_3d()}, ci}};
    auto* e =
        make_tensor_object(ctx, make_tensor_name("v"), std::move(slots), 1);

    IndexNameMap map;
    auto s1 = render_latex(*e, map);
    auto s2 = render_latex(*e, map);
    EXPECT_EQ(s1, s2); // same map → same name
}

// ---- ScalarLiteral -----------------------------------------------------

TEST(RenderScalar, Integer)
{
    Context ctx;
    EXPECT_EQ(latex(*make_scalar(ctx, Rational{3})), "3");
}

TEST(RenderScalar, NegativeInteger)
{
    Context ctx;
    EXPECT_EQ(latex(*make_scalar(ctx, Rational{-2})), "-2");
}

TEST(RenderScalar, Fraction)
{
    Context ctx;
    EXPECT_EQ(latex(*make_scalar(ctx, Rational{3, 2})), "\\frac{3}{2}");
}

TEST(RenderScalar, NegativeFraction)
{
    Context ctx;
    EXPECT_EQ(latex(*make_scalar(ctx, Rational{-1, 3})), "\\frac{-1}{3}");
}

// ---- TensorObject name & boldface --------------------------------------

TEST(RenderTensorObject, Rank0Plain)
{
    Context ctx;
    EXPECT_EQ(latex(*T(ctx, "f", 0)), "f");
}

TEST(RenderTensorObject, Rank1Bold)
{
    Context ctx;
    EXPECT_EQ(latex(*T(ctx, "v", 1)), "\\mathbf{v}");
}

TEST(RenderTensorObject, RankNulloptBold)
{
    Context ctx;
    EXPECT_EQ(latex(*T(ctx, "T")), "\\mathbf{T}");
}

TEST(RenderTensorObject, LatexCommandBold)
{
    Context ctx;
    EXPECT_EQ(latex(*T(ctx, "\\sigma", 1)), "\\boldsymbol{\\sigma}");
}

TEST(RenderTensorObject, IdentityBold)
{
    Context ctx;
    EXPECT_EQ(latex(*make_identity(ctx)), "\\mathbf{I}");
}

// ---- TensorObject index slots ------------------------------------------

TEST(RenderTensorObject, UpperIndex)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    std::vector<SlotBinding> slots = {
        SlotBinding{IndexSlot{Level::Upper, Realm::Orthonormal, space_3d()}, i}};
    auto* e =
        make_tensor_object(ctx, make_tensor_name("v"), std::move(slots), 1);

    IndexNameMap map;
    map.assign(i, make_index_name("i"));
    EXPECT_EQ(render_latex(*e, map), "\\mathbf{v}^{i}");
}

TEST(RenderTensorObject, LowerIndex)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    std::vector<SlotBinding> slots = {
        SlotBinding{IndexSlot{Level::Lower, Realm::Orthonormal, space_3d()}, i}};
    auto* e =
        make_tensor_object(ctx, make_tensor_name("v"), std::move(slots), 1);

    IndexNameMap map;
    map.assign(i, make_index_name("i"));
    EXPECT_EQ(render_latex(*e, map), "\\mathbf{v}_{i}");
}

TEST(RenderTensorObject, UpperAndLowerStacked)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    std::vector<SlotBinding> slots = {
        SlotBinding{IndexSlot{Level::Upper, Realm::Oblique, space_3d()}, i},
        SlotBinding{IndexSlot{Level::Lower, Realm::Oblique, space_3d()}, j}};
    auto* e =
        make_tensor_object(ctx, make_tensor_name("T"), std::move(slots), 2);

    IndexNameMap map;
    map.assign(i, make_index_name("i"));
    map.assign(j, make_index_name("j"));
    EXPECT_EQ(render_latex(*e, map), "\\mathbf{T}^{i\\cdot}_{\\cdot j}");
}

TEST(RenderTensorObject, ConcreteIndex)
{
    Context ctx;
    std::vector<SlotBinding> slots = {SlotBinding{
        IndexSlot{Level::Upper, Realm::Orthonormal, space_3d()},
        ConcreteIndex{2}}};
    auto* e =
        make_tensor_object(ctx, make_tensor_name("v"), std::move(slots), 1);
    EXPECT_EQ(latex(*e), "\\mathbf{v}^{2}");
}

TEST(RenderTensorObject, LabelIndex)
{
    Context ctx;
    std::vector<SlotBinding> slots = {SlotBinding{
        IndexSlot{Level::Lower, Realm::Label, nullptr},
        LabelIndex{make_index_name("vol")}}};
    auto* e = make_tensor_object(ctx, make_tensor_name("A"), std::move(slots));
    EXPECT_EQ(latex(*e), "\\mathbf{A}_{vol}");
}

TEST(RenderTensorObject, NulloptIndexSlot)
{
    Context ctx;
    std::vector<SlotBinding> slots = {SlotBinding{
        IndexSlot{Level::Upper, Realm::Orthonormal, space_3d()}, std::nullopt}};
    auto* e =
        make_tensor_object(ctx, make_tensor_name("v"), std::move(slots), 1);
    EXPECT_EQ(latex(*e), "\\mathbf{v}^{\\bullet}");
}

TEST(RenderTensorObject, DeltaWithIndices)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto* e = make_delta(
        ctx, Realm::Oblique, space_3d(), Level::Upper, Level::Lower, i, j);

    IndexNameMap map;
    map.assign(i, make_index_name("i"));
    map.assign(j, make_index_name("j"));
    EXPECT_EQ(render_latex(*e, map), "\\delta^{i}_{j}");
}

TEST(RenderTensorObject, LeviCivitaRank3)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto k = CountableIndex{ctx.alloc_index_id()};
    auto* e = make_levi_civita(
        ctx,
        Realm::Orthonormal,
        space_3d(),
        {Level::Lower, Level::Lower, Level::Lower},
        {i, j, k});

    IndexNameMap map;
    map.assign(i, make_index_name("i"));
    map.assign(j, make_index_name("j"));
    map.assign(k, make_index_name("k"));
    EXPECT_EQ(render_latex(*e, map), "\\varepsilon_{ijk}");
}

// ---- Unary -------------------------------------------------------------

TEST(RenderUnary, NegateAtom)
{
    Context ctx;
    EXPECT_EQ(latex(*make_negate(ctx, make_scalar(ctx, Rational{3}))), "-3");
}

TEST(RenderUnary, NegateVector)
{
    Context ctx;
    EXPECT_EQ(latex(*make_negate(ctx, T(ctx, "v", 1))), "-\\mathbf{v}");
}

TEST(RenderUnary, NegateSum)
{
    Context ctx;
    auto* s = make_sum(ctx, T(ctx, "a", 1), T(ctx, "b", 1));
    EXPECT_EQ(latex(*make_negate(ctx, s)), "-(\\mathbf{a} + \\mathbf{b})");
}

TEST(RenderUnary, NegateTensorProduct)
{
    // No parens needed: mul binds tighter than unary minus.
    Context ctx;
    auto* p = make_tensor_product(ctx, T(ctx, "a", 1), T(ctx, "b", 1));
    EXPECT_EQ(latex(*make_negate(ctx, p)), "-\\mathbf{a} \\, \\mathbf{b}");
}

// ---- Binary operations -------------------------------------------------

TEST(RenderBinary, Sum)
{
    Context ctx;
    EXPECT_EQ(
        latex(*make_sum(ctx, T(ctx, "a", 1), T(ctx, "b", 1))),
        "\\mathbf{a} + \\mathbf{b}");
}

TEST(RenderBinary, SumWithNegatedRight)
{
    // Sum(A, Negate(B)) should render as subtraction, not "A + (-B)".
    Context ctx;
    auto* neg = make_negate(ctx, T(ctx, "b", 1));
    EXPECT_EQ(
        latex(*make_sum(ctx, T(ctx, "a", 1), neg)),
        "\\mathbf{a} - \\mathbf{b}");
}

TEST(RenderBinary, SumWithNegatedProductRight)
{
    // Sum(A, Negate(B*C)) → "A - B \, C", no double parens.
    Context ctx;
    auto* p = make_tensor_product(ctx, T(ctx, "b", 1), T(ctx, "c", 1));
    auto* neg = make_negate(ctx, p);
    EXPECT_EQ(
        latex(*make_sum(ctx, T(ctx, "a", 1), neg)),
        "\\mathbf{a} - \\mathbf{b} \\, \\mathbf{c}");
}

TEST(RenderBinary, Difference)
{
    Context ctx;
    EXPECT_EQ(
        latex(*make_difference(ctx, T(ctx, "a", 1), T(ctx, "b", 1))),
        "\\mathbf{a} - \\mathbf{b}");
}

TEST(RenderBinary, TensorProduct)
{
    Context ctx;
    EXPECT_EQ(
        latex(*make_tensor_product(ctx, T(ctx, "a", 1), T(ctx, "b", 1))),
        "\\mathbf{a} \\, \\mathbf{b}");
}

TEST(RenderBinary, TensorProductAllScalarsUsesCdot)
{
    // 2 * 3 * 4 — left-associative tree with scalar chain — all \cdot.
    Context ctx;
    auto* s1 = make_scalar(ctx, Rational{2});
    auto* s2 = make_scalar(ctx, Rational{3});
    auto* s3 = make_scalar(ctx, Rational{4});
    auto* inner = make_tensor_product(ctx, s1, s2);
    EXPECT_EQ(
        latex(*make_tensor_product(ctx, inner, s3)), "2 \\cdot 3 \\cdot 4");
}

TEST(RenderBinary, Dot)
{
    Context ctx;
    EXPECT_EQ(
        latex(*make_dot(ctx, T(ctx, "a", 1), T(ctx, "b", 1))),
        "\\mathbf{a} \\cdot \\mathbf{b}");
}

TEST(RenderBinary, Cross)
{
    Context ctx;
    EXPECT_EQ(
        latex(*make_cross(ctx, T(ctx, "a", 1), T(ctx, "b", 1))),
        "\\mathbf{a} \\times \\mathbf{b}");
}

TEST(RenderBinary, DDot)
{
    Context ctx;
    EXPECT_EQ(
        latex(*make_ddot(ctx, T(ctx, "A", 2), T(ctx, "B", 2))),
        "\\mathbf{A} : \\mathbf{B}");
}

TEST(RenderBinary, DDotAlt)
{
    Context ctx;
    EXPECT_EQ(
        latex(*make_ddot_alt(ctx, T(ctx, "A", 2), T(ctx, "B", 2))),
        "\\mathbf{A} \\cdot\\!\\cdot \\mathbf{B}");
}

TEST(RenderBinary, ScalarDiv)
{
    Context ctx;
    EXPECT_EQ(
        latex(*make_scalar_div(
            ctx, T(ctx, "v", 1), make_scalar(ctx, Rational{2}))),
        "\\frac{\\mathbf{v}}{2}");
}

// ---- Sum annotation nodes ----------------------------------------------

TEST(RenderAnnotation, ExplicitSum)
{
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    std::vector<SlotBinding> slots = {
        SlotBinding{IndexSlot{Level::Upper, Realm::Orthonormal, space_3d()}, i}};
    auto* v =
        make_tensor_object(ctx, make_tensor_name("v"), std::move(slots), 1);
    auto* e = make_explicit_sum(ctx, i, v);

    IndexNameMap map;
    map.assign(i, make_index_name("i"));
    EXPECT_EQ(render_latex(*e, map), "\\sum_{i} \\mathbf{v}^{i}");
}

TEST(RenderAnnotation, ExplicitSumBodyIsSum)
{
    // ExplicitSum body that is itself a Sum: body should be parenthesised.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    std::vector<SlotBinding> sa = {
        SlotBinding{IndexSlot{Level::Upper, Realm::Orthonormal, space_3d()}, i}};
    std::vector<SlotBinding> sb = {
        SlotBinding{IndexSlot{Level::Upper, Realm::Orthonormal, space_3d()}, i}};
    auto* va = make_tensor_object(ctx, make_tensor_name("a"), std::move(sa), 1);
    auto* vb = make_tensor_object(ctx, make_tensor_name("b"), std::move(sb), 1);
    auto* body = make_sum(ctx, va, vb);
    auto* e = make_explicit_sum(ctx, i, body);

    IndexNameMap map;
    map.assign(i, make_index_name("i"));
    EXPECT_EQ(
        render_latex(*e, map), "\\sum_{i} (\\mathbf{a}^{i} + \\mathbf{b}^{i})");
}

TEST(RenderAnnotation, NoSum)
{
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    std::vector<SlotBinding> slots = {
        SlotBinding{IndexSlot{Level::Upper, Realm::Orthonormal, space_3d()}, i}};
    auto* v =
        make_tensor_object(ctx, make_tensor_name("v"), std::move(slots), 1);
    auto* e = make_no_sum(ctx, i, v);

    IndexNameMap map;
    map.assign(i, make_index_name("i"));
    EXPECT_EQ(render_latex(*e, map), "\\cancel{\\sum}_{i} \\mathbf{v}^{i}");
}

// ---- Parenthesisation --------------------------------------------------

TEST(RenderParens, SumInsideTensorProduct)
{
    Context ctx;
    auto* s = make_sum(ctx, T(ctx, "a", 1), T(ctx, "b", 1));
    auto* e = make_tensor_product(ctx, s, T(ctx, "c", 1));
    EXPECT_EQ(latex(*e), "(\\mathbf{a} + \\mathbf{b}) \\, \\mathbf{c}");
}

TEST(RenderParens, DiffRightIsSum)
{
    // a - (b + c) must keep parens to avoid reading as (a-b)+c.
    Context ctx;
    auto* s = make_sum(ctx, T(ctx, "b", 1), T(ctx, "c", 1));
    auto* e = make_difference(ctx, T(ctx, "a", 1), s);
    EXPECT_EQ(latex(*e), "\\mathbf{a} - (\\mathbf{b} + \\mathbf{c})");
}

TEST(RenderParens, DiffRightIsDiff)
{
    // a - (b - c) must keep parens.
    Context ctx;
    auto* inner = make_difference(ctx, T(ctx, "b", 1), T(ctx, "c", 1));
    auto* e = make_difference(ctx, T(ctx, "a", 1), inner);
    EXPECT_EQ(latex(*e), "\\mathbf{a} - (\\mathbf{b} - \\mathbf{c})");
}

TEST(RenderParens, DiffRightIsMul)
{
    // a - b*c: no parens needed (mul binds tighter than diff).
    Context ctx;
    auto* p = make_tensor_product(ctx, T(ctx, "b", 1), T(ctx, "c", 1));
    auto* e = make_difference(ctx, T(ctx, "a", 1), p);
    EXPECT_EQ(latex(*e), "\\mathbf{a} - \\mathbf{b} \\, \\mathbf{c}");
}

TEST(RenderParens, SumNestedInSum)
{
    // (a+b)+c renders without redundant parens.
    Context ctx;
    auto* inner = make_sum(ctx, T(ctx, "a", 1), T(ctx, "b", 1));
    auto* e = make_sum(ctx, inner, T(ctx, "c", 1));
    EXPECT_EQ(latex(*e), "\\mathbf{a} + \\mathbf{b} + \\mathbf{c}");
}

TEST(RenderParens, CrossRightNeedsParens)
{
    // a × (b × c): the right operand of a non-associative cross is wrapped,
    // since a × b × c reads as (a × b) × c.
    Context ctx;
    auto* bc = make_cross(ctx, T(ctx, "b", 1), T(ctx, "c", 1));
    auto* e = make_cross(ctx, T(ctx, "a", 1), bc);
    EXPECT_EQ(
        latex(*e), "\\mathbf{a} \\times (\\mathbf{b} \\times \\mathbf{c})");
}

TEST(RenderParens, CrossLeftNoParens)
{
    // (a × b) × c renders as a × b × c (left-associative reading).
    Context ctx;
    auto* ab = make_cross(ctx, T(ctx, "a", 1), T(ctx, "b", 1));
    auto* e = make_cross(ctx, ab, T(ctx, "c", 1));
    EXPECT_EQ(latex(*e), "\\mathbf{a} \\times \\mathbf{b} \\times \\mathbf{c}");
}

TEST(RenderParens, DotOfCrossNeedsParens)
{
    // a · (b × c): the cross on the right is wrapped.
    Context ctx;
    auto* bc = make_cross(ctx, T(ctx, "b", 1), T(ctx, "c", 1));
    auto* e = make_dot(ctx, T(ctx, "a", 1), bc);
    EXPECT_EQ(latex(*e), "\\mathbf{a} \\cdot (\\mathbf{b} \\times \\mathbf{c})");
}

TEST(RenderParens, TensorProductRightNoParens)
{
    // a ⊗ (b ⊗ c): tensor product is associative — no parens.
    Context ctx;
    auto* bc = make_tensor_product(ctx, T(ctx, "b", 1), T(ctx, "c", 1));
    auto* e = make_tensor_product(ctx, T(ctx, "a", 1), bc);
    EXPECT_EQ(latex(*e), "\\mathbf{a} \\, \\mathbf{b} \\, \\mathbf{c}");
}
