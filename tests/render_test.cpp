#include <tender/basis.hpp>
#include <tender/coord_system.hpp>
#include <tender/derivation.hpp>
#include <tender/index_space.hpp>
#include <tender/nf_lower.hpp>
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

// The `\cdot` (scalar·scalar) vs `\,` (juxtaposition) product separator is
// chosen by is_scalar_expr, which classifies every node kind.  Render a product
// of each kind against a scalar literal and check the separator: a `\,` appears
// iff the left operand is non-scalar.  This drives every is_scalar_expr arm.
TEST(RenderBinary, ProductSeparatorClassifiesEveryNodeKind)
{
    Context ctx;
    auto* lit = make_scalar(ctx, Rational{5});
    auto* A = make_field(ctx, make_tensor_name("A"), 2, {});
    auto* a = T(ctx, "a", 1);
    auto* b = T(ctx, "b", 1);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0, false);
    CountableIndex idx{ctx.alloc_index_id()};

    // Rendering TP(node, lit): the separator is `\,` exactly when `node` is
    // non-scalar per is_scalar_expr.
    auto non_scalar = [&](Expr const* node)
    {
        auto s = latex(*make_tensor_product(ctx, node, lit));
        return s.find(" \\, ") != std::string::npos;
    };

    // Scalar-valued kinds → `\cdot`, no `\,`.
    EXPECT_FALSE(non_scalar(make_negate(ctx, lit)));
    EXPECT_FALSE(non_scalar(make_sum(ctx, lit, lit)));
    EXPECT_FALSE(non_scalar(make_difference(ctx, lit, lit)));
    EXPECT_FALSE(non_scalar(make_scalar_div(ctx, lit, lit)));
    EXPECT_FALSE(non_scalar(make_pow(ctx, lit, lit)));
    EXPECT_FALSE(non_scalar(make_explicit_sum(ctx, idx, lit)));
    EXPECT_FALSE(non_scalar(make_no_sum(ctx, idx, lit)));
    EXPECT_FALSE(non_scalar(make_scalar_fn(ctx, ScalarFnKind::Cos, lit)));

    // Tensor-valued kinds → `\,`.
    EXPECT_TRUE(non_scalar(make_trace(ctx, A)));
    EXPECT_TRUE(non_scalar(make_vector_invariant(ctx, A)));
    EXPECT_TRUE(non_scalar(make_transpose(ctx, A)));
    EXPECT_TRUE(non_scalar(make_dot(ctx, a, b)));
    EXPECT_TRUE(non_scalar(make_cross(ctx, a, b)));
    EXPECT_TRUE(non_scalar(make_deriv(ctx, x)));
    EXPECT_TRUE(non_scalar(make_nabla(ctx)));
}

// scalar_fn_str renders each unary scalar function; tan and log complete the
// set the other tests do not reach.
TEST(RenderScalarFn, TanAndLog)
{
    Context ctx;
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0, false);
    EXPECT_EQ(
        latex(*make_scalar_fn(ctx, ScalarFnKind::Tan, x)),
        "\\tan\\left(x\\right)");
    EXPECT_EQ(
        latex(*make_scalar_fn(ctx, ScalarFnKind::Log, x)),
        "\\log\\left(x\\right)");
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

TEST(RenderAnnotation, ExplicitSumBodyIsNegatedSubSum)
{
    // Σ_j -(Σ_i v^i): the inner sum is negated.  Without parens this renders as
    // "\sum_{j} -\sum_{i} …", which reads like the difference "Σ_j - Σ_i …"
    // (an empty outer body); the negated sub-sum must be parenthesised
    // (vibe 000064 #8b).
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    std::vector<SlotBinding> slots = {
        SlotBinding{IndexSlot{Level::Upper, Realm::Orthonormal, space_3d()}, i}};
    auto* v =
        make_tensor_object(ctx, make_tensor_name("v"), std::move(slots), 1);
    auto* inner = make_explicit_sum(ctx, i, v);
    auto* e = make_explicit_sum(ctx, j, make_negate(ctx, inner));

    IndexNameMap map;
    map.assign(i, make_index_name("i"));
    map.assign(j, make_index_name("j"));
    EXPECT_EQ(render_latex(*e, map), "\\sum_{j} (-\\sum_{i} \\mathbf{v}^{i})");
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
    // a × (b × c): the cross is non-associative, so the grouping is explicit.
    Context ctx;
    auto* bc = make_cross(ctx, T(ctx, "b", 1), T(ctx, "c", 1));
    auto* e = make_cross(ctx, T(ctx, "a", 1), bc);
    EXPECT_EQ(
        latex(*e), "\\mathbf{a} \\times (\\mathbf{b} \\times \\mathbf{c})");
}

TEST(RenderParens, CrossLeftAlsoNeedsParens)
{
    // (a × b) × c: the cross is non-associative, so the left grouping is shown
    // too — distinct from a × (b × c).
    Context ctx;
    auto* ab = make_cross(ctx, T(ctx, "a", 1), T(ctx, "b", 1));
    auto* e = make_cross(ctx, ab, T(ctx, "c", 1));
    EXPECT_EQ(
        latex(*e), "(\\mathbf{a} \\times \\mathbf{b}) \\times \\mathbf{c}");
}

TEST(RenderParens, DotOfCrossNeedsParens)
{
    // a · (b × c): the cross is wrapped (the scalar triple product).
    Context ctx;
    auto* bc = make_cross(ctx, T(ctx, "b", 1), T(ctx, "c", 1));
    auto* e = make_dot(ctx, T(ctx, "a", 1), bc);
    EXPECT_EQ(latex(*e), "\\mathbf{a} \\cdot (\\mathbf{b} \\times \\mathbf{c})");
}

TEST(RenderParens, ScalarContractionInProductNoParens)
{
    // (a · b) ⊗ c with a, b vectors: a · b is a scalar (rank 0), so it reads as
    // an atom in the product — no parens: a·b \, c.  Likewise (a·c)(b·d).
    Context ctx;
    auto* ab = make_dot(ctx, T(ctx, "a", 1), T(ctx, "b", 1));
    auto* e = make_tensor_product(ctx, ab, T(ctx, "c", 1));
    EXPECT_EQ(latex(*e), "\\mathbf{a} \\cdot \\mathbf{b} \\, \\mathbf{c}");
}

TEST(RenderParens, NonScalarContractionInProductNeedsParens)
{
    // (A · b) ⊗ c with A rank 2: A · b is rank 1 (a vector), not a scalar, so
    // it is wrapped — juxtaposition binds tighter than the contraction.
    Context ctx;
    auto* Ab = make_dot(ctx, T(ctx, "A", 2), T(ctx, "b", 1));
    auto* e = make_tensor_product(ctx, Ab, T(ctx, "c", 1));
    EXPECT_EQ(latex(*e), "(\\mathbf{A} \\cdot \\mathbf{b}) \\, \\mathbf{c}");
}

TEST(RenderParens, TensorProductRightNoParens)
{
    // a ⊗ (b ⊗ c): tensor product is associative — no parens.
    Context ctx;
    auto* bc = make_tensor_product(ctx, T(ctx, "b", 1), T(ctx, "c", 1));
    auto* e = make_tensor_product(ctx, T(ctx, "a", 1), bc);
    EXPECT_EQ(latex(*e), "\\mathbf{a} \\, \\mathbf{b} \\, \\mathbf{c}");
}

TEST(RenderUnary, TraceVecTranspose)
{
    Context ctx;
    auto* a = T(ctx, "a", 1);
    auto* b = T(ctx, "b", 1);
    auto* dyad = make_tensor_product(ctx, a, b);
    EXPECT_EQ(
        latex(*make_trace(ctx, dyad)),
        "\\operatorname{tr}(\\mathbf{a} \\, \\mathbf{b})");
    // vec of a bare tensor: A_× ; vec of a dyad: (a b)_×.
    EXPECT_EQ(
        latex(*make_vector_invariant(ctx, T(ctx, "A", 2))),
        "\\mathbf{A}_\\times");
    EXPECT_EQ(
        latex(*make_vector_invariant(ctx, dyad)),
        "(\\mathbf{a} \\, \\mathbf{b})_\\times");
    EXPECT_EQ(
        latex(*make_transpose(ctx, T(ctx, "A", 2))),
        "\\mathbf{A}^{\\mathsf{T}}");
    // A transpose of a composite operand wraps it.
    EXPECT_EQ(
        latex(*make_transpose(ctx, dyad)),
        "(\\mathbf{a} \\, \\mathbf{b})^{\\mathsf{T}}");
}

// ---- basis coordinate-letter naming (vibe 000067, increment 4) ------------

namespace
{
// A rank-0 coordinate "a" with one lower concrete-index slot, value v, tagged
// with basis b.
auto concrete_coord(Context& ctx, Basis const& b, int v) -> Expr const*
{
    return make_tensor_object(
        ctx,
        make_tensor_name("a"),
        {SlotBinding{
            IndexSlot{
                Level::Lower, Realm::Orthonormal, space_3d(), b.basis_id()},
            ConcreteIndex{v}}},
        0);
}
} // namespace

TEST(RenderBasisNames, ConcreteIndexUsesCoordinateLetterWithContext)
{
    Context ctx;
    auto b = cylindrical(ctx); // value names r, \theta, z
    IndexNameMap m1, m2, m3;
    EXPECT_EQ(render_latex(*concrete_coord(ctx, b, 1), m1, &ctx), "a_{r}");
    EXPECT_EQ(render_latex(*concrete_coord(ctx, b, 2), m2, &ctx), "a_{\\theta}");
    EXPECT_EQ(render_latex(*concrete_coord(ctx, b, 3), m3, &ctx), "a_{z}");
}

TEST(RenderBasisNames, NumericFallbackWithoutContext)
{
    // Without a Context the renderer cannot resolve basis_id, so a concrete
    // index stays numeric.
    Context ctx;
    auto b = cylindrical(ctx);
    IndexNameMap m;
    EXPECT_EQ(render_latex(*concrete_coord(ctx, b, 1), m), "a_{1}");
}

TEST(RenderBasisNames, BasisVectorReadsAsCoordinateLetter)
{
    // The cylindrical basis vector e with a concrete index renders e_{\theta}.
    Context ctx;
    auto b = cylindrical(ctx);
    auto const* e = make_tensor_object(
        ctx,
        make_tensor_name("e"),
        {SlotBinding{
            IndexSlot{
                Level::Lower, Realm::Orthonormal, space_3d(), b.basis_id()},
            ConcreteIndex{2}}},
        1);
    IndexNameMap m;
    EXPECT_EQ(render_latex(*e, m, &ctx), "\\mathbf{e}_{\\theta}");
}

TEST(RenderBasisNames, UntaggedConcreteIndexStaysNumeric)
{
    // A basis-unaware (basis_id 0) concrete index is numeric even with a ctx.
    Context ctx;
    (void)cylindrical(ctx);
    auto const* a = make_tensor_object(
        ctx,
        make_tensor_name("a"),
        {SlotBinding{
            IndexSlot{Level::Lower, Realm::Orthonormal, space_3d()},
            ConcreteIndex{1}}},
        0);
    IndexNameMap m;
    EXPECT_EQ(render_latex(*a, m, &ctx), "a_{1}");
}

// ---- basis vector-symbol override & display label (vibe 000067, 4b) -------

TEST(RenderBasisNames, WcsVectorSymbolsAndCoordinateLetters)
{
    // WCS frame vectors print as standalone i, j, k; coordinates as x, y, z.
    Context ctx;
    auto b = wcs(ctx);
    auto vec_k = make_tensor_object(
        ctx,
        make_tensor_name("e"),
        {SlotBinding{
            IndexSlot{
                Level::Lower, Realm::Orthonormal, space_3d(), b.basis_id()},
            ConcreteIndex{3}}},
        1);
    IndexNameMap m1;
    EXPECT_EQ(render_latex(*vec_k, m1, &ctx), "\\mathbf{k}");
    // A coordinate keeps the e_x-style letter naming (override is vector-only).
    auto coord_x = make_tensor_object(
        ctx,
        make_tensor_name("a"),
        {SlotBinding{
            IndexSlot{
                Level::Lower, Realm::Orthonormal, space_3d(), b.basis_id()},
            ConcreteIndex{1}}},
        0);
    IndexNameMap m2;
    EXPECT_EQ(render_latex(*coord_x, m2, &ctx), "a_{x}");
}

TEST(RenderBasisNames, DisplayLabelMarksIndices)
{
    // A labelled basis appends its frame marker to every index (primed-index
    // convention), so two frames are distinguishable in one term.
    Context ctx;
    auto b = make_orthonormal_basis(
        ctx,
        space_3d(),
        {make_tensor_object(ctx, make_tensor_name("p"), {}, 1),
         make_tensor_object(ctx, make_tensor_name("q"), {}, 1),
         make_tensor_object(ctx, make_tensor_name("s"), {}, 1)},
        make_tensor_name("e"),
        Handedness::Right,
        BasisNaming{.label = std::string{"'"}});
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto e_i = make_tensor_object(
        ctx,
        make_tensor_name("e"),
        {SlotBinding{
            IndexSlot{
                Level::Lower, Realm::Orthonormal, space_3d(), b.basis_id()},
            i}},
        1);
    IndexNameMap m;
    m.assign(i, make_index_name("i"));
    EXPECT_EQ(render_latex(*e_i, m, &ctx), "\\mathbf{e}_{i'}");
}

// vibe 000073: a field derivative ∂_q T renders with a prefix that binds looser
// than juxtaposition, so as a product factor it must be parenthesized — else
// "∂_r f \, r" reads as ∂_r(f r) rather than (∂_r f) r.
TEST(Render, FieldDerivativeWrapsAsProductFactor)
{
    Context ctx;
    auto* f = make_field(ctx, make_tensor_name("f"), 0, {});
    auto* df = make_field_derivative(
        ctx, f, make_tensor_name("r"), CoordinateRef{1, 0, false});
    auto* r = make_coordinate(ctx, make_tensor_name("r"), 1, 0, false);
    EXPECT_EQ(
        latex(*make_tensor_product(ctx, df, r)), "(\\partial_{r} f) \\, r");
    EXPECT_EQ(latex(*df), "\\partial_{r} f"); // standalone: no parentheses
}

// ---- ∂ operator (vibe 000077, step A) ----------------------------------

TEST(Render, DerivOperator)
{
    Context ctx;
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0, false);
    auto* dx = make_deriv(ctx, x);
    EXPECT_EQ(latex(*dx), "\\partial_{x}");
    // In a product the operator renders in position, left of its (rightward)
    // operand: ∂_x acting on a field f.
    auto* f = make_field(ctx, make_tensor_name("f"), 0, {});
    EXPECT_EQ(latex(*make_tensor_product(ctx, dx, f)), "\\partial_{x} \\, f");
}

// ---- chart-free ∇ operator (vibe 000078, increment 1) ------------------

TEST(Render, NablaOperator)
{
    Context ctx;
    auto* del = make_nabla(ctx);
    EXPECT_EQ(latex(*del), "\\nabla");
    // grad/div/rot as ∇ combined with ⊗ / · / ×.
    auto* eps = make_field(ctx, make_tensor_name("e"), 2, {}, /*sym=*/true);
    EXPECT_EQ(latex(*make_dot(ctx, del, eps)), "\\nabla \\cdot \\mathbf{e}");
    EXPECT_EQ(latex(*make_cross(ctx, del, eps)), "\\nabla \\times \\mathbf{e}");
    // ∇·(∇⊗X) is the Laplacian ΔX (vibe 000080 Increment 3), not the
    // misleading "power" ∇²; a plain ∇·(∇×X) is not folded.
    auto* lap = make_dot(ctx, del, make_tensor_product(ctx, del, eps));
    EXPECT_EQ(latex(*lap), "\\Delta \\mathbf{e}");
    auto* f = make_field(ctx, make_tensor_name("f"), 0, {});
    EXPECT_EQ(
        latex(*make_dot(ctx, del, make_tensor_product(ctx, del, f))),
        "\\Delta f");
    // A Δ operand that is a sum wraps in parens: Δ(a + b).
    auto* a = T(ctx, "a", 1);
    auto* b = T(ctx, "b", 1);
    EXPECT_EQ(
        latex(*make_dot(
            ctx, del, make_tensor_product(ctx, del, make_sum(ctx, a, b)))),
        "\\Delta (\\mathbf{a} + \\mathbf{b})");
    // Not a Laplacian: ∇·(∇×X) keeps the divergence form.
    EXPECT_EQ(
        latex(*make_dot(ctx, del, make_cross(ctx, del, eps))),
        "\\nabla \\cdot (\\nabla \\times \\mathbf{e})");
    // Laplacian, floated canonical form (vibe 000083): canonicalize rewrites
    // ∇·(∇⊗X) into the equivalent (∇·∇)⊗X = tprod(dot(∇,∇), X); it is still ΔX,
    // so it renders the same — a Δ must survive a canonicalize (e.g. inside
    // apply_identity), not degrade back to ∇·∇.
    auto* floated = make_tensor_product(ctx, make_dot(ctx, del, del), eps);
    EXPECT_EQ(latex(*floated), "\\Delta \\mathbf{e}");
    EXPECT_EQ(
        latex(*make_tensor_product(ctx, make_dot(ctx, del, del), f)),
        "\\Delta f");
    // But a bare ∇·∇ with no operand is NOT a Laplacian — it stays ∇·∇.
    EXPECT_EQ(latex(*make_dot(ctx, del, del)), "\\nabla \\cdot \\nabla");
    // A scalar coefficient rides through: canonicalize left-associates
    // c·∇·(∇⊗X) into (c ∇·∇)⊗X = tprod(tprod(c, ∇·∇), X), burying ∇·∇ under the
    // scalar.  It must still render as c Δ X (vibe 000083 — the μ∇·∇u vs μΔu
    // navier_lame case), not degrade to c ∇·∇ X.
    auto* mu = T(ctx, "\\mu", 0);
    auto* coeff_lap = make_tensor_product(
        ctx, make_tensor_product(ctx, mu, make_dot(ctx, del, del)), eps);
    EXPECT_EQ(latex(*coeff_lap), "\\mu \\, \\Delta \\mathbf{e}");

    // Operator-left normalisation (vibe 000080 Increment 6): a canonical
    // reorder can leave ∇ on the *right*; render it left where it acts. X·∇
    // (divergence commuted) → ∇·X.
    EXPECT_EQ(latex(*make_dot(ctx, eps, del)), "\\nabla \\cdot \\mathbf{e}");
    // v⊗∇ (transpose of a gradient) → (∇v)ᵀ, since v⊗∇ = (∇⊗v)ᵀ.
    auto* v = make_field(ctx, make_tensor_name("v"), 1, {});
    EXPECT_EQ(
        latex(*make_tensor_product(ctx, v, del)),
        "(\\nabla \\mathbf{v})^{\\mathsf{T}}");
    // f⊗∇ for a rank-0 f is the gradient ∇f (its transpose is itself), no ᵀ.
    EXPECT_EQ(latex(*make_tensor_product(ctx, f, del)), "\\nabla f");
    // A rank-0 divergence-valued operand keeps no spurious ᵀ: (∇·v)⊗∇ → ∇(∇·v)
    // (the divergence operand of the gradient is parenthesised).
    EXPECT_EQ(
        latex(*make_tensor_product(ctx, make_dot(ctx, del, v), del)),
        "\\nabla (\\nabla \\cdot \\mathbf{v})");
}

// ---- render_nf_latex (the Nf normal-form renderer) ---------------------

// Lower an Expr to its normal form and render that.  render_nf_latex walks the
// nf IR (nf.hpp) — a separate renderer from render_latex — so it needs its own
// coverage across the factor kinds it can meet.
static auto nf_latex(Context& ctx, Expr const* e) -> std::string
{
    IndexNameMap map;
    return render_nf_latex(*nf::canonicalize_nf(ctx, e), map, &ctx);
}

TEST(RenderNf, ScalarAndTensorFactors)
{
    Context ctx;
    // A plain scalar, a bare tensor, and a product with a numeric coefficient.
    EXPECT_EQ(nf_latex(ctx, make_scalar(ctx, Rational{3})), "3");
    auto* a = T(ctx, "a", 1);
    auto* b = T(ctx, "b", 1);
    // a·b — a contraction of two vectors renders through the atom/factor path.
    auto s = nf_latex(ctx, make_dot(ctx, a, b));
    EXPECT_NE(s.find("\\cdot"), std::string::npos);
    // 2 (a⊗b) — a scalar coefficient in front of a dyad.
    auto d = nf_latex(
        ctx,
        make_tensor_product(
            ctx, make_scalar(ctx, Rational{2}), make_tensor_product(ctx, a, b)));
    EXPECT_NE(d.find("2"), std::string::npos);
}

TEST(RenderNf, InvariantsAndTranspose)
{
    Context ctx;
    auto* A = make_field(ctx, make_tensor_name("A"), 2, {});
    // tr(A) — the Trace unary of the nf renderer.
    EXPECT_NE(
        nf_latex(ctx, make_trace(ctx, A)).find("\\operatorname{tr}"),
        std::string::npos);
    // Aᵀ — the Transpose unary.
    EXPECT_NE(
        nf_latex(ctx, make_transpose(ctx, A)).find("\\mathsf{T}"),
        std::string::npos);
    // vec(A) — the VectorInvariant unary renders with the _\times subscript.
    EXPECT_NE(
        nf_latex(ctx, make_vector_invariant(ctx, A)).find("_\\times"),
        std::string::npos);
}

TEST(RenderNf, ScalarFnPowFractionAndOperators)
{
    Context ctx;
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0, false);
    // sin(x) — a ScalarFn factor.
    EXPECT_NE(
        nf_latex(ctx, make_scalar_fn(ctx, ScalarFnKind::Sin, x)).find("\\sin"),
        std::string::npos);
    // x² — a Pow factor.
    EXPECT_NE(
        nf_latex(ctx, make_pow(ctx, x, make_scalar(ctx, Rational{2})))
            .find("^{2}"),
        std::string::npos);
    // ∂_x — a bare (unapplied) ∂ operator renders through the nf Deriv arm.
    EXPECT_NE(
        nf_latex(ctx, make_deriv(ctx, x)).find("\\partial_{x}"),
        std::string::npos);
    // ∇ — the bare Nabla operator.
    EXPECT_NE(nf_latex(ctx, make_nabla(ctx)).find("\\nabla"), std::string::npos);
}

TEST(RenderNf, MultiFactorDoubleDotAndWrappedPow)
{
    Context ctx;
    auto* A = make_field(ctx, make_tensor_name("A"), 2, {});
    auto* B = make_field(ctx, make_tensor_name("B"), 2, {});
    auto* a = T(ctx, "a", 1);
    auto* b = T(ctx, "b", 1);
    auto* c = T(ctx, "c", 1);
    auto* d = T(ctx, "d", 1);
    auto* x = make_coordinate(ctx, make_tensor_name("x"), 7, 0, false);

    // A:B — the DDot contraction operator (cop_str) in the nf renderer.
    EXPECT_NE(
        nf_latex(ctx, make_ddot(ctx, A, B)).find(" : "), std::string::npos);
    // A··B — the DDotAlt contraction operator.
    EXPECT_NE(
        nf_latex(ctx, make_ddot_alt(ctx, A, B)).find("\\cdot\\!\\cdot"),
        std::string::npos);
    // (a·b)(c·d) — a term with two contraction factors exercises the per-factor
    // precedence/sub() path (prec(nf::Contraction)).
    auto multi = nf_latex(
        ctx,
        make_tensor_product(ctx, make_dot(ctx, a, b), make_dot(ctx, c, d)));
    EXPECT_NE(multi.find(" \\, "), std::string::npos);
    // (x + 1)² — a Pow whose base is a sum wraps the base in parens.
    auto* pow = make_pow(
        ctx,
        make_sum(ctx, x, make_scalar(ctx, Rational{1})),
        make_scalar(ctx, Rational{2}));
    EXPECT_NE(nf_latex(ctx, pow).find("(x + 1)^{2}"), std::string::npos);
}
