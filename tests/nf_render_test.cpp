#include <tender/render.hpp>

#include <tender/index_space.hpp> // space_3d
#include <tender/nf_lower.hpp>    // canonicalize_nf

#include <gtest/gtest.h>

using namespace tender;

namespace
{

auto vec(Context& ctx, std::string_view name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 1);
}

auto mat(Context& ctx, std::string_view name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 2);
}

auto ivec(Context& ctx, std::string_view name, Level lvl, CountableIndex idx)
    -> Expr const*
{
    return make_tensor_object(
        ctx,
        make_tensor_name(name),
        {SlotBinding{
            IndexSlot{lvl, Realm::Oblique, space_3d()}, IndexAssoc{idx}}},
        1);
}

// Lower `e` to its Nf and render it.
auto rnf(Context& ctx, Expr const* e) -> std::string
{
    IndexNameMap map;
    return render_nf_latex(*nf::canonicalize_nf(ctx, e), map);
}

} // namespace

// ---- additive layer ----------------------------------------------------

TEST(RenderNf, SingleAtom)
{
    Context ctx;
    EXPECT_EQ(rnf(ctx, vec(ctx, "a")), "\\mathbf{a}");
}

TEST(RenderNf, NegatedTerm)
{
    Context ctx;
    EXPECT_EQ(rnf(ctx, make_negate(ctx, vec(ctx, "a"))), "-\\mathbf{a}");
}

TEST(RenderNf, Sum)
{
    Context ctx;
    EXPECT_EQ(
        rnf(ctx, make_sum(ctx, vec(ctx, "a"), vec(ctx, "b"))),
        "\\mathbf{a} + \\mathbf{b}");
}

TEST(RenderNf, Difference)
{
    Context ctx;
    EXPECT_EQ(
        rnf(ctx, make_difference(ctx, vec(ctx, "a"), vec(ctx, "b"))),
        "\\mathbf{a} - \\mathbf{b}");
}

TEST(RenderNf, ZeroIsLiteralZero)
{
    Context ctx;
    auto const* a = vec(ctx, "a");
    EXPECT_EQ(rnf(ctx, make_difference(ctx, a, a)), "0");
}

TEST(RenderNf, LikeTermsMerge)
{
    Context ctx;
    auto const* a = vec(ctx, "a");
    auto sa = [&](int n)
    { return make_tensor_product(ctx, make_scalar(ctx, Rational{n}), a); };
    EXPECT_EQ(rnf(ctx, make_sum(ctx, sa(2), sa(3))), "5 \\, \\mathbf{a}");
}

// ---- coefficient + region juxtaposition --------------------------------

TEST(RenderNf, IntegerCoefficient)
{
    Context ctx;
    auto const* e =
        make_tensor_product(ctx, make_scalar(ctx, Rational{2}), vec(ctx, "a"));
    EXPECT_EQ(rnf(ctx, e), "2 \\, \\mathbf{a}");
}

TEST(RenderNf, ScalarDotIsAtomic)
{
    Context ctx;
    EXPECT_EQ(
        rnf(ctx, make_dot(ctx, vec(ctx, "a"), vec(ctx, "b"))),
        "\\mathbf{a} \\cdot \\mathbf{b}");
}

TEST(RenderNf, WedgedScalarFloatsAndReadsAsAtom)
{
    // (a·b) ⊗ C → "a · b \, C": the scalar dot floats out and is not wrapped.
    Context ctx;
    auto const* e = make_tensor_product(
        ctx, make_dot(ctx, vec(ctx, "a"), vec(ctx, "b")), mat(ctx, "C"));
    EXPECT_EQ(rnf(ctx, e), "\\mathbf{a} \\cdot \\mathbf{b} \\, \\mathbf{C}");
}

TEST(RenderNf, LoneCrossIsUnwrapped)
{
    Context ctx;
    EXPECT_EQ(
        rnf(ctx, make_cross(ctx, vec(ctx, "a"), vec(ctx, "b"))),
        "\\mathbf{a} \\times \\mathbf{b}");
}

TEST(RenderNf, JuxtaposedCrossIsWrapped)
{
    // (a×b) ⊗ C → "(a × b) \, C": the cross wraps once it has a sibling.
    Context ctx;
    auto const* e = make_tensor_product(
        ctx, make_cross(ctx, vec(ctx, "a"), vec(ctx, "b")), mat(ctx, "C"));
    EXPECT_EQ(rnf(ctx, e), "(\\mathbf{a} \\times \\mathbf{b}) \\, \\mathbf{C}");
}

// ---- composites --------------------------------------------------------

TEST(RenderNf, ParenSum)
{
    // A·(b+c): the genuine sum renders parenthesized inside the contraction.
    Context ctx;
    auto const* e = make_dot(
        ctx, mat(ctx, "A"), make_sum(ctx, vec(ctx, "b"), vec(ctx, "c")));
    EXPECT_EQ(rnf(ctx, e), "\\mathbf{A} \\cdot (\\mathbf{b} + \\mathbf{c})");
}

TEST(RenderNf, Trace)
{
    Context ctx;
    EXPECT_EQ(
        rnf(ctx, make_trace(ctx, mat(ctx, "A"))),
        "\\operatorname{tr}(\\mathbf{A})");
}

// ---- summation ---------------------------------------------------------

TEST(RenderNf, ImplicitContractionHasNoSigma)
{
    // a^i b_i: a Default (realm-implicit) dummy renders as repeated indices —
    // no \sum prefix.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* e = make_tensor_product(
        ctx, ivec(ctx, "a", Level::Upper, i), ivec(ctx, "b", Level::Lower, i));
    EXPECT_EQ(rnf(ctx, e), "\\mathbf{a}^{i} \\, \\mathbf{b}_{i}");
}

TEST(RenderNf, ExplicitSumGetsSigma)
{
    // Σ_i a_i: an explicit non-default sum keeps its \sum prefix.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* e = make_explicit_sum(ctx, i, ivec(ctx, "a", Level::Lower, i));
    EXPECT_EQ(rnf(ctx, e), "\\sum_{i} \\mathbf{a}_{i}");
}

TEST(RenderNf, NoSumGetsCancelledSigma)
{
    // ⌀Σ_i (a^i b_i): a NoSum override renders a cancelled \sum.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* e = make_no_sum(
        ctx,
        i,
        make_tensor_product(
            ctx,
            ivec(ctx, "a", Level::Upper, i),
            ivec(ctx, "b", Level::Lower, i)));
    EXPECT_EQ(
        rnf(ctx, e), "\\cancel{\\sum}_{i} \\mathbf{a}^{i} \\, \\mathbf{b}_{i}");
}

TEST(RenderNf, SymbolicDivision)
{
    // A/(a·b) → \frac{A}{a · b}: division renders as a fraction.
    Context ctx;
    auto const* e = make_scalar_div(
        ctx, mat(ctx, "A"), make_dot(ctx, vec(ctx, "a"), vec(ctx, "b")));
    EXPECT_EQ(
        rnf(ctx, e), "\\frac{\\mathbf{A}}{\\mathbf{a} \\cdot \\mathbf{b}}");
}
