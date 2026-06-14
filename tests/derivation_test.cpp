#include <gtest/gtest.h>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/index_space.hpp>

using namespace tender;

// ---- helpers ---------------------------------------------------------------

static auto scalar_value(Expr const* e) -> std::optional<Rational>
{
    if (auto const* s = std::get_if<ScalarLiteral>(&e->node))
        return s->value;
    return std::nullopt;
}

// ---- unroll_sums -----------------------------------------------------------

TEST(UnrollSums, DeltaTrace3D)
{
    // sum_i δ^i_i in 3D → scalar 3
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};

    auto const* d =
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, i, i);
    auto const* expr = make_explicit_sum(ctx, i, d);

    auto* after = steps::unroll_sums(ctx, expr);
    // Should produce Sum(Sum(δ^1_1, δ^2_2), δ^3_3); no more ExplicitSum.
    EXPECT_EQ(std::get_if<ExplicitSum>(&after->node), nullptr);
    EXPECT_NE(std::get_if<Sum>(&after->node), nullptr);
}

TEST(UnrollSums, SymbolicBoundUnchanged)
{
    // ExplicitSum with a symbolic bound must not be unrolled.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* n = make_scalar(ctx, Rational{3});
    auto const* body = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* expr = make_explicit_sum(ctx, i, body, n);

    auto const* after = steps::unroll_sums(ctx, expr);
    EXPECT_EQ(after, expr); // pointer unchanged
}

// ---- eval_delta_concrete ---------------------------------------------------

TEST(EvalDeltaConcrete, DiagonalIsOne)
{
    Context ctx;
    auto const* sp = space_3d();
    auto const* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        ConcreteIndex{1},
        ConcreteIndex{2});
    auto const* after = steps::eval_delta_concrete(ctx, d);
    auto v = scalar_value(after);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, Rational{0});
}

TEST(EvalDeltaConcrete, OffDiagonalIsZero)
{
    Context ctx;
    auto const* sp = space_3d();
    auto const* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        ConcreteIndex{2},
        ConcreteIndex{2});
    auto const* after = steps::eval_delta_concrete(ctx, d);
    auto v = scalar_value(after);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, Rational{1});
}

TEST(EvalDeltaConcrete, AbstractIndexUnchanged)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    auto const* d =
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, i, i);
    auto const* after = steps::eval_delta_concrete(ctx, d);
    EXPECT_EQ(after, d); // pointer unchanged — nothing to evaluate
}

// ---- fold_arithmetic -------------------------------------------------------

TEST(FoldArithmetic, SumOfScalars)
{
    Context ctx;
    auto const* a = make_scalar(ctx, Rational{2});
    auto const* b = make_scalar(ctx, Rational{3});
    auto const* expr = make_sum(ctx, a, b);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(scalar_value(after), Rational{5});
}

TEST(FoldArithmetic, ProductOfScalars)
{
    Context ctx;
    auto const* a = make_scalar(ctx, Rational{3});
    auto const* b = make_scalar(ctx, Rational{4});
    auto const* expr = make_tensor_product(ctx, a, b);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(scalar_value(after), Rational{12});
}

TEST(FoldArithmetic, NestedSumFoldsCompletely)
{
    // (1 + 1) + 1 → 3
    Context ctx;
    auto* one = make_scalar(ctx, Rational{1});
    auto* expr = make_sum(ctx, make_sum(ctx, one, one), one);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(scalar_value(after), Rational{3});
}

TEST(FoldArithmetic, NonScalarUnchanged)
{
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* b = make_scalar(ctx, Rational{1});
    auto const* expr = make_sum(ctx, a, b);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(after, expr); // can't fold — left is not scalar
}

// ---- Full derivation: δ^i_i = 3 -----------------------------------------

TEST(Derivation, DeltaTraceIs3)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};

    auto const* expr = make_explicit_sum(
        ctx,
        i,
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, i, i));

    Derivation drv(ctx, expr);
    drv.step(steps::unroll_sums)
        .step(steps::eval_delta_concrete)
        .step(steps::fold_arithmetic);

    ASSERT_EQ(drv.history().size(), 4u); // initial + 3 steps
    EXPECT_EQ(scalar_value(drv.current()), Rational{3});
}

// ---- Full derivation: δ^i_j δ^i_j = 3 ------------------------------------

TEST(Derivation, DeltaSquaredIs3)
{
    // sum_i sum_j δ^i_j * δ^i_j = 3  in 3D
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};

    auto const* d1 =
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, i, j);
    auto const* d2 =
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, i, j);
    auto const* expr = make_explicit_sum(
        ctx, i, make_explicit_sum(ctx, j, make_tensor_product(ctx, d1, d2)));

    Derivation drv(ctx, expr);
    drv.step(steps::unroll_sums)
        .step(steps::eval_delta_concrete)
        .step(steps::fold_arithmetic);

    EXPECT_EQ(scalar_value(drv.current()), Rational{3});
}

// ---- Derivation history ---------------------------------------------------

TEST(Derivation, HistoryLengthMatchesSteps)
{
    Context ctx;
    auto const* e = make_scalar(ctx, Rational{1});
    Derivation drv(ctx, e);
    EXPECT_EQ(drv.history().size(), 1u);
    drv.step(steps::fold_arithmetic);
    EXPECT_EQ(drv.history().size(), 2u);
    drv.step(steps::fold_arithmetic);
    EXPECT_EQ(drv.history().size(), 3u);
    EXPECT_EQ(drv.initial(), e);
    EXPECT_EQ(drv.current(), drv.history().back());
}
