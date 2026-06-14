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

// ---- expand_eps ------------------------------------------------------------

TEST(ExpandEps, Rank3ProducesSixTerms)
{
    // ε_ijk in 3D → Sum tree with 6 leaves (delta products)
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};

    auto const* eps = make_levi_civita(
        ctx,
        Realm::Oblique,
        sp,
        {Level::Lower, Level::Lower, Level::Lower},
        {IndexAssoc{i}, IndexAssoc{j}, IndexAssoc{k}});

    auto const* after = steps::expand_eps(ctx, eps);
    EXPECT_NE(after, eps); // something changed
    // The result is a Sum tree — not a TensorObject any more.
    EXPECT_EQ(std::get_if<TensorObject>(&after->node), nullptr);
}

TEST(ExpandEps, ConcreteTraceIsCorrect)
{
    // ε_ijk evaluated at i=1,j=2,k=3 should give +1 (even permutation).
    // Derivation: expand_eps → eval_delta_concrete → fold_arithmetic.
    Context ctx;
    auto const* sp = space_3d();

    auto const* eps = make_levi_civita(
        ctx,
        Realm::Oblique,
        sp,
        {Level::Lower, Level::Lower, Level::Lower},
        {IndexAssoc{ConcreteIndex{1}},
         IndexAssoc{ConcreteIndex{2}},
         IndexAssoc{ConcreteIndex{3}}});

    auto const* e1 = steps::expand_eps(ctx, eps);
    auto const* e2 = steps::eval_delta_concrete(ctx, e1);
    auto const* e3 = steps::fold_arithmetic(ctx, e2);
    EXPECT_EQ(scalar_value(e3), Rational{1});
}

TEST(ExpandEps, OddPermutationIsMinusOne)
{
    // ε_ijk at i=1,j=3,k=2 → -1 (odd permutation).
    Context ctx;
    auto const* sp = space_3d();

    auto const* eps = make_levi_civita(
        ctx,
        Realm::Oblique,
        sp,
        {Level::Lower, Level::Lower, Level::Lower},
        {IndexAssoc{ConcreteIndex{1}},
         IndexAssoc{ConcreteIndex{3}},
         IndexAssoc{ConcreteIndex{2}}});

    auto const* e1 = steps::expand_eps(ctx, eps);
    auto const* e2 = steps::eval_delta_concrete(ctx, e1);
    auto const* e3 = steps::fold_arithmetic(ctx, e2);
    EXPECT_EQ(scalar_value(e3), Rational{-1});
}

TEST(ExpandEps, RepeatedIndexIsZero)
{
    // ε_ijk at i=1,j=1,k=2 → 0 (repeated index).
    Context ctx;
    auto const* sp = space_3d();

    auto const* eps = make_levi_civita(
        ctx,
        Realm::Oblique,
        sp,
        {Level::Lower, Level::Lower, Level::Lower},
        {IndexAssoc{ConcreteIndex{1}},
         IndexAssoc{ConcreteIndex{1}},
         IndexAssoc{ConcreteIndex{2}}});

    auto const* e1 = steps::expand_eps(ctx, eps);
    auto const* e2 = steps::eval_delta_concrete(ctx, e1);
    auto const* e3 = steps::fold_arithmetic(ctx, e2);
    EXPECT_EQ(scalar_value(e3), Rational{0});
}

// ---- fold_sums -------------------------------------------------------------

TEST(FoldSums, ThreeTermCycle)
{
    // δ^1_k δ^1_l + δ^2_k δ^2_l + δ^3_k δ^3_l  →  ExplicitSum{m, δ^m_k δ^m_l}
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};

    auto delta = [&](int v, CountableIndex idx)
    {
        return make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{idx});
    };

    // Build Sum(Sum(d1*d1, d2*d2), d3*d3)
    auto* t1 = make_tensor_product(ctx, delta(1, k), delta(1, l));
    auto* t2 = make_tensor_product(ctx, delta(2, k), delta(2, l));
    auto* t3 = make_tensor_product(ctx, delta(3, k), delta(3, l));
    auto* sum = make_sum(ctx, make_sum(ctx, t1, t2), t3);

    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_NE(after, sum);
    EXPECT_NE(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(FoldSums, MismatchedStructureUnchanged)
{
    // δ^1_k + δ^2_l + δ^3_k  — the second addend uses a different symbolic
    // index, so no cycle exists; the sum must be left unchanged.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};

    auto delta_k = [&](int v)
    {
        return make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{k});
    };
    auto delta_l = [&](int v)
    {
        return make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{l});
    };

    auto* sum =
        make_sum(ctx, make_sum(ctx, delta_k(1), delta_l(2)), delta_k(3));
    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_EQ(after, sum); // unchanged
}

// ---- contract_delta --------------------------------------------------------

TEST(ContractDelta, SumOfSquaredDeltas)
{
    // Σ_m δ^m_k δ^m_l  →  δ_{kl}  (after fold_sums + contract_delta)
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};

    // Build the concrete sum and fold it first.
    auto delta = [&](int v, CountableIndex idx)
    {
        return make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{idx});
    };
    auto* t1 = make_tensor_product(ctx, delta(1, k), delta(1, l));
    auto* t2 = make_tensor_product(ctx, delta(2, k), delta(2, l));
    auto* t3 = make_tensor_product(ctx, delta(3, k), delta(3, l));
    auto* sum = make_sum(ctx, make_sum(ctx, t1, t2), t3);

    auto const* folded = steps::fold_sums(ctx, sum);
    ASSERT_NE(std::get_if<ExplicitSum>(&folded->node), nullptr);

    auto const* contracted = steps::contract_delta(ctx, folded);
    // Result must be a single TensorObject (Kronecker delta).
    auto const* res = std::get_if<TensorObject>(&contracted->node);
    ASSERT_NE(res, nullptr);
    ASSERT_TRUE(res->traits.has_value());
    EXPECT_EQ(res->traits->well_known, WellKnownKind::Delta);
    // Both surviving slots should use the symbolic indices k and l.
    ASSERT_EQ(res->slots.size(), 2u);
    auto const* c0 = std::get_if<CountableIndex>(&*res->slots[0].index);
    auto const* c1 = std::get_if<CountableIndex>(&*res->slots[1].index);
    ASSERT_NE(c0, nullptr);
    ASSERT_NE(c1, nullptr);
    // The fresh summation index is gone; both remaining ids are k.id or l.id.
    EXPECT_TRUE(
        (c0->id == k.id && c1->id == l.id)
        || (c0->id == l.id && c1->id == k.id));
}

TEST(ContractDelta, NoContractWithMismatchedLevels)
{
    // Σ_m δ^m_k δ_m^l  — the summation slot has different levels in each
    // delta → contraction rule does not fire.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};

    auto* d1 = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{m},
        IndexAssoc{k}); // Upper m
    auto* d2 = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Lower,
        Level::Upper,
        IndexAssoc{m},
        IndexAssoc{l}); // Lower m  (different level)
    auto* prod = make_tensor_product(ctx, d1, d2);
    auto* expr = make_explicit_sum(ctx, m, prod);

    auto const* after = steps::contract_delta(ctx, expr);
    EXPECT_EQ(after, expr); // unchanged — levels don't match
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
