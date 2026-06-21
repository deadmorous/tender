#include <gtest/gtest.h>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/index_space.hpp>
#include <tender/render.hpp>

#include <string>
#include <utility>
#include <vector>

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

// ---- contract_eps_pair -----------------------------------------------------

namespace
{
auto render_with_names(
    Expr const* e,
    std::vector<std::pair<CountableIndex, char const*>> names) -> std::string
{
    IndexNameMap map;
    for (auto const& [ci, nm]: names)
        map.assign(ci, make_index_name(nm));
    return render_latex(*e, map);
}
} // namespace

TEST(ContractIdentity, IdentityDotVectorIsVector)
{
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* I = make_identity(ctx);
    EXPECT_TRUE(
        structural_eq(steps::contract_identity(ctx, make_dot(ctx, I, a)), a));
    EXPECT_TRUE(
        structural_eq(steps::contract_identity(ctx, make_dot(ctx, a, I)), a));
}

TEST(ContractIdentity, IdentityDotIdentityIsIdentity)
{
    Context ctx;
    auto const* I = make_identity(ctx);
    EXPECT_TRUE(
        structural_eq(steps::contract_identity(ctx, make_dot(ctx, I, I)), I));
}

TEST(ContractIdentity, NonIdentityDotUnchanged)
{
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    auto const* dot = make_dot(ctx, a, b);
    EXPECT_EQ(steps::contract_identity(ctx, dot), dot);
}

TEST(ContractIdentity, WalksNestedDot)
{
    // (I · a) + b  →  a + b
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    auto const* I = make_identity(ctx);
    auto const* expr = make_sum(ctx, make_dot(ctx, I, a), b);
    EXPECT_TRUE(
        structural_eq(steps::contract_identity(ctx, expr), make_sum(ctx, a, b)));
}

TEST(ContractEpsPair, OneSharedIndex)
{
    // Σ_i ε^{ijk} ε_{iml}  →  δ^j_m δ^k_l − δ^j_l δ^k_m
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};

    auto const* ea = make_levi_civita(
        ctx,
        Realm::Oblique,
        sp,
        {Level::Upper, Level::Upper, Level::Upper},
        {IndexAssoc{i}, IndexAssoc{j}, IndexAssoc{k}});
    auto const* eb = make_levi_civita(
        ctx,
        Realm::Oblique,
        sp,
        {Level::Lower, Level::Lower, Level::Lower},
        {IndexAssoc{i}, IndexAssoc{m}, IndexAssoc{l}});
    auto const* expr =
        make_explicit_sum(ctx, i, make_tensor_product(ctx, ea, eb));

    auto const* after = steps::contract_eps_pair(ctx, expr);
    EXPECT_NE(after, expr);
    EXPECT_EQ(
        render_with_names(after, {{j, "j"}, {k, "k"}, {m, "m"}, {l, "l"}}),
        "\\delta^{j}_{m} \\, \\delta^{k}_{l} - \\delta^{j}_{l} \\, "
        "\\delta^{k}_{m}");
}

TEST(ContractEpsPair, TwoSharedIndices)
{
    // Σ_{ij} ε^{ijk} ε_{ijl}  →  2 δ^k_l
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};

    auto const* ea = make_levi_civita(
        ctx,
        Realm::Oblique,
        sp,
        {Level::Upper, Level::Upper, Level::Upper},
        {IndexAssoc{i}, IndexAssoc{j}, IndexAssoc{k}});
    auto const* eb = make_levi_civita(
        ctx,
        Realm::Oblique,
        sp,
        {Level::Lower, Level::Lower, Level::Lower},
        {IndexAssoc{i}, IndexAssoc{j}, IndexAssoc{l}});
    auto const* expr = make_explicit_sum(
        ctx, j, make_explicit_sum(ctx, i, make_tensor_product(ctx, ea, eb)));

    auto const* after = steps::contract_eps_pair(ctx, expr);
    EXPECT_NE(after, expr);
    EXPECT_EQ(
        render_with_names(after, {{k, "k"}, {l, "l"}}),
        "2 \\, \\delta^{k}_{l}");
}

TEST(ContractEpsPair, SwappedFreeIndicesFlipSign)
{
    // Σ_i ε^{ijk} ε_{ilm}  →  δ^j_l δ^k_m − δ^j_m δ^k_l
    // (free lowers are l,m here instead of m,l: the determinant columns swap.)
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};

    auto const* ea = make_levi_civita(
        ctx,
        Realm::Oblique,
        sp,
        {Level::Upper, Level::Upper, Level::Upper},
        {IndexAssoc{i}, IndexAssoc{j}, IndexAssoc{k}});
    auto const* eb = make_levi_civita(
        ctx,
        Realm::Oblique,
        sp,
        {Level::Lower, Level::Lower, Level::Lower},
        {IndexAssoc{i}, IndexAssoc{l}, IndexAssoc{m}});
    auto const* expr =
        make_explicit_sum(ctx, i, make_tensor_product(ctx, ea, eb));

    auto const* after = steps::contract_eps_pair(ctx, expr);
    EXPECT_EQ(
        render_with_names(after, {{j, "j"}, {k, "k"}, {l, "l"}, {m, "m"}}),
        "\\delta^{j}_{l} \\, \\delta^{k}_{m} - \\delta^{j}_{m} \\, "
        "\\delta^{k}_{l}");
}

TEST(ContractEpsPair, NonEpsProductUnchanged)
{
    // Σ_i δ^i_k δ^i_l is not a pair of ε's → left unchanged.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};

    auto* d1 =
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, i, k);
    auto* d2 =
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, i, l);
    auto* expr = make_explicit_sum(ctx, i, make_tensor_product(ctx, d1, d2));

    auto const* after = steps::contract_eps_pair(ctx, expr);
    EXPECT_EQ(after, expr); // unchanged
}

TEST(ContractEpsPair, WalksEveryNodeKindWithoutMatch)
{
    // No ε-pair anywhere: the top-down walker must recurse through every
    // composite node kind and return the expression unchanged.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()}, j{ctx.alloc_index_id()};
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"), {}, 2);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    auto const* d =
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, i, j);

    auto const* e = make_negate(
        ctx,
        make_sum(
            ctx,
            make_difference(ctx, make_ddot(ctx, A, B), make_ddot_alt(ctx, A, B)),
            make_sum(
                ctx,
                make_dot(ctx, a, b),
                make_sum(
                    ctx,
                    make_cross(ctx, a, b),
                    make_sum(
                        ctx,
                        make_scalar_div(ctx, A, B),
                        make_sum(
                            ctx,
                            make_tensor_product(ctx, d, d),
                            make_sum(
                                ctx,
                                make_explicit_sum(ctx, i, d),
                                make_no_sum(ctx, i, d))))))));

    auto const* after = steps::contract_eps_pair(ctx, e);
    EXPECT_EQ(after, e); // no ε-pair → pointer preserved
}

// ---- fold_arithmetic: Difference, ScalarDiv, Negate -----------------------

TEST(FoldArithmetic, DifferenceOfScalars)
{
    // 5 - 2 → 3
    Context ctx;
    auto* a = make_scalar(ctx, Rational{5});
    auto* b = make_scalar(ctx, Rational{2});
    auto* expr = make_difference(ctx, a, b);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(scalar_value(after), Rational{3});
}

TEST(FoldArithmetic, DifferenceWithNonScalar)
{
    // A - 1 → unchanged (left is not a scalar)
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* b = make_scalar(ctx, Rational{1});
    auto const* expr = make_difference(ctx, a, b);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(after, expr);
}

TEST(FoldArithmetic, ScalarDivision)
{
    // 6 / 3 → 2
    Context ctx;
    auto* a = make_scalar(ctx, Rational{6});
    auto* b = make_scalar(ctx, Rational{3});
    auto* expr = make_scalar_div(ctx, a, b);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(scalar_value(after), Rational{2});
}

TEST(FoldArithmetic, NegateOfScalar)
{
    // -3 → scalar -3
    Context ctx;
    auto* v = make_scalar(ctx, Rational{3});
    auto* expr = make_negate(ctx, v);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(scalar_value(after), Rational{-3});
}

TEST(FoldArithmetic, NegateNonScalarUnchanged)
{
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* expr = make_negate(ctx, a);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(after, expr);
}

TEST(FoldArithmetic, SumNegateRightToDifference)
{
    // A + (-B) → A - B
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* b = make_tensor_object(ctx, make_tensor_name("B"));
    auto const* expr = make_sum(ctx, a, make_negate(ctx, b));
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* d = std::get_if<Difference>(&after->node);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->left, a);
    EXPECT_EQ(d->right, b);
}

TEST(FoldArithmetic, SumNegateLeftToDifference)
{
    // (-A) + B → B - A
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* b = make_tensor_object(ctx, make_tensor_name("B"));
    auto const* expr = make_sum(ctx, make_negate(ctx, a), b);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* d = std::get_if<Difference>(&after->node);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->left, b);
    EXPECT_EQ(d->right, a);
}

TEST(FoldArithmetic, NestedSumWithNegateLeftChild)
{
    // Sum(A, Sum(Negate(B), C)) → Sum(A, Difference(C, B)) — inner converts
    // bottom-up, preventing "A + -B + C" rendering.
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* b = make_tensor_object(ctx, make_tensor_name("B"));
    auto const* c = make_tensor_object(ctx, make_tensor_name("C"));
    auto const* inner = make_sum(ctx, make_negate(ctx, b), c);
    auto const* expr = make_sum(ctx, a, inner);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    // Outer Sum(A, Difference(C, B))
    auto const* s = std::get_if<Sum>(&after->node);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->left, a);
    auto const* d = std::get_if<Difference>(&s->right->node);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->left, c);
    EXPECT_EQ(d->right, b);
}

// ---- rewrite_tree traversal: Dot, DDot, DDotAlt, Cross, NoSum ------------
// These exercises ensure every structural node type is traversed when a step
// recurses into a tree.  The inner scalars get folded; the wrapper node is
// rebuilt if its child changed.

TEST(FoldArithmetic, DotChildFolds)
{
    // dot(1+1, 1+1) — fold_arithmetic folds the leaves, Dot is rebuilt.
    Context ctx;
    auto* one = make_scalar(ctx, Rational{1});
    auto* sum = make_sum(ctx, one, one);
    auto* expr = make_dot(ctx, sum, sum);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* d = std::get_if<Dot>(&after->node);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(scalar_value(d->left), Rational{2});
    EXPECT_EQ(scalar_value(d->right), Rational{2});
}

TEST(FoldArithmetic, DDotChildFolds)
{
    Context ctx;
    auto* one = make_scalar(ctx, Rational{1});
    auto* sum = make_sum(ctx, one, one);
    auto* expr = make_ddot(ctx, sum, sum);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* d = std::get_if<DDot>(&after->node);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(scalar_value(d->left), Rational{2});
}

TEST(FoldArithmetic, DDotAltChildFolds)
{
    Context ctx;
    auto* one = make_scalar(ctx, Rational{1});
    auto* sum = make_sum(ctx, one, one);
    auto* expr = make_ddot_alt(ctx, sum, sum);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* d = std::get_if<DDotAlt>(&after->node);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(scalar_value(d->left), Rational{2});
}

TEST(FoldArithmetic, CrossChildFolds)
{
    Context ctx;
    auto* one = make_scalar(ctx, Rational{1});
    auto* sum = make_sum(ctx, one, one);
    auto* expr = make_cross(ctx, sum, sum);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* d = std::get_if<Cross>(&after->node);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(scalar_value(d->left), Rational{2});
}

TEST(FoldArithmetic, ScalarDivChildFolds)
{
    // (1+1) / (1+2) → 2/3
    Context ctx;
    auto* one = make_scalar(ctx, Rational{1});
    auto* two = make_scalar(ctx, Rational{2});
    auto* num = make_sum(ctx, one, one);
    auto* den = make_sum(ctx, one, two);
    auto* expr = make_scalar_div(ctx, num, den);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(scalar_value(after), Rational(2, 3));
}

TEST(FoldArithmetic, DifferenceChildFolds)
{
    // (1+4) - (1+1) → 3
    Context ctx;
    auto* one = make_scalar(ctx, Rational{1});
    auto* four = make_scalar(ctx, Rational{4});
    auto* lhs = make_sum(ctx, one, four);
    auto* rhs = make_sum(ctx, one, one);
    auto* expr = make_difference(ctx, lhs, rhs);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(scalar_value(after), Rational{3});
}

TEST(FoldArithmetic, NoSumBodyFolds)
{
    // NoSum(i, 1+1) → NoSum(i, 2)
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto* one = make_scalar(ctx, Rational{1});
    auto* expr = make_no_sum(ctx, i, make_sum(ctx, one, one));
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* ns = std::get_if<NoSum>(&after->node);
    ASSERT_NE(ns, nullptr);
    EXPECT_EQ(scalar_value(ns->body), Rational{2});
}

// ---- unroll_sums: edge cases -----------------------------------------------

TEST(UnrollSums, IndexNotInBodyUnchanged)
{
    // ExplicitSum{i, A} where A doesn't contain i → space not found →
    // unchanged.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* body = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* expr = make_explicit_sum(ctx, i, body);

    auto const* after = steps::unroll_sums(ctx, expr);
    EXPECT_EQ(after, expr);
}

// ---- expand_eps: edge cases ------------------------------------------------

TEST(ExpandEps, TwoDimSpaceUnchanged)
{
    // ε over a 2D space produces rank-2 (2 slots) — expand_eps only handles
    // rank-3, so this must be left unchanged.
    Context ctx;
    auto const* sp = space_2d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};

    auto const* eps = make_levi_civita(
        ctx,
        Realm::Oblique,
        sp,
        {Level::Lower, Level::Lower},
        {IndexAssoc{i}, IndexAssoc{j}});

    auto const* after = steps::expand_eps(ctx, eps);
    EXPECT_EQ(after, eps);
}

// ---- fold_sums: edge cases -------------------------------------------------

TEST(FoldSums, NoConcreteSlotInFirstAddend)
{
    // If the first addend has no ConcreteIndex at all, no space can be found
    // and the expression must be left unchanged.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};

    // Three fully-symbolic deltas — no concrete slot anywhere.
    auto mk = [&](CountableIndex a, CountableIndex b)
    {
        return make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{a},
            IndexAssoc{b});
    };
    auto* sum = make_sum(ctx, make_sum(ctx, mk(i, j), mk(i, j)), mk(i, j));
    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_EQ(after, sum);
}

TEST(FoldSums, AddendCountMismatchUnchanged)
{
    // Only 2 addends for a 3-value space → size mismatch → unchanged.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};

    auto delta = [&](int v)
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

    auto* sum = make_sum(ctx, delta(1), delta(2)); // only 2 of 3
    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_EQ(after, sum);
}

// ---- contract_delta: more edge cases ---------------------------------------

TEST(ContractDelta, BoundedExplicitSumUnchanged)
{
    // ExplicitSum with a symbolic bound → unchanged.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};

    auto* d1 = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{m},
        IndexAssoc{k});
    auto* d2 = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{m},
        IndexAssoc{k});
    auto* bound = make_scalar(ctx, Rational{3});
    auto* expr =
        make_explicit_sum(ctx, m, make_tensor_product(ctx, d1, d2), bound);

    auto const* after = steps::contract_delta(ctx, expr);
    EXPECT_EQ(after, expr);
}

TEST(ContractDelta, NonProductBodyUnchanged)
{
    // ExplicitSum{m, δ^m_k} — body is a single delta, not a product →
    // unchanged.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};

    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{m},
        IndexAssoc{k});
    auto* expr = make_explicit_sum(ctx, m, d);

    auto const* after = steps::contract_delta(ctx, expr);
    EXPECT_EQ(after, expr);
}

TEST(ContractDelta, NonDeltaInProductUnchanged)
{
    // ExplicitSum{m, A · δ^m_k} — left side is not a delta → unchanged.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};

    auto* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{m},
        IndexAssoc{k});
    auto* expr = make_explicit_sum(ctx, m, make_tensor_product(ctx, a, d));

    auto const* after = steps::contract_delta(ctx, expr);
    EXPECT_EQ(after, expr);
}

TEST(ContractDelta, SumIndexNotInEitherDeltaUnchanged)
{
    // ExplicitSum{m, δ^k_l · δ^k_l} — neither delta uses m → unchanged.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};

    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{k},
        IndexAssoc{l});
    auto* expr = make_explicit_sum(ctx, m, make_tensor_product(ctx, d, d));

    auto const* after = steps::contract_delta(ctx, expr);
    EXPECT_EQ(after, expr);
}

TEST(ContractDelta, MismatchedRealmUnchanged)
{
    // Σ_m δ^m_k(Oblique) · δ^m_l(Covariant) — different realms → unchanged.
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
        IndexAssoc{k});
    auto* d2 = make_delta(
        ctx,
        Realm::Orthonormal,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{m},
        IndexAssoc{l});
    auto* expr = make_explicit_sum(ctx, m, make_tensor_product(ctx, d1, d2));

    auto const* after = steps::contract_delta(ctx, expr);
    EXPECT_EQ(after, expr);
}

// ---- fold_sums: Negate and Difference addend types -------------------------
// These tests exercise the Negate / Difference walker branches inside
// find_space_from_concrete, collect_concrete_values, and structural_eq.

TEST(FoldSums, NegatedTermCycle)
{
    // -δ^1_k + -δ^2_k + -δ^3_k  →  ExplicitSum{m, -δ^m_k}
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};

    auto neg_delta = [&](int v)
    {
        auto* d = make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{k});
        return make_negate(ctx, d);
    };

    auto* sum =
        make_sum(ctx, make_sum(ctx, neg_delta(1), neg_delta(2)), neg_delta(3));
    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_NE(after, sum);
    EXPECT_NE(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(FoldSums, DifferenceTermCycle)
{
    // (δ^1_k - δ^1_l) + (δ^2_k - δ^2_l) + (δ^3_k - δ^3_l)
    //   →  ExplicitSum{m, δ^m_k - δ^m_l}
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};

    auto diff_delta = [&](int v)
    {
        auto* dk = make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{k});
        auto* dl = make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{l});
        return make_difference(ctx, dk, dl);
    };

    auto* sum = make_sum(
        ctx, make_sum(ctx, diff_delta(1), diff_delta(2)), diff_delta(3));
    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_NE(after, sum);
    EXPECT_NE(std::get_if<ExplicitSum>(&after->node), nullptr);
}

// ---- unroll_sums: Difference body exercises find_index_space ---------------

TEST(UnrollSums, DifferenceBody)
{
    // ExplicitSum{i, δ^i_j - δ^i_k}  →  unrolled Sum of differences.
    // Exercises the Difference walker branch inside find_index_space.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};

    auto* dj = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    auto* dk = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{k});
    auto* expr = make_explicit_sum(ctx, i, make_difference(ctx, dj, dk));

    auto const* after = steps::unroll_sums(ctx, expr);
    // Result must be a Sum tree (no more ExplicitSum for i).
    EXPECT_EQ(std::get_if<ExplicitSum>(&after->node), nullptr);
    EXPECT_NE(std::get_if<Sum>(&after->node), nullptr);
}

// ---- fold_arithmetic: non-scalar TensorProduct guard -----------------------

TEST(FoldArithmetic, TensorProductWithOneSimplifies)
{
    // TensorProduct(A, 1) → A (multiplicative identity).
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* one = make_scalar(ctx, Rational{1});
    auto const* expr = make_tensor_product(ctx, a, one);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(after, a);
}

TEST(FoldArithmetic, TensorProductWithZeroCollapses)
{
    // TensorProduct(A, 0) → 0.
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* zero = make_scalar(ctx, Rational{0});
    auto const* expr = make_tensor_product(ctx, a, zero);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* sl = std::get_if<ScalarLiteral>(&after->node);
    ASSERT_NE(sl, nullptr);
    EXPECT_EQ(sl->value, Rational{0});
}

TEST(FoldArithmetic, SumWithZeroSimplifies)
{
    // Sum(0, A) → A (additive identity).
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* zero = make_scalar(ctx, Rational{0});
    auto const* expr = make_sum(ctx, zero, a);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(after, a);
}

TEST(FoldArithmetic, NonScalarTensorProductUnchanged)
{
    // TensorProduct(A, 2) — neither 0 nor 1 on the right, left is non-scalar.
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* two = make_scalar(ctx, Rational{2});
    auto const* expr = make_tensor_product(ctx, a, two);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(after, expr);
}

TEST(FoldArithmetic, NegateOfNegateSimplifies)
{
    // -(-A) → A
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* neg_a = make_negate(ctx, a);
    auto const* neg_neg_a = make_negate(ctx, neg_a);
    auto const* after = steps::fold_arithmetic(ctx, neg_neg_a);
    EXPECT_EQ(after, a);
}

TEST(FoldArithmetic, NegTimesNegSimplifies)
{
    // (-A)(-B) → A*B
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* b = make_tensor_object(ctx, make_tensor_name("B"));
    auto const* expr =
        make_tensor_product(ctx, make_negate(ctx, a), make_negate(ctx, b));
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* tp = std::get_if<TensorProduct>(&after->node);
    ASSERT_NE(tp, nullptr);
    EXPECT_EQ(tp->left, a);
    EXPECT_EQ(tp->right, b);
}

TEST(FoldSums, PartialCycleInLargerSum)
{
    // 6-addend sum containing two copies of a 3-cycle:
    // δ^1^k δ^1_j, δ^2^k δ^2_j, δ^3^k δ^3_j repeated twice.
    // fold_sums should fold one 3-cycle into ExplicitSum, leaving 3 addends.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};

    auto mk = [&](int v) -> Expr const*
    {
        return make_tensor_product(
            ctx,
            make_delta(
                ctx,
                Realm::Oblique,
                sp,
                Level::Upper,
                Level::Upper,
                IndexAssoc{CountableIndex{k}},
                IndexAssoc{ConcreteIndex{v}}),
            make_delta(
                ctx,
                Realm::Oblique,
                sp,
                Level::Upper,
                Level::Lower,
                IndexAssoc{CountableIndex{k}},
                IndexAssoc{CountableIndex{j}}));
    };

    // Build 6-addend sum: mk(1)+mk(2)+mk(3)+mk(1)+mk(2)+mk(3)
    auto const* sum = make_sum(
        ctx,
        make_sum(
            ctx,
            make_sum(
                ctx, make_sum(ctx, make_sum(ctx, mk(1), mk(2)), mk(3)), mk(1)),
            mk(2)),
        mk(3));

    auto const* after = steps::fold_sums(ctx, sum);
    // One 3-cycle folded: the result should be Sum(ExplicitSum, 3 remaining).
    EXPECT_NE(after, sum);
    // The sum should not be entirely gone — it still has remaining addends.
    auto const* s = std::get_if<Sum>(&after->node);
    EXPECT_NE(s, nullptr);
}

// ---- unroll_sums: more body types ------------------------------------------

TEST(UnrollSums, NegateBody)
{
    // ExplicitSum{i, -δ^i_j}  →  sum of Negate terms.
    // Exercises the Negate walker branch inside find_index_space.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};

    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    auto* expr = make_explicit_sum(ctx, i, make_negate(ctx, d));

    auto const* after = steps::unroll_sums(ctx, expr);
    EXPECT_EQ(std::get_if<ExplicitSum>(&after->node), nullptr);
    EXPECT_NE(std::get_if<Sum>(&after->node), nullptr);
}

TEST(UnrollSums, BodyContainsAbstractTensor)
{
    // ExplicitSum{i, δ^i_j * A} where A has no slots.
    // substitute() visits A and finds no matching slot → !changed → return e
    // (exercises the !changed early-return in substitute).
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};

    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    auto* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto* expr = make_explicit_sum(ctx, i, make_tensor_product(ctx, d, a));

    auto const* after = steps::unroll_sums(ctx, expr);
    EXPECT_EQ(std::get_if<ExplicitSum>(&after->node), nullptr);
}

// ---- fold_sums: ExplicitSum and NoSum wrapped addends ----------------------

TEST(FoldSums, ExplicitSumWrappedCycle)
{
    // ExplicitSum{j,δ^1_j} + ExplicitSum{j,δ^2_j} + ExplicitSum{j,δ^3_j}
    //   →  ExplicitSum{m, ExplicitSum{j, δ^m_j}}
    // Exercises ExplicitSum in find_space_from_concrete,
    // collect_concrete_values, rewrite_tree (ExplicitSum), and structural_eq.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex j{ctx.alloc_index_id()};

    auto mk = [&](int v)
    {
        auto* d = make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{j});
        return make_explicit_sum(ctx, j, d);
    };

    auto* sum = make_sum(ctx, make_sum(ctx, mk(1), mk(2)), mk(3));
    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_NE(after, sum);
    EXPECT_NE(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(FoldSums, NoSumWrappedCycle)
{
    // NoSum(k,δ^1_k) + NoSum(k,δ^2_k) + NoSum(k,δ^3_k)
    //   →  ExplicitSum{m, NoSum(k, δ^m_k)}
    // Exercises NoSum in find_space_from_concrete, collect_concrete_values,
    // rewrite_tree (NoSum), and structural_eq.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};

    auto mk = [&](int v)
    {
        auto* d = make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{k});
        return make_no_sum(ctx, k, d);
    };

    auto* sum = make_sum(ctx, make_sum(ctx, mk(1), mk(2)), mk(3));
    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_NE(after, sum);
    EXPECT_NE(std::get_if<ExplicitSum>(&after->node), nullptr);
}

// ---- eval_delta_concrete: non-delta tensor leaves unchanged ----------------

TEST(EvalDeltaConcrete, LeviCivitaUnchanged)
{
    // A LeviCivita is not a delta — eval_delta_concrete must leave it alone.
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

    auto const* after = steps::eval_delta_concrete(ctx, eps);
    EXPECT_EQ(after, eps);
}

// ---- fold_arithmetic: ScalarDiv guard (non-scalar operands) ----------------

TEST(FoldArithmetic, ScalarDivNonScalarUnchanged)
{
    // A / 1 → unchanged (left is not a scalar literal).
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* one = make_scalar(ctx, Rational{1});
    auto const* expr = make_scalar_div(ctx, a, one);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(after, expr);
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

// ---- expand_products -------------------------------------------------------

TEST(ExpandProducts, LeftSumDistributes)
{
    // (A + B) * C  →  A*C + B*C
    Context ctx;
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto* C = make_tensor_object(ctx, make_tensor_name("C"));
    auto* expr = make_tensor_product(ctx, make_sum(ctx, A, B), C);
    auto const* after = steps::expand_products(ctx, expr);
    auto const* s = std::get_if<Sum>(&after->node);
    ASSERT_NE(s, nullptr);
    EXPECT_NE(std::get_if<TensorProduct>(&s->left->node), nullptr);
    EXPECT_NE(std::get_if<TensorProduct>(&s->right->node), nullptr);
}

TEST(ExpandProducts, LeftDifferenceDistributes)
{
    // (A - B) * C  →  A*C - B*C
    Context ctx;
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto* C = make_tensor_object(ctx, make_tensor_name("C"));
    auto* expr = make_tensor_product(ctx, make_difference(ctx, A, B), C);
    auto const* after = steps::expand_products(ctx, expr);
    EXPECT_NE(std::get_if<Difference>(&after->node), nullptr);
}

TEST(ExpandProducts, RightSumDistributes)
{
    // A * (B + C)  →  A*B + A*C
    Context ctx;
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto* C = make_tensor_object(ctx, make_tensor_name("C"));
    auto* expr = make_tensor_product(ctx, A, make_sum(ctx, B, C));
    auto const* after = steps::expand_products(ctx, expr);
    EXPECT_NE(std::get_if<Sum>(&after->node), nullptr);
}

TEST(ExpandProducts, RightDifferenceDistributes)
{
    // A * (B - C)  →  A*B - A*C
    Context ctx;
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto* C = make_tensor_object(ctx, make_tensor_name("C"));
    auto* expr = make_tensor_product(ctx, A, make_difference(ctx, B, C));
    auto const* after = steps::expand_products(ctx, expr);
    EXPECT_NE(std::get_if<Difference>(&after->node), nullptr);
}

TEST(ExpandProducts, PlainProductUnchanged)
{
    // A * B  —  no Sum/Difference to distribute over; result is still a
    // TensorProduct with the same children.
    Context ctx;
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto* expr = make_tensor_product(ctx, A, B);
    auto const* after = steps::expand_products(ctx, expr);
    auto const* tp = std::get_if<TensorProduct>(&after->node);
    ASSERT_NE(tp, nullptr);
    EXPECT_EQ(tp->left, A);
    EXPECT_EQ(tp->right, B);
}

TEST(ExpandProducts, BothSumsFullyDistributed)
{
    // (A + B) * (C + D)  →  four TensorProduct leaves under two Sum nodes.
    Context ctx;
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto* C = make_tensor_object(ctx, make_tensor_name("C"));
    auto* D = make_tensor_object(ctx, make_tensor_name("D"));
    auto* expr =
        make_tensor_product(ctx, make_sum(ctx, A, B), make_sum(ctx, C, D));
    auto const* after = steps::expand_products(ctx, expr);
    EXPECT_NE(after, expr);
    // Result must NOT be a TensorProduct at the top level.
    EXPECT_EQ(std::get_if<TensorProduct>(&after->node), nullptr);
}

// ---- fold_arithmetic: missing identity branches ----------------------------

TEST(FoldArithmetic, SumRightZeroSimplifies)
{
    // Sum(A, 0) → A
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* zero = make_scalar(ctx, Rational{0});
    auto const* expr = make_sum(ctx, a, zero);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(after, a);
}

TEST(FoldArithmetic, DifferenceRightZeroSimplifies)
{
    // Difference(A, 0) → A
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* zero = make_scalar(ctx, Rational{0});
    auto const* expr = make_difference(ctx, a, zero);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(after, a);
}

TEST(FoldArithmetic, ProductLeftIsZero)
{
    // TensorProduct(0, A) → 0
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* zero = make_scalar(ctx, Rational{0});
    auto const* expr = make_tensor_product(ctx, zero, a);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(scalar_value(after), Rational{0});
}

TEST(FoldArithmetic, ProductLeftIsOne)
{
    // TensorProduct(1, A) → A
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* one = make_scalar(ctx, Rational{1});
    auto const* expr = make_tensor_product(ctx, one, a);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    EXPECT_EQ(after, a);
}

// ---- unroll_sums: unusual body node types (covers find_index_space) --------

TEST(UnrollSums, ScalarLiteralBodyUnchanged)
{
    // ExplicitSum{i, 1}  —  index i absent from body → space not found →
    // unchanged. Exercises the ScalarLiteral arm of find_index_space.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* expr = make_explicit_sum(ctx, i, make_scalar(ctx, Rational{1}));
    auto const* after = steps::unroll_sums(ctx, expr);
    EXPECT_EQ(after, expr);
}

TEST(UnrollSums, DotBody)
{
    // ExplicitSum{i, Dot(δ^i_j, A)}  —  exercises Dot arm of find_index_space.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* expr = make_explicit_sum(ctx, i, make_dot(ctx, d, A));
    auto const* after = steps::unroll_sums(ctx, expr);
    EXPECT_EQ(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(UnrollSums, DDotBody)
{
    // ExplicitSum{i, DDot(δ^i_j, A)}  —  exercises DDot arm of
    // find_index_space.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* expr = make_explicit_sum(ctx, i, make_ddot(ctx, d, A));
    auto const* after = steps::unroll_sums(ctx, expr);
    EXPECT_EQ(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(UnrollSums, DDotAltBody)
{
    // ExplicitSum{i, DDotAlt(δ^i_j, A)}  —  exercises DDotAlt arm.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* expr = make_explicit_sum(ctx, i, make_ddot_alt(ctx, d, A));
    auto const* after = steps::unroll_sums(ctx, expr);
    EXPECT_EQ(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(UnrollSums, CrossBody)
{
    // ExplicitSum{i, Cross(δ^i_j, A)}  —  exercises Cross arm.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* expr = make_explicit_sum(ctx, i, make_cross(ctx, d, A));
    auto const* after = steps::unroll_sums(ctx, expr);
    EXPECT_EQ(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(UnrollSums, ScalarDivBody)
{
    // ExplicitSum{i, δ^i_j / A}  —  exercises ScalarDiv arm of
    // find_index_space.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* expr = make_explicit_sum(ctx, i, make_scalar_div(ctx, d, A));
    auto const* after = steps::unroll_sums(ctx, expr);
    EXPECT_EQ(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(UnrollSums, NoSumBody)
{
    // ExplicitSum{i, NoSum{k, δ^i_k}}  —  exercises NoSum arm of
    // find_index_space.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{k});
    auto* expr = make_explicit_sum(ctx, i, make_no_sum(ctx, k, d));
    auto const* after = steps::unroll_sums(ctx, expr);
    EXPECT_EQ(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(UnrollSums, NestedBoundedExplicitSumBody)
{
    // ExplicitSum{i, ExplicitSum{j, δ^i_j, N}}
    // The inner ExplicitSum has a bound; find_index_space must search both
    // body and bound to locate index i.  Exercises the 'if (s.bound)' branch.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    auto* bound = make_scalar(ctx, Rational{3});
    auto* inner = make_explicit_sum(ctx, j, d, bound);
    auto* expr = make_explicit_sum(ctx, i, inner);
    auto const* after = steps::unroll_sums(ctx, expr);
    EXPECT_EQ(std::get_if<ExplicitSum>(&after->node), nullptr);
}

// ---- fold_sums: unusual addend types (covers find_space_from_concrete,
//      collect_concrete_values, and structural_eq Dot/DDot/DDotAlt/Cross) ----

TEST(FoldSums, DotWrappedCycle)
{
    // Dot(δ^1_k, A) + Dot(δ^2_k, A) + Dot(δ^3_k, A)
    //   →  ExplicitSum{m, Dot(δ^m_k, A)}
    // Exercises Dot branches in find_space_from_concrete,
    // collect_concrete_values, and structural_eq.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));

    auto mk = [&](int v)
    {
        auto* d = make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{k});
        return make_dot(ctx, d, A);
    };

    auto* sum = make_sum(ctx, make_sum(ctx, mk(1), mk(2)), mk(3));
    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_NE(after, sum);
    EXPECT_NE(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(FoldSums, DDotWrappedCycle)
{
    // DDot(δ^1_k, A) + DDot(δ^2_k, A) + DDot(δ^3_k, A)
    //   →  ExplicitSum{m, DDot(δ^m_k, A)}
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));

    auto mk = [&](int v)
    {
        auto* d = make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{k});
        return make_ddot(ctx, d, A);
    };

    auto* sum = make_sum(ctx, make_sum(ctx, mk(1), mk(2)), mk(3));
    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_NE(after, sum);
    EXPECT_NE(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(FoldSums, DDotAltWrappedCycle)
{
    // DDotAlt(δ^1_k, A) + DDotAlt(δ^2_k, A) + DDotAlt(δ^3_k, A)
    //   →  ExplicitSum{m, DDotAlt(δ^m_k, A)}
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));

    auto mk = [&](int v)
    {
        auto* d = make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{k});
        return make_ddot_alt(ctx, d, A);
    };

    auto* sum = make_sum(ctx, make_sum(ctx, mk(1), mk(2)), mk(3));
    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_NE(after, sum);
    EXPECT_NE(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(FoldSums, CrossWrappedCycle)
{
    // Cross(δ^1_k, A) + Cross(δ^2_k, A) + Cross(δ^3_k, A)
    //   →  ExplicitSum{m, Cross(δ^m_k, A)}
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));

    auto mk = [&](int v)
    {
        auto* d = make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{k});
        return make_cross(ctx, d, A);
    };

    auto* sum = make_sum(ctx, make_sum(ctx, mk(1), mk(2)), mk(3));
    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_NE(after, sum);
    EXPECT_NE(std::get_if<ExplicitSum>(&after->node), nullptr);
}

TEST(FoldSums, ScalarDivWrappedCycle)
{
    // (δ^1_k / A) + (δ^2_k / A) + (δ^3_k / A)
    //   →  ExplicitSum{m, δ^m_k / A}
    // Exercises ScalarDiv branches in find_space_from_concrete,
    // collect_concrete_values, and structural_eq.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex k{ctx.alloc_index_id()};
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));

    auto mk = [&](int v)
    {
        auto* d = make_delta(
            ctx,
            Realm::Oblique,
            sp,
            Level::Upper,
            Level::Lower,
            IndexAssoc{ConcreteIndex{v}},
            IndexAssoc{k});
        return make_scalar_div(ctx, d, A);
    };

    auto* sum = make_sum(ctx, make_sum(ctx, mk(1), mk(2)), mk(3));
    auto const* after = steps::fold_sums(ctx, sum);
    EXPECT_NE(after, sum);
    EXPECT_NE(std::get_if<ExplicitSum>(&after->node), nullptr);
}

// ---- expand_eps: non-LeviCivita early exits --------------------------------

TEST(ExpandEps, PlainTensorUnchanged)
{
    // A tensor with no traits is not a LeviCivita; expand_eps must leave it
    // unchanged.  Exercises the '!t->traits' guard.
    Context ctx;
    auto const* t = make_tensor_object(ctx, make_tensor_name("T"));
    auto const* after = steps::expand_eps(ctx, t);
    EXPECT_EQ(after, t);
}

TEST(ExpandEps, DeltaUnchanged)
{
    // A Kronecker delta has traits but well_known != LeviCivita; must be
    // unchanged.  Exercises the 'well_known != LeviCivita' guard.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto const* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    auto const* after = steps::expand_eps(ctx, d);
    EXPECT_EQ(after, d);
}

// ---- fold_arithmetic: Sum/Difference normalisation with Negate -------------

TEST(FoldArithmetic, SumWithNegateRightBecomeDiff)
{
    // Sum(A, -B)  →  Diff(A, B)
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto const* expr = make_sum(ctx, A, make_negate(ctx, B));
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* d = std::get_if<Difference>(&after->node);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->left, A);
    EXPECT_EQ(d->right, B);
}

TEST(FoldArithmetic, DiffWithNegateRightBecomeSum)
{
    // Diff(A, -B)  →  Sum(A, B)
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto const* expr = make_difference(ctx, A, make_negate(ctx, B));
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* s = std::get_if<Sum>(&after->node);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->left, A);
    EXPECT_EQ(s->right, B);
}

TEST(FoldArithmetic, ProductRightNegateBecomesNegateProduct)
{
    // A * (-B)  →  -(A * B)
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto const* expr = make_tensor_product(ctx, A, make_negate(ctx, B));
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* neg = std::get_if<Negate>(&after->node);
    ASSERT_NE(neg, nullptr);
    auto const* tp = std::get_if<TensorProduct>(&neg->operand->node);
    ASSERT_NE(tp, nullptr);
    EXPECT_EQ(tp->left, A);
    EXPECT_EQ(tp->right, B);
}

TEST(FoldArithmetic, ProductLeftNegateBecomesNegateProduct)
{
    // (-A) * B  →  -(A * B)
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto const* expr = make_tensor_product(ctx, make_negate(ctx, A), B);
    auto const* after = steps::fold_arithmetic(ctx, expr);
    auto const* neg = std::get_if<Negate>(&after->node);
    ASSERT_NE(neg, nullptr);
    auto const* tp = std::get_if<TensorProduct>(&neg->operand->node);
    ASSERT_NE(tp, nullptr);
    EXPECT_EQ(tp->left, A);
    EXPECT_EQ(tp->right, B);
}

// ---- fold_equal_addends ---------------------------------------------------

TEST(FoldEqualAddends, DuplicateBecomesScalarMultiple)
{
    // A + A  →  2 * A
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* expr = make_sum(ctx, A, A);
    auto const* after = steps::fold_equal_addends(ctx, expr);
    auto const* tp = std::get_if<TensorProduct>(&after->node);
    ASSERT_NE(tp, nullptr);
    auto const* sl = std::get_if<ScalarLiteral>(&tp->left->node);
    ASSERT_NE(sl, nullptr);
    EXPECT_EQ(sl->value, Rational{2});
    EXPECT_EQ(tp->right, A);
}

TEST(FoldEqualAddends, ExistingCoefficientAccumulates)
{
    // 2*A + A  →  3 * A
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* two = make_scalar(ctx, Rational{2});
    auto const* expr = make_sum(ctx, make_tensor_product(ctx, two, A), A);
    auto const* after = steps::fold_equal_addends(ctx, expr);
    auto const* tp = std::get_if<TensorProduct>(&after->node);
    ASSERT_NE(tp, nullptr);
    auto const* sl = std::get_if<ScalarLiteral>(&tp->left->node);
    ASSERT_NE(sl, nullptr);
    EXPECT_EQ(sl->value, Rational{3});
    EXPECT_EQ(tp->right, A);
}

TEST(FoldEqualAddends, DifferentAddendsUnchanged)
{
    // A + B  —  different objects, no merging possible.
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto const* expr = make_sum(ctx, A, B);
    auto const* after = steps::fold_equal_addends(ctx, expr);
    EXPECT_EQ(after, expr);
}

TEST(FoldEqualAddends, NegatedCoefficientMerges)
{
    // (-A) + A  →  0
    // Negate acts as coefficient -1; -1 + 1 = 0 → dropped.
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* neg_a = make_negate(ctx, A);
    auto const* expr = make_sum(ctx, neg_a, A);
    auto const* after = steps::fold_equal_addends(ctx, expr);
    auto const* sl = std::get_if<ScalarLiteral>(&after->node);
    ASSERT_NE(sl, nullptr);
    EXPECT_EQ(sl->value, Rational{0});
}

TEST(FoldEqualAddends, DifferenceOfEqualIsZero)
{
    // A - A  →  0  (collection sees through Difference)
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* expr = make_difference(ctx, A, A);
    auto const* after = steps::fold_equal_addends(ctx, expr);
    auto const* sl = std::get_if<ScalarLiteral>(&after->node);
    ASSERT_NE(sl, nullptr);
    EXPECT_EQ(sl->value, Rational{0});
}

TEST(FoldEqualAddends, DifferenceAccumulates)
{
    // 2A - A  →  A
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* two = make_scalar(ctx, Rational{2});
    auto const* expr =
        make_difference(ctx, make_tensor_product(ctx, two, A), A);
    auto const* after = steps::fold_equal_addends(ctx, expr);
    EXPECT_EQ(after, A);
}

TEST(FoldEqualAddends, SubtractionInSumMerges)
{
    // A + B - A  →  B
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto const* expr = make_difference(ctx, make_sum(ctx, A, B), A);
    auto const* after = steps::fold_equal_addends(ctx, expr);
    EXPECT_EQ(after, B);
}

TEST(FoldEqualAddends, RightScalarCoefficient)
{
    // A*2 + A  →  3A  (scalar on the right of the product)
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* two = make_scalar(ctx, Rational{2});
    auto const* expr = make_sum(ctx, make_tensor_product(ctx, A, two), A);
    auto const* after = steps::fold_equal_addends(ctx, expr);
    auto const* tp = std::get_if<TensorProduct>(&after->node);
    ASSERT_NE(tp, nullptr);
    auto const* sl = std::get_if<ScalarLiteral>(&tp->left->node);
    ASSERT_NE(sl, nullptr);
    EXPECT_EQ(sl->value, Rational{3});
    EXPECT_EQ(tp->right, A);
}

TEST(FoldEqualAddends, RationalCoefficientsCollect)
{
    // ½A + ½A  →  A  (coefficients accumulate as exact rationals)
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* half = make_scalar(ctx, Rational{1, 2});
    auto const* expr = make_sum(
        ctx,
        make_tensor_product(ctx, half, A),
        make_tensor_product(ctx, half, A));
    auto const* after = steps::fold_equal_addends(ctx, expr);
    EXPECT_EQ(after, A);
}

// ---- canonicalize (algebraic normal form, vibe 000037) --------------------

namespace
{
// Render the canonical form of an expression, naming the given indices.
auto canon_str(
    Context& ctx,
    Expr const* e,
    std::vector<std::pair<CountableIndex, char const*>> names = {})
    -> std::string
{
    return render_with_names(steps::canonicalize(ctx, e), std::move(names));
}

auto delta_ul(Context& ctx, CountableIndex a, CountableIndex b) -> Expr const*
{
    return make_delta(
        ctx, Realm::Oblique, space_3d(), Level::Upper, Level::Lower, a, b);
}
} // namespace

TEST(Canonicalize, SumCommutes)
{
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"));
    EXPECT_EQ(
        canon_str(ctx, make_sum(ctx, A, B)),
        canon_str(ctx, make_sum(ctx, B, A)));
}

TEST(Canonicalize, ComponentProductCommutes)
{
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()}, j{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()}, l{ctx.alloc_index_id()};
    auto names = {
        std::pair{i, "i"},
        std::pair{j, "j"},
        std::pair{k, "k"},
        std::pair{l, "l"}};
    auto const* p1 =
        make_tensor_product(ctx, delta_ul(ctx, i, j), delta_ul(ctx, k, l));
    auto const* p2 =
        make_tensor_product(ctx, delta_ul(ctx, k, l), delta_ul(ctx, i, j));
    EXPECT_EQ(canon_str(ctx, p1, names), canon_str(ctx, p2, names));
}

TEST(Canonicalize, LikeTermsCombine)
{
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()}, j{ctx.alloc_index_id()};
    auto const* d = delta_ul(ctx, i, j);
    EXPECT_EQ(
        canon_str(ctx, make_sum(ctx, d, d), {{i, "i"}, {j, "j"}}),
        "2 \\, \\delta^{i}_{j}");
}

TEST(Canonicalize, Cancellation)
{
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()}, j{ctx.alloc_index_id()};
    auto const* d = delta_ul(ctx, i, j);
    EXPECT_EQ(canon_str(ctx, make_difference(ctx, d, d)), "0");
}

TEST(Canonicalize, NumericFoldThroughProduct)
{
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()}, j{ctx.alloc_index_id()};
    auto const* d = delta_ul(ctx, i, j);
    auto const* expr = make_tensor_product(
        ctx,
        make_scalar(ctx, Rational{2}),
        make_tensor_product(ctx, make_scalar(ctx, Rational{3}), d));
    EXPECT_EQ(
        canon_str(ctx, expr, {{i, "i"}, {j, "j"}}), "6 \\, \\delta^{i}_{j}");
}

TEST(Canonicalize, InvariantDyadDoesNotCommute)
{
    // a⊗b is a non-commutative dyad of slot-less rank-1 invariants.
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    EXPECT_NE(
        canon_str(ctx, make_tensor_product(ctx, a, b)),
        canon_str(ctx, make_tensor_product(ctx, b, a)));
}

TEST(Canonicalize, DotCommutesForVectors)
{
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    EXPECT_EQ(
        canon_str(ctx, make_dot(ctx, a, b)),
        canon_str(ctx, make_dot(ctx, b, a)));
}

TEST(Canonicalize, CrossAnticommutesForVectors)
{
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    // The swapped form canonicalizes to the negation of the ordered form.
    auto const* ordered = steps::canonicalize(ctx, make_cross(ctx, a, b));
    auto const* swapped = steps::canonicalize(ctx, make_cross(ctx, b, a));
    auto const* neg = std::get_if<Negate>(&swapped->node);
    ASSERT_NE(neg, nullptr);
    EXPECT_EQ(
        render_with_names(neg->operand, {}), render_with_names(ordered, {}));
}

TEST(Canonicalize, AlphaEquivalentDummies)
{
    // Σ_i δ^i_a δ^i_b  ≡  Σ_p δ^p_a δ^p_b   (bound index renamed)
    Context ctx;
    CountableIndex a{ctx.alloc_index_id()}, b{ctx.alloc_index_id()};
    CountableIndex i{ctx.alloc_index_id()}, p{ctx.alloc_index_id()};
    auto const* e1 = make_explicit_sum(
        ctx,
        i,
        make_tensor_product(ctx, delta_ul(ctx, i, a), delta_ul(ctx, i, b)));
    auto const* e2 = make_explicit_sum(
        ctx,
        p,
        make_tensor_product(ctx, delta_ul(ctx, p, a), delta_ul(ctx, p, b)));
    auto names = {std::pair{a, "a"}, std::pair{b, "b"}};
    EXPECT_EQ(canon_str(ctx, e1, names), canon_str(ctx, e2, names));
}

TEST(Canonicalize, AlphaNestedDummies)
{
    // Σ_i Σ_j δ^i_j  ≡  Σ_p Σ_q δ^p_q   (nested binders α-renamed)
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()}, j{ctx.alloc_index_id()};
    CountableIndex p{ctx.alloc_index_id()}, q{ctx.alloc_index_id()};
    auto const* e1 = make_explicit_sum(
        ctx, i, make_explicit_sum(ctx, j, delta_ul(ctx, i, j)));
    auto const* e2 = make_explicit_sum(
        ctx, p, make_explicit_sum(ctx, q, delta_ul(ctx, p, q)));
    EXPECT_EQ(canon_str(ctx, e1), canon_str(ctx, e2));
}

TEST(Canonicalize, DistinctFreeIndexStillDiffers)
{
    // α-normalization must not over-identify: a free index is not a dummy.
    Context ctx;
    CountableIndex a{ctx.alloc_index_id()}, b{ctx.alloc_index_id()};
    CountableIndex i{ctx.alloc_index_id()};
    auto const* e1 = make_explicit_sum(ctx, i, delta_ul(ctx, i, a));
    auto const* e2 = make_explicit_sum(ctx, i, delta_ul(ctx, i, b));
    auto names = {std::pair{a, "a"}, std::pair{b, "b"}};
    EXPECT_NE(canon_str(ctx, e1, names), canon_str(ctx, e2, names));
}

namespace
{
auto canon_idempotent(Context& ctx, Expr const* e) -> bool
{
    auto const* c1 = steps::canonicalize(ctx, e);
    auto const* c2 = steps::canonicalize(ctx, c1);
    return render_with_names(c1, {}) == render_with_names(c2, {});
}
auto vec(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 1);
}
auto mat(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 2);
}
} // namespace

TEST(Canonicalize, IdempotentAcrossNodeKinds)
{
    Context ctx;
    auto const* a = vec(ctx, "a");
    auto const* b = vec(ctx, "b");
    auto const* c = vec(ctx, "c");
    auto const* d = vec(ctx, "d");
    auto const* A = mat(ctx, "A");
    auto const* B = mat(ctx, "B");
    CountableIndex i{ctx.alloc_index_id()}, j{ctx.alloc_index_id()};
    CountableIndex p{ctx.alloc_index_id()}, q{ctx.alloc_index_id()};
    auto const* N = make_tensor_object(ctx, make_tensor_name("N"), {}, 0);

    std::vector<Expr const*> cases = {
        // invariant binary ops (canon arms + is_component_valued false arms)
        make_ddot(ctx, A, B),
        make_ddot_alt(ctx, A, B),
        make_dot(ctx, A, b),        // rank-2 · rank-1: no reorder
        make_cross(ctx, A, B),      // rank-2 ×: no reorder
        make_scalar_div(ctx, A, B), // symbolic division (no fold)
        make_scalar_div(
            ctx,
            make_scalar(ctx, Rational{6}),
            make_scalar(ctx, Rational{2})), // folds to 3
        make_no_sum(ctx, i, delta_ul(ctx, i, j)),
        make_explicit_sum(ctx, i, delta_ul(ctx, i, j), N), // symbolic bound
        // sorting two same-kind cores → exercises expr_cmp arms
        make_sum(ctx, make_dot(ctx, a, b), make_dot(ctx, c, d)),
        make_sum(ctx, make_ddot(ctx, A, B), make_ddot(ctx, B, A)),
        make_sum(ctx, make_ddot_alt(ctx, A, B), make_ddot_alt(ctx, B, A)),
        make_sum(ctx, make_cross(ctx, a, b), make_cross(ctx, c, d)),
        make_sum(
            ctx,
            make_explicit_sum(ctx, i, delta_ul(ctx, i, j)),
            make_explicit_sum(ctx, p, delta_ul(ctx, p, q))),
        // invariant factors inside a product (is_component_valued false arms)
        make_tensor_product(ctx, delta_ul(ctx, i, j), make_dot(ctx, a, b)),
        make_tensor_product(
            ctx, delta_ul(ctx, i, j), make_scalar_div(ctx, A, B)),
        // a product of two sum atoms (no distribution; expr_cmp Sum arm)
        make_tensor_product(
            ctx,
            make_sum(ctx, delta_ul(ctx, i, j), delta_ul(ctx, p, q)),
            make_sum(ctx, delta_ul(ctx, j, i), delta_ul(ctx, q, p))),
    };
    for (auto const* e: cases)
        EXPECT_TRUE(canon_idempotent(ctx, e));
}

TEST(Canonicalize, FoldsNumericConstants)
{
    Context ctx;
    auto two = []() { return Rational{2}; };
    EXPECT_EQ(
        canon_str(
            ctx,
            make_sum(
                ctx,
                make_scalar(ctx, Rational{2}),
                make_scalar(ctx, Rational{3}))),
        "5");
    EXPECT_EQ(
        canon_str(
            ctx,
            make_difference(
                ctx, make_scalar(ctx, two()), make_scalar(ctx, two()))),
        "0");
    // δ + 2 + 3 → δ + 5 (constant appended last)
    CountableIndex i{ctx.alloc_index_id()}, j{ctx.alloc_index_id()};
    auto const* expr = make_sum(
        ctx,
        make_sum(ctx, delta_ul(ctx, i, j), make_scalar(ctx, Rational{2})),
        make_scalar(ctx, Rational{3}));
    EXPECT_EQ(canon_str(ctx, expr, {{i, "i"}, {j, "j"}}), "\\delta^{i}_{j} + 5");
}

TEST(Canonicalize, ProductReducesToScalarOrZero)
{
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()}, j{ctx.alloc_index_id()};
    auto const* d = delta_ul(ctx, i, j);
    // 2 · 3 → 6 (build_term with no non-scalar factors)
    EXPECT_EQ(
        canon_str(
            ctx,
            make_tensor_product(
                ctx,
                make_scalar(ctx, Rational{2}),
                make_scalar(ctx, Rational{3}))),
        "6");
    // 0 · δ → 0 (build_term coeff 0)
    EXPECT_EQ(
        canon_str(
            ctx, make_tensor_product(ctx, make_scalar(ctx, Rational{0}), d)),
        "0");
    // -δ → -δ (build_term coeff -1)
    EXPECT_EQ(
        canon_str(ctx, make_negate(ctx, d), {{i, "i"}, {j, "j"}}),
        "-\\delta^{i}_{j}");
}

// ---- implicit Einstein summation (vibe 000028) ----------------------------

namespace
{
// delta with explicit realm/levels, for implicit-summation tests.
auto idelta(
    Context& ctx,
    Realm realm,
    Level l0,
    Level l1,
    CountableIndex a,
    CountableIndex b) -> Expr const*
{
    return make_delta(ctx, realm, space_3d(), l0, l1, a, b);
}
} // namespace

TEST(ImplicitSum, OrthonormalPairContracts)
{
    // δ^r_m δ^r_n  (orthonormal, r twice)  ==  Σ_r δ^r_m δ^r_n
    Context ctx;
    CountableIndex r{ctx.alloc_index_id()}, m{ctx.alloc_index_id()},
        n{ctx.alloc_index_id()};
    auto const* implicit = make_tensor_product(
        ctx,
        idelta(ctx, Realm::Orthonormal, Level::Upper, Level::Lower, r, m),
        idelta(ctx, Realm::Orthonormal, Level::Upper, Level::Lower, r, n));
    auto const* explicit_ = make_explicit_sum(ctx, r, implicit);
    EXPECT_TRUE(algebraic_eq(ctx, implicit, explicit_));
}

TEST(ImplicitSum, ObliqueMixedLevelContracts)
{
    // δ^r_m δ_r^n  (oblique, one upper one lower)  ==  Σ_r …
    Context ctx;
    CountableIndex r{ctx.alloc_index_id()}, m{ctx.alloc_index_id()},
        n{ctx.alloc_index_id()};
    auto const* implicit = make_tensor_product(
        ctx,
        idelta(ctx, Realm::Oblique, Level::Upper, Level::Lower, r, m),
        idelta(ctx, Realm::Oblique, Level::Lower, Level::Upper, r, n));
    auto const* explicit_ = make_explicit_sum(ctx, r, implicit);
    EXPECT_TRUE(algebraic_eq(ctx, implicit, explicit_));
}

TEST(ImplicitSum, ObliqueTraceContracts)
{
    // A lone δ^i_i is a trace: contracts to Σ_i δ^i_i.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* implicit =
        idelta(ctx, Realm::Oblique, Level::Upper, Level::Lower, i, i);
    auto const* explicit_ = make_explicit_sum(ctx, i, implicit);
    EXPECT_TRUE(algebraic_eq(ctx, implicit, explicit_));
}

TEST(ImplicitSum, ObliqueSameLevelThrows)
{
    // δ^r_m δ^r_n with r upper in both is ill-formed in Oblique (no override).
    Context ctx;
    CountableIndex r{ctx.alloc_index_id()}, m{ctx.alloc_index_id()},
        n{ctx.alloc_index_id()};
    auto const* bad = make_tensor_product(
        ctx,
        idelta(ctx, Realm::Oblique, Level::Upper, Level::Lower, r, m),
        idelta(ctx, Realm::Oblique, Level::Upper, Level::Lower, r, n));
    EXPECT_THROW(steps::canonicalize(ctx, bad), std::invalid_argument);
}

TEST(ImplicitSum, ThreeOccurrencesThrows)
{
    // r appearing three times (orthonormal) needs an explicit annotation.
    Context ctx;
    CountableIndex r{ctx.alloc_index_id()}, m{ctx.alloc_index_id()},
        n{ctx.alloc_index_id()}, p{ctx.alloc_index_id()};
    auto const* bad = make_tensor_product(
        ctx,
        make_tensor_product(
            ctx,
            idelta(ctx, Realm::Orthonormal, Level::Upper, Level::Lower, r, m),
            idelta(ctx, Realm::Orthonormal, Level::Upper, Level::Lower, r, n)),
        idelta(ctx, Realm::Orthonormal, Level::Upper, Level::Lower, r, p));
    EXPECT_THROW(steps::canonicalize(ctx, bad), std::invalid_argument);
}

TEST(ImplicitSum, ExplicitSumOverrideSilencesError)
{
    // The same same-level Oblique pair is valid once wrapped: the binder is the
    // override, so no implicit rule (and no error) applies to r.
    Context ctx;
    CountableIndex r{ctx.alloc_index_id()}, m{ctx.alloc_index_id()},
        n{ctx.alloc_index_id()};
    auto const* wrapped = make_explicit_sum(
        ctx,
        r,
        make_tensor_product(
            ctx,
            idelta(ctx, Realm::Oblique, Level::Upper, Level::Lower, r, m),
            idelta(ctx, Realm::Oblique, Level::Upper, Level::Lower, r, n)));
    EXPECT_NO_THROW(steps::canonicalize(ctx, wrapped));
}

TEST(ImplicitSum, NoSumOverrideKeepsIndexFree)
{
    // NoSum suppresses contraction: a repeated orthonormal index stays free and
    // does not error — so it is NOT equal to the summed form.
    Context ctx;
    CountableIndex r{ctx.alloc_index_id()}, m{ctx.alloc_index_id()},
        n{ctx.alloc_index_id()};
    auto const* body = make_tensor_product(
        ctx,
        idelta(ctx, Realm::Orthonormal, Level::Upper, Level::Lower, r, m),
        idelta(ctx, Realm::Orthonormal, Level::Upper, Level::Lower, r, n));
    auto const* kept = make_no_sum(ctx, r, body);
    EXPECT_NO_THROW(steps::canonicalize(ctx, kept));
    EXPECT_FALSE(algebraic_eq(ctx, kept, make_explicit_sum(ctx, r, body)));
}

TEST(ImplicitSum, CollectionDoesNotAutoContract)
{
    // Collection realm never auto-sums: a doubled index stays a free product,
    // distinct from the explicit-sum form, and does not error.
    Context ctx;
    CountableIndex r{ctx.alloc_index_id()}, m{ctx.alloc_index_id()},
        n{ctx.alloc_index_id()};
    auto const* implicit = make_tensor_product(
        ctx,
        idelta(ctx, Realm::Collection, Level::Upper, Level::Lower, r, m),
        idelta(ctx, Realm::Collection, Level::Upper, Level::Lower, r, n));
    EXPECT_NO_THROW(steps::canonicalize(ctx, implicit));
    EXPECT_FALSE(
        algebraic_eq(ctx, implicit, make_explicit_sum(ctx, r, implicit)));
}

// ---- expand_products: Dot / Cross distribution ----------------------------

TEST(ExpandProducts, DotLeftSumDistributes)
{
    // Dot(A + B, C)  →  Dot(A, C) + Dot(B, C)
    Context ctx;
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto* C = make_tensor_object(ctx, make_tensor_name("C"));
    auto* expr = make_dot(ctx, make_sum(ctx, A, B), C);
    auto const* after = steps::expand_products(ctx, expr);
    auto const* s = std::get_if<Sum>(&after->node);
    ASSERT_NE(s, nullptr);
    EXPECT_NE(std::get_if<Dot>(&s->left->node), nullptr);
    EXPECT_NE(std::get_if<Dot>(&s->right->node), nullptr);
}

TEST(ExpandProducts, CrossRightDiffDistributes)
{
    // Cross(A, B - C)  →  Cross(A, B) - Cross(A, C)
    Context ctx;
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto* C = make_tensor_object(ctx, make_tensor_name("C"));
    auto* expr = make_cross(ctx, A, make_difference(ctx, B, C));
    auto const* after = steps::expand_products(ctx, expr);
    EXPECT_NE(std::get_if<Difference>(&after->node), nullptr);
}

// ---- unroll_sums_for / has_explicit_sum_for --------------------------------

TEST(UnrollSumsFor, OnlySpecifiedIndexUnrolled)
{
    // ExplicitSum{i, ExplicitSum{j, δ^i_j}}  — ask to unroll only j.
    // After the call, the outer sum (i) must remain; the inner (j) is gone.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};

    auto* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    auto* inner = make_explicit_sum(ctx, j, d);
    auto* expr = make_explicit_sum(ctx, i, inner);

    auto const* after = steps::unroll_sums_for(ctx, expr, {CountableIndex{j}});

    // Outer ExplicitSum (i) must survive.
    auto const* outer = std::get_if<ExplicitSum>(&after->node);
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->index.id, i.id);
    // Body must no longer be an ExplicitSum (j was unrolled).
    EXPECT_EQ(std::get_if<ExplicitSum>(&outer->body->node), nullptr);
}

TEST(HasExplicitSumFor, ReturnsTrueWhenPresent)
{
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto* body = make_tensor_object(ctx, make_tensor_name("A"));
    auto* expr = make_explicit_sum(ctx, i, body);
    EXPECT_TRUE(steps::has_explicit_sum_for(expr, {CountableIndex{i}}));
}

TEST(HasExplicitSumFor, ReturnsFalseWhenAbsent)
{
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto* body = make_tensor_object(ctx, make_tensor_name("A"));
    auto* expr = make_explicit_sum(ctx, i, body);
    // j is not in expr
    EXPECT_FALSE(steps::has_explicit_sum_for(expr, {CountableIndex{j}}));
}

TEST(HasExplicitSumFor, NestedInsideBinaryNodes)
{
    // ExplicitSum nested inside Dot / ScalarDiv / Negate — the walker must
    // descend through all binary node types.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto* body = make_tensor_object(ctx, make_tensor_name("A"));
    auto* es = make_explicit_sum(ctx, i, body);
    auto* B = make_tensor_object(ctx, make_tensor_name("B"));

    // Wrap in Dot, Negate, ScalarDiv, DDot, DDotAlt, Cross — walker must find
    // i.
    EXPECT_TRUE(
        steps::has_explicit_sum_for(make_dot(ctx, es, B), {CountableIndex{i}}));
    EXPECT_TRUE(steps::has_explicit_sum_for(
        make_scalar_div(ctx, es, B), {CountableIndex{i}}));
    EXPECT_TRUE(
        steps::has_explicit_sum_for(make_negate(ctx, es), {CountableIndex{i}}));
    EXPECT_TRUE(
        steps::has_explicit_sum_for(make_ddot(ctx, es, B), {CountableIndex{i}}));
    EXPECT_TRUE(steps::has_explicit_sum_for(
        make_ddot_alt(ctx, es, B), {CountableIndex{i}}));
    EXPECT_TRUE(steps::has_explicit_sum_for(
        make_cross(ctx, es, B), {CountableIndex{i}}));
    EXPECT_TRUE(
        steps::has_explicit_sum_for(make_sum(ctx, es, B), {CountableIndex{i}}));
    EXPECT_TRUE(steps::has_explicit_sum_for(
        make_difference(ctx, es, B), {CountableIndex{i}}));
    // NoSum body
    CountableIndex k{ctx.alloc_index_id()};
    EXPECT_TRUE(steps::has_explicit_sum_for(
        make_no_sum(ctx, k, es), {CountableIndex{i}}));
    // Bounded ExplicitSum body — the walker must check the bound too.
    auto* bound_es =
        make_explicit_sum(ctx, i, body, make_scalar(ctx, Rational{3}));
    EXPECT_TRUE(steps::has_explicit_sum_for(bound_es, {CountableIndex{i}}));
}

TEST(UnrollSumsFor, ImplicitIndexUnrolled)
{
    // δ^i_i (no ExplicitSum wrapper) — asking to unroll i should substitute
    // concrete values and produce a Sum tree.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};

    auto const* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{i});

    auto const* after = steps::unroll_sums_for(ctx, d, {CountableIndex{i}});

    // Must no longer be a TensorObject — should be a Sum tree.
    EXPECT_EQ(std::get_if<TensorObject>(&after->node), nullptr);
    EXPECT_NE(std::get_if<Sum>(&after->node), nullptr);
}

TEST(HasExplicitSumFor, FreeIndexDetected)
{
    // δ^i_i has no ExplicitSum but i is free — should return true.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};

    auto const* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{i});

    EXPECT_TRUE(steps::has_explicit_sum_for(d, {CountableIndex{i}}));
}

TEST(HasExplicitSumFor, FreeIndexSuppressedByNoSum)
{
    // NoSum{i, δ^i_i} — i is suppressed, should return false.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};

    auto const* d = make_delta(
        ctx,
        Realm::Oblique,
        sp,
        Level::Upper,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{i});
    auto const* ns = make_no_sum(ctx, i, d);

    EXPECT_FALSE(steps::has_explicit_sum_for(ns, {CountableIndex{i}}));
}

TEST(ExpandProducts, DDotDistributes)
{
    // DDot(A + B, C)  →  DDot(A, C) + DDot(B, C)
    Context ctx;
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto* C = make_tensor_object(ctx, make_tensor_name("C"));
    auto* expr = make_ddot(ctx, make_sum(ctx, A, B), C);
    auto const* after = steps::expand_products(ctx, expr);
    auto const* s = std::get_if<Sum>(&after->node);
    ASSERT_NE(s, nullptr);
    EXPECT_NE(std::get_if<DDot>(&s->left->node), nullptr);
}

TEST(ExpandProducts, DDotAltDistributes)
{
    // DDotAlt(A, B + C)  →  DDotAlt(A, B) + DDotAlt(A, C)
    Context ctx;
    auto* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto* B = make_tensor_object(ctx, make_tensor_name("B"));
    auto* C = make_tensor_object(ctx, make_tensor_name("C"));
    auto* expr = make_ddot_alt(ctx, A, make_sum(ctx, B, C));
    auto const* after = steps::expand_products(ctx, expr);
    EXPECT_NE(std::get_if<Sum>(&after->node), nullptr);
}

// ---- symmetry orbit canonicalization (vibe 000047) -------------------------

namespace
{
auto eps3(Context& ctx, Level lvl, IndexAssoc x, IndexAssoc y, IndexAssoc z)
    -> Expr const*
{
    return make_levi_civita(
        ctx, Realm::Oblique, space_3d(), {lvl, lvl, lvl}, {x, y, z});
}
} // namespace

TEST(SymmetryCanon, GeneratorlessTensorUnchanged)
{
    // A well-known tensor that carries traits but no symmetry generators (the
    // identity tensor I) passes through canon_symmetry untouched.
    Context ctx;
    auto const* I = make_identity(ctx);
    EXPECT_TRUE(structural_eq(steps::canonicalize(ctx, I), I));
}

TEST(SymmetryCanon, DeltaSlotSwapIsEqual)
{
    // δ is symmetric: δ^a_b == δ_b^a (the slots swapped, levels travelling with
    // their index).  Both must canonicalize to the same orbit-minimal form.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    auto const* d_ab =
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, a, b);
    auto const* d_ba =
        make_delta(ctx, Realm::Oblique, sp, Level::Lower, Level::Upper, b, a);
    EXPECT_TRUE(algebraic_eq(ctx, d_ab, d_ba));
}

TEST(SymmetryCanon, EpsCyclicShiftIsEqual)
{
    // Even (cyclic) permutation preserves value: ε^{ijk} == ε^{jki}.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    auto const* ijk =
        eps3(ctx, Level::Upper, IndexAssoc{i}, IndexAssoc{j}, IndexAssoc{k});
    auto const* jki =
        eps3(ctx, Level::Upper, IndexAssoc{j}, IndexAssoc{k}, IndexAssoc{i});
    EXPECT_TRUE(algebraic_eq(ctx, ijk, jki));
}

TEST(SymmetryCanon, EpsTranspositionFlipsSign)
{
    // Odd permutation flips sign: ε^{ijk} + ε^{jik} == 0.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    auto const* ijk =
        eps3(ctx, Level::Upper, IndexAssoc{i}, IndexAssoc{j}, IndexAssoc{k});
    auto const* jik =
        eps3(ctx, Level::Upper, IndexAssoc{j}, IndexAssoc{i}, IndexAssoc{k});
    EXPECT_TRUE(algebraic_eq(
        ctx, make_sum(ctx, ijk, jik), make_scalar(ctx, Rational{0})));
}

TEST(SymmetryCanon, EpsRepeatedIndexIsZero)
{
    // An arrangement reachable with both signs is identically zero: ε^{11k} =
    // 0. (Repeated *countable* indices would be rejected earlier as an
    // ill-formed Oblique same-level contraction; concrete repeats are the path
    // that reaches the sign-conflict branch in canon.)
    Context ctx;
    auto const* z = eps3(
        ctx,
        Level::Upper,
        IndexAssoc{ConcreteIndex{1}},
        IndexAssoc{ConcreteIndex{1}},
        IndexAssoc{ConcreteIndex{2}});
    EXPECT_TRUE(algebraic_eq(ctx, z, make_scalar(ctx, Rational{0})));
}

// ---- infer_rank ------------------------------------------------------------

TEST(InferRank, Leaves)
{
    Context ctx;
    EXPECT_EQ(
        infer_rank(make_tensor_object(ctx, make_tensor_name("a"), {}, 1)),
        std::optional<int>{1});
    EXPECT_EQ(infer_rank(make_identity(ctx)), std::optional<int>{2});
    EXPECT_EQ(infer_rank(make_scalar(ctx, Rational{3})), std::optional<int>{0});
    EXPECT_EQ(
        infer_rank(make_tensor_object(ctx, make_tensor_name("B"))),
        std::nullopt);
}

TEST(InferRank, ThroughOperators)
{
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    auto const* I = make_identity(ctx);
    EXPECT_EQ(infer_rank(make_tensor_product(ctx, a, b)), std::optional<int>{2});
    EXPECT_EQ(infer_rank(make_dot(ctx, a, b)), std::optional<int>{0});
    EXPECT_EQ(infer_rank(make_dot(ctx, a, I)), std::optional<int>{1});
    EXPECT_EQ(infer_rank(make_cross(ctx, a, I)), std::optional<int>{2});
    EXPECT_EQ(infer_rank(make_ddot(ctx, I, I)), std::optional<int>{0});
    EXPECT_EQ(infer_rank(make_sum(ctx, a, b)), std::optional<int>{1});
    EXPECT_EQ(infer_rank(make_negate(ctx, a)), std::optional<int>{1});
}

TEST(InferRank, UnknownLeafAndIllFormed)
{
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* B = make_tensor_object(ctx, make_tensor_name("B")); // no rank
    EXPECT_EQ(infer_rank(make_tensor_product(ctx, B, a)), std::nullopt);
    // A dot of two scalars would drive the rank to -2: ill-formed → nullopt.
    auto const* s = make_scalar(ctx, Rational{2});
    EXPECT_EQ(infer_rank(make_dot(ctx, s, s)), std::nullopt);
}

// ---- distribute_contraction ------------------------------------------------

namespace
{
auto rank1(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 1);
}
} // namespace

TEST(DistributeContraction, CrossOverRightDyad)
{
    // a × (u ⊗ v) → (a × u) ⊗ v
    Context ctx;
    auto const* a = rank1(ctx, "a");
    auto const* u = rank1(ctx, "u");
    auto const* v = rank1(ctx, "v");
    auto const* res = steps::distribute_contraction(
        ctx, make_cross(ctx, a, make_tensor_product(ctx, u, v)));
    auto const* tp = std::get_if<TensorProduct>(&res->node);
    ASSERT_NE(tp, nullptr);
    EXPECT_TRUE(std::holds_alternative<Cross>(tp->left->node));
    EXPECT_EQ(tp->right, v); // v carried unchanged
}

TEST(DistributeContraction, DotOverLeftDyad)
{
    // (u ⊗ v) · a → u ⊗ (v · a)
    Context ctx;
    auto const* a = rank1(ctx, "a");
    auto const* u = rank1(ctx, "u");
    auto const* v = rank1(ctx, "v");
    auto const* res = steps::distribute_contraction(
        ctx, make_dot(ctx, make_tensor_product(ctx, u, v), a));
    auto const* tp = std::get_if<TensorProduct>(&res->node);
    ASSERT_NE(tp, nullptr);
    EXPECT_EQ(tp->left, u);
    EXPECT_TRUE(std::holds_alternative<Dot>(tp->right->node));
}

TEST(DistributeContraction, FloatsScalarThrough)
{
    // a × (s ⊗ v) → s ⊗ (a × v): a scalar near leg floats out.
    Context ctx;
    auto const* a = rank1(ctx, "a");
    auto const* v = rank1(ctx, "v");
    auto const* s = make_scalar(ctx, Rational{3});
    auto const* res = steps::distribute_contraction(
        ctx, make_cross(ctx, a, make_tensor_product(ctx, s, v)));
    auto const* tp = std::get_if<TensorProduct>(&res->node);
    ASSERT_NE(tp, nullptr);
    EXPECT_EQ(tp->left, s);
    EXPECT_TRUE(std::holds_alternative<Cross>(tp->right->node));
}

TEST(DistributeContraction, PlainContractionUnchanged)
{
    Context ctx;
    auto const* dot = make_dot(ctx, rank1(ctx, "a"), rank1(ctx, "b"));
    EXPECT_EQ(steps::distribute_contraction(ctx, dot), dot);
}

TEST(Canonicalize, DyadLegsDoNotCommute)
{
    // A dyad of two basis vectors is ordered: e_i ⊗ e_j ≠ e_j ⊗ e_i.  A basis
    // vector is a rank-1 indexed tensor — invariant, not a commuting scalar.
    Context ctx;
    auto evec = [&](CountableIndex idx)
    {
        return make_tensor_object(
            ctx,
            make_tensor_name("e"),
            {SlotBinding{
                IndexSlot{Level::Lower, Realm::Orthonormal, space_3d()},
                IndexAssoc{idx}}},
            1);
    };
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* ij = make_tensor_product(ctx, evec(i), evec(j));
    auto const* ji = make_tensor_product(ctx, evec(j), evec(i));
    EXPECT_FALSE(algebraic_eq(ctx, ij, ji));
}

TEST(Canonicalize, NestedSumOrderIsNormalized)
{
    // Σ_i Σ_j body == Σ_j Σ_i body (Fubini), even when i and j play distinct
    // roles in the body (asymmetric coordinate, ordered dyad).
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto slot = [&](CountableIndex x)
    {
        return SlotBinding{
            IndexSlot{Level::Lower, Realm::Orthonormal, space_3d()},
            IndexAssoc{x}};
    };
    auto const* coord =
        make_tensor_object(ctx, make_tensor_name("A"), {slot(i), slot(j)}, 0);
    auto const* ei =
        make_tensor_object(ctx, make_tensor_name("e"), {slot(i)}, 1);
    auto const* ej =
        make_tensor_object(ctx, make_tensor_name("e"), {slot(j)}, 1);
    // body = A_ij e_i e_j
    auto const* body =
        make_tensor_product(ctx, make_tensor_product(ctx, coord, ei), ej);
    auto const* sij =
        make_explicit_sum(ctx, i, make_explicit_sum(ctx, j, body));
    auto const* sji =
        make_explicit_sum(ctx, j, make_explicit_sum(ctx, i, body));

    EXPECT_TRUE(algebraic_eq(ctx, sij, sji));
    // Idempotent.
    auto const* c = steps::canonicalize(ctx, sij);
    EXPECT_TRUE(structural_eq(steps::canonicalize(ctx, c), c));
}

// ---- expand_double_dot -----------------------------------------------------

TEST(ExpandDoubleDot, DyadVertical)
{
    // (a⊗b):(c⊗d) → (a·c)(b·d)
    Context ctx;
    auto v = [&](char const* n)
    { return make_tensor_object(ctx, make_tensor_name(n), {}, 1); };
    auto const* a = v("a");
    auto const* b = v("b");
    auto const* c = v("c");
    auto const* d = v("d");
    auto const* e = make_ddot(
        ctx, make_tensor_product(ctx, a, b), make_tensor_product(ctx, c, d));
    auto const* expected =
        make_tensor_product(ctx, make_dot(ctx, a, c), make_dot(ctx, b, d));
    EXPECT_TRUE(algebraic_eq(ctx, steps::expand_double_dot(ctx, e), expected));
}

TEST(ExpandDoubleDot, DyadAlternate)
{
    // (a⊗b)··(c⊗d) → (a·d)(b·c)
    Context ctx;
    auto v = [&](char const* n)
    { return make_tensor_object(ctx, make_tensor_name(n), {}, 1); };
    auto const* a = v("a");
    auto const* b = v("b");
    auto const* c = v("c");
    auto const* d = v("d");
    auto const* e = make_ddot_alt(
        ctx, make_tensor_product(ctx, a, b), make_tensor_product(ctx, c, d));
    auto const* expected =
        make_tensor_product(ctx, make_dot(ctx, a, d), make_dot(ctx, b, c));
    EXPECT_TRUE(algebraic_eq(ctx, steps::expand_double_dot(ctx, e), expected));
}

TEST(ExpandDoubleDot, DistributesThroughSummationBinders)
{
    // (Σ_i e_i⊗e_i) : (Σ_j e_j⊗e_j) expands and (in 3D, orthonormal) the body
    // becomes (e·e)(e·e); the sums survive (one per side).
    Context ctx;
    auto evec = [&](CountableIndex idx)
    {
        return make_tensor_object(
            ctx,
            make_tensor_name("e"),
            {SlotBinding{
                IndexSlot{Level::Lower, Realm::Orthonormal, space_3d()},
                IndexAssoc{idx}}},
            1);
    };
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* li =
        make_explicit_sum(ctx, i, make_tensor_product(ctx, evec(i), evec(i)));
    auto const* lj =
        make_explicit_sum(ctx, j, make_tensor_product(ctx, evec(j), evec(j)));
    auto const* res = steps::expand_double_dot(ctx, make_ddot(ctx, li, lj));
    // Two nested sums over a product of two dots — no DDot left.
    auto const* outer = std::get_if<ExplicitSum>(&res->node);
    ASSERT_NE(outer, nullptr);
    EXPECT_TRUE(std::holds_alternative<ExplicitSum>(outer->body->node));
}

TEST(ExpandDoubleDot, NonDyadUnchanged)
{
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);
    // A : (a) — right side is not a 2-leg dyad; left alone.
    auto const* e = make_ddot(ctx, A, a);
    EXPECT_EQ(steps::expand_double_dot(ctx, e), e);
}

// ---- expand_dyad_ops -------------------------------------------------------

TEST(ExpandDyadOps, TraceVecTransposeOnDyad)
{
    Context ctx;
    auto v = [&](char const* n)
    { return make_tensor_object(ctx, make_tensor_name(n), {}, 1); };
    auto const* a = v("a");
    auto const* b = v("b");
    auto const* dyad = make_tensor_product(ctx, a, b);

    EXPECT_TRUE(algebraic_eq(
        ctx,
        steps::expand_dyad_ops(ctx, make_trace(ctx, dyad)),
        make_dot(ctx, a, b)));
    EXPECT_TRUE(algebraic_eq(
        ctx,
        steps::expand_dyad_ops(ctx, make_vector_invariant(ctx, dyad)),
        make_cross(ctx, a, b)));
    EXPECT_TRUE(algebraic_eq(
        ctx,
        steps::expand_dyad_ops(ctx, make_transpose(ctx, dyad)),
        make_tensor_product(ctx, b, a)));
}

TEST(ExpandDyadOps, TransposeOfSymmetricWellKnownIsSelf)
{
    Context ctx;
    auto const* I = make_identity(ctx);
    EXPECT_TRUE(
        structural_eq(steps::expand_dyad_ops(ctx, make_transpose(ctx, I)), I));
}

TEST(ExpandDyadOps, DistributesOverSum)
{
    // tr(a⊗b + c⊗d) → a·b + c·d
    Context ctx;
    auto v = [&](char const* n)
    { return make_tensor_object(ctx, make_tensor_name(n), {}, 1); };
    auto const* a = v("a");
    auto const* b = v("b");
    auto const* c = v("c");
    auto const* d = v("d");
    auto const* sum = make_sum(
        ctx, make_tensor_product(ctx, a, b), make_tensor_product(ctx, c, d));
    auto const* expected =
        make_sum(ctx, make_dot(ctx, a, b), make_dot(ctx, c, d));
    EXPECT_TRUE(algebraic_eq(
        ctx, steps::expand_dyad_ops(ctx, make_trace(ctx, sum)), expected));
}

TEST(ExpandDyadOps, NonDyadUnchanged)
{
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);
    auto const* e = make_trace(ctx, A); // tr of a bare rank-2 tensor
    EXPECT_EQ(steps::expand_dyad_ops(ctx, e), e);
}

TEST(InferRank, UnaryOps)
{
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);
    EXPECT_EQ(infer_rank(make_trace(ctx, A)), std::optional<int>{0});
    EXPECT_EQ(infer_rank(make_vector_invariant(ctx, A)), std::optional<int>{1});
    EXPECT_EQ(infer_rank(make_transpose(ctx, A)), std::optional<int>{2});
}

TEST(ContractDelta, FiresOnImplicitSumForm)
{
    // δ_ij δ_ij with sums still implicit (no ExplicitSum) contracts the same as
    // the explicit Σ form: contract over i gives δ_jj (then j is implicit).
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto delta = [&](CountableIndex a, CountableIndex b)
    {
        return make_delta(
            ctx,
            Realm::Orthonormal,
            sp,
            Level::Lower,
            Level::Lower,
            IndexAssoc{a},
            IndexAssoc{b});
    };
    auto const* prod = make_tensor_product(ctx, delta(i, j), delta(i, j));
    auto const* res = steps::contract_delta(ctx, prod);
    // Expected: Σ_j δ_jj  (trace form), equivalently the canonical of it.
    auto const* expected = make_explicit_sum(ctx, j, delta(j, j));
    EXPECT_TRUE(algebraic_eq(ctx, res, expected));
}
