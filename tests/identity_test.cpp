#include <gtest/gtest.h>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/identity.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>

#include <vector>

using namespace tender;

// ---- helpers ---------------------------------------------------------------

namespace
{

auto delta_ul(Context& ctx, IndexSpace const* sp, IndexAssoc a, IndexAssoc b)
    -> Expr const*
{
    return make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, a, b);
}

auto delta_ll(Context& ctx, IndexSpace const* sp, IndexAssoc a, IndexAssoc b)
    -> Expr const*
{
    return make_delta(ctx, Realm::Oblique, sp, Level::Lower, Level::Lower, a, b);
}

// The delta-contraction identity  Σ_p δ^p_A δ^p_B  =  δ_{AB}.
auto delta_contraction(Context& ctx, IndexSpace const* sp) -> Identity
{
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    auto const* lhs = make_explicit_sum(
        ctx,
        p,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, p, a), delta_ul(ctx, sp, p, b)));
    auto const* rhs = delta_ll(ctx, sp, a, b);
    return Identity{"delta-contraction", lhs, rhs};
}

} // namespace

// ---- apply_identity --------------------------------------------------------

TEST(ApplyIdentity, DeltaContractionRewritesToDelta)
{
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        q,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, q, m), delta_ul(ctx, sp, q, n)));

    auto const* result = apply_identity(ctx, target, id);
    EXPECT_TRUE(algebraic_eq(ctx, result, delta_ll(ctx, sp, m, n)));
}

TEST(ApplyIdentity, FactorOrderIndependent)
{
    // The same contraction with the two deltas in the other order still fires
    // (the scalars/tensors regions are matched as the canonical form arranges).
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        q,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, q, n), delta_ul(ctx, sp, q, m)));

    auto const* result = apply_identity(ctx, target, id);
    EXPECT_TRUE(algebraic_eq(ctx, result, delta_ll(ctx, sp, m, n)));
}

TEST(ApplyIdentity, NoMatchReturnsCanonicalUnchanged)
{
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* target = delta_ul(ctx, sp, m, n);

    auto const* result = apply_identity(ctx, target, id);
    EXPECT_TRUE(algebraic_eq(ctx, result, target));
}

TEST(ApplyIdentity, MatchesNestedSubexpression)
{
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    CountableIndex r{ctx.alloc_index_id()};
    CountableIndex s{ctx.alloc_index_id()};
    auto const* contraction = make_explicit_sum(
        ctx,
        q,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, q, m), delta_ul(ctx, sp, q, n)));
    // δ_{rs} + Σ_q δ^q_m δ^q_n  →  δ_{rs} + δ_{mn}
    auto const* target = make_sum(ctx, delta_ll(ctx, sp, r, s), contraction);

    auto const* result = apply_identity(ctx, target, id);
    auto const* expected =
        make_sum(ctx, delta_ll(ctx, sp, r, s), delta_ll(ctx, sp, m, n));
    EXPECT_TRUE(algebraic_eq(ctx, result, expected));
}

TEST(ApplyIdentity, AsDerivationStep)
{
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        q,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, q, m), delta_ul(ctx, sp, q, n)));

    Derivation drv{ctx, target};
    drv.step(steps::apply_identity(id));
    EXPECT_TRUE(algebraic_eq(ctx, drv.current(), delta_ll(ctx, sp, m, n)));
    EXPECT_EQ(drv.history().size(), 2u);
}

// ---- partial sub-product matching (vibe 000058 / C14) ----------------------

TEST(ApplyIdentity, FiresOnSubProductInsideLargerTerm)
{
    // The δ-contraction sits among an extra (unrelated) tensor factor A.  The
    // binary-tree matcher could not fire here — there is no node holding just
    // Σ_p δ^p_m δ^p_n once the binder floats to the head — but the flat-form
    // matcher reduces the sub-product, carrying A through.
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);
    auto const* target = make_tensor_product(
        ctx,
        make_explicit_sum(
            ctx,
            p,
            make_tensor_product(
                ctx, delta_ul(ctx, sp, p, m), delta_ul(ctx, sp, p, n))),
        A);

    auto const* result = apply_identity(ctx, target, id);
    auto const* expected = make_tensor_product(ctx, delta_ll(ctx, sp, m, n), A);
    EXPECT_TRUE(algebraic_eq(ctx, result, expected));
}

TEST(ApplyIdentity, RepeatedApplicationReducesEachSubProduct)
{
    // Two independent contractions in one term: Σ_p Σ_q δ^p_m δ^p_n δ^q_r
    // δ^q_s. Each apply fires once, so two applications reduce both to δ_mn
    // δ_rs.
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    CountableIndex r{ctx.alloc_index_id()};
    CountableIndex s{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        p,
        make_explicit_sum(
            ctx,
            q,
            make_tensor_product(
                ctx,
                make_tensor_product(
                    ctx, delta_ul(ctx, sp, p, m), delta_ul(ctx, sp, p, n)),
                make_tensor_product(
                    ctx, delta_ul(ctx, sp, q, r), delta_ul(ctx, sp, q, s)))));

    auto const* once = apply_identity(ctx, target, id);
    auto const* twice = apply_identity(ctx, once, id);
    auto const* expected = make_tensor_product(
        ctx, delta_ll(ctx, sp, m, n), delta_ll(ctx, sp, r, s));
    EXPECT_TRUE(algebraic_eq(ctx, twice, expected));
}

TEST(ApplyIdentity, CommuteFiresInsideCrossChain)
{
    // I × x = x × I applied to a × I × b: the flat Cross chain [a, I, b] holds
    // no `I × b` node, but sub-chain matching rewrites the run in place to
    // reach a × b × I.  (The same payoff the binary-tree fallback gave, now on
    // the flat form.)
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    auto const* x = make_tensor_object(ctx, make_tensor_name("x"), {}, 1);
    auto const* I = make_identity(ctx);
    Identity const commute{
        "I-commute", make_cross(ctx, I, x), make_cross(ctx, x, I)};

    auto const* target = make_cross(ctx, make_cross(ctx, a, I), b);
    auto const* result = apply_identity(ctx, target, commute);
    auto const* expected = make_cross(ctx, a, make_cross(ctx, b, I));
    EXPECT_TRUE(algebraic_eq(ctx, result, expected));
}

// ---- a real index identity: two-index eps-delta ----------------------------

TEST(ApplyIdentity, EpsDeltaTwoIndex)
{
    Context ctx;
    auto const* sp = space_3d();

    auto eps = [&](Level lvl, IndexAssoc x, IndexAssoc y, IndexAssoc z)
    {
        return make_levi_civita(
            ctx, Realm::Oblique, sp, {lvl, lvl, lvl}, {x, y, z});
    };

    // Identity:  Σ_i Σ_j ε^{ijk} ε_{ijl}  =  2 δ^k_l.
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};
    auto const* lhs = make_explicit_sum(
        ctx,
        i,
        make_explicit_sum(
            ctx,
            j,
            make_tensor_product(
                ctx, eps(Level::Upper, i, j, k), eps(Level::Lower, i, j, l))));
    auto const* rhs = make_tensor_product(
        ctx, make_scalar(ctx, Rational{2}), delta_ul(ctx, sp, k, l));
    Identity id{"eps-delta-2", lhs, rhs};

    // Target written with fresh indices.
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex c{ctx.alloc_index_id()};
    CountableIndex d{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        a,
        make_explicit_sum(
            ctx,
            b,
            make_tensor_product(
                ctx, eps(Level::Upper, a, b, c), eps(Level::Lower, a, b, d))));

    auto const* result = apply_identity(ctx, target, id);
    auto const* expected = make_tensor_product(
        ctx, make_scalar(ctx, Rational{2}), delta_ul(ctx, sp, c, d));
    EXPECT_TRUE(algebraic_eq(ctx, result, expected));
}

// ---- subtree pattern variables (vibe 000051) -------------------------------

TEST(SubtreeVars, MatchesAndInstantiates)
{
    // Identity (a⊗b):(c⊗d) = (a·c)(b·d) with a,b,c,d as subtree variables.
    Context ctx;
    auto v = [&](char const* n)
    { return make_tensor_object(ctx, make_tensor_name(n), {}, 1); };
    auto const* a = v("a");
    auto const* b = v("b");
    auto const* c = v("c");
    auto const* d = v("d");
    Identity id{
        "ddot",
        make_ddot(
            ctx, make_tensor_product(ctx, a, b), make_tensor_product(ctx, c, d)),
        make_tensor_product(ctx, make_dot(ctx, a, c), make_dot(ctx, b, d))};

    // Apply to a different dyad pair (x⊗y):(u⊗v).
    auto const* x = v("x");
    auto const* y = v("y");
    auto const* u = v("u");
    auto const* w = v("w");
    auto const* target = make_ddot(
        ctx, make_tensor_product(ctx, x, y), make_tensor_product(ctx, u, w));
    auto const* result = apply_identity(ctx, target, id);

    auto const* expected =
        make_tensor_product(ctx, make_dot(ctx, x, u), make_dot(ctx, y, w));
    EXPECT_TRUE(algebraic_eq(ctx, result, expected));
}

TEST(SubtreeVars, ConsistencyAcrossOccurrences)
{
    // transpose(a⊗a) = a⊗a only matches when both legs are the same subtree.
    Context ctx;
    auto v = [&](char const* n)
    { return make_tensor_object(ctx, make_tensor_name(n), {}, 1); };
    auto const* a = v("a");
    Identity id{
        "sym",
        make_transpose(ctx, make_tensor_product(ctx, a, a)),
        make_tensor_product(ctx, a, a)};

    auto const* x = v("x");
    auto const* y = v("y");
    // Same leg: matches.
    EXPECT_TRUE(algebraic_eq(
        ctx,
        apply_identity(
            ctx, make_transpose(ctx, make_tensor_product(ctx, x, x)), id),
        make_tensor_product(ctx, x, x)));
    // Different legs: does not match (transpose(x⊗y) stays, modulo canonical).
    auto const* xy = make_transpose(ctx, make_tensor_product(ctx, x, y));
    EXPECT_TRUE(algebraic_eq(ctx, apply_identity(ctx, xy, id), xy));
}

TEST(SubtreeVars, RankCheckRejectsMismatch)
{
    // bac-cab — a×(b×c) = b(a·c) − c(a·b) — declares a,b,c as rank-1 subtree
    // variables.  The fenced chain a×(I×b) (identity I is rank 2) is
    // structurally a×(Y×Z), but the triple-product expansion is invalid there:
    // a rank-1 variable must not capture the rank-2 I.  So the identity must
    // NOT fire and the target is returned unchanged (vibe 000059 follow-up 1).
    Context ctx;
    auto v = [&](char const* n)
    { return make_tensor_object(ctx, make_tensor_name(n), {}, 1); };
    auto const* a = v("a");
    auto const* b = v("b");
    auto const* c = v("c");
    auto const* I = make_identity(ctx);
    Identity id{
        "bac-cab",
        make_cross(ctx, a, make_cross(ctx, b, c)),
        make_difference(
            ctx,
            make_tensor_product(ctx, b, make_dot(ctx, a, c)),
            make_tensor_product(ctx, c, make_dot(ctx, a, b)))};

    auto const* x = v("x");
    auto const* y = v("y");
    auto const* target = make_cross(ctx, x, make_cross(ctx, I, y));
    EXPECT_TRUE(algebraic_eq(ctx, apply_identity(ctx, target, id), target));
}

TEST(ApplyIdentity, SubtreeVarFiresUnderFloatedBinders)
{
    // The (a⊗b):(c⊗d) = (a·c)(b·d) identity fires on a target whose dyads carry
    // summation binders, once they are floated to the head (part 1 + part 2).
    Context ctx;
    auto v = [&](char const* n)
    { return make_tensor_object(ctx, make_tensor_name(n), {}, 1); };
    auto const* a = v("a");
    auto const* b = v("b");
    auto const* c = v("c");
    auto const* d = v("d");
    Identity id{
        "ddot",
        make_ddot(
            ctx, make_tensor_product(ctx, a, b), make_tensor_product(ctx, c, d)),
        make_tensor_product(ctx, make_dot(ctx, a, c), make_dot(ctx, b, d))};

    auto slot = [&](CountableIndex x)
    {
        return SlotBinding{
            IndexSlot{Level::Lower, Realm::Orthonormal, space_3d()},
            IndexAssoc{x}};
    };
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto evec = [&](CountableIndex x)
    { return make_tensor_object(ctx, make_tensor_name("e"), {slot(x)}, 1); };
    auto const* target = make_ddot(
        ctx,
        make_explicit_sum(ctx, i, make_tensor_product(ctx, evec(i), evec(i))),
        make_explicit_sum(ctx, j, make_tensor_product(ctx, evec(j), evec(j))));

    // After firing: Σ_i Σ_j (e_i·e_j)(e_i·e_j).  Build the same and compare.
    auto const* result = apply_identity(ctx, target, id);
    auto const* expected = make_explicit_sum(
        ctx,
        i,
        make_explicit_sum(
            ctx,
            j,
            make_tensor_product(
                ctx,
                make_dot(ctx, evec(i), evec(j)),
                make_dot(ctx, evec(i), evec(j)))));
    EXPECT_TRUE(algebraic_eq(ctx, result, expected));
}
