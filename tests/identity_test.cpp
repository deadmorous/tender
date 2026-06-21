#include <gtest/gtest.h>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/identity.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>

#include <optional>
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

// ---- match -----------------------------------------------------------------

TEST(Match, DeltaContractionBindsFreeIndices)
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

    // match expects canonical forms (apply_identity arranges this).
    auto bnd = match(
        steps::canonicalize(ctx, id.lhs), steps::canonicalize(ctx, target));
    ASSERT_TRUE(bnd.has_value());
}

TEST(Match, DistinctNodeKindsDoNotMatch)
{
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    // A bare delta is not an ExplicitSum, so the LHS pattern must not match.
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* target = delta_ul(ctx, sp, m, n);

    auto bnd = match(
        steps::canonicalize(ctx, id.lhs), steps::canonicalize(ctx, target));
    EXPECT_FALSE(bnd.has_value());
}

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
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    // Same contraction with the two delta factors written in the other order;
    // AC matching must still fire.
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

// ---- match: node-kind and negative-path coverage ---------------------------

namespace
{
auto obj(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n));
}

auto tslot(
    Level lvl,
    Realm realm,
    IndexSpace const* sp,
    std::optional<IndexAssoc> idx) -> SlotBinding
{
    return SlotBinding{IndexSlot{lvl, realm, sp}, idx};
}

auto tobj(Context& ctx, char const* n, std::vector<SlotBinding> slots)
    -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), std::move(slots));
}
} // namespace

TEST(MatchNodeKinds, InvariantBinaryOps)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");
    EXPECT_TRUE(match(make_dot(ctx, A, B), make_dot(ctx, A, B)).has_value());
    EXPECT_TRUE(match(make_ddot(ctx, A, B), make_ddot(ctx, A, B)).has_value());
    EXPECT_TRUE(
        match(make_ddot_alt(ctx, A, B), make_ddot_alt(ctx, A, B)).has_value());
    EXPECT_TRUE(
        match(make_cross(ctx, A, B), make_cross(ctx, A, B)).has_value());
    EXPECT_TRUE(match(make_difference(ctx, A, B), make_difference(ctx, A, B))
                    .has_value());
    EXPECT_TRUE(match(make_scalar_div(ctx, A, B), make_scalar_div(ctx, A, B))
                    .has_value());
    // A slot-less (invariant) product matches positionally, not modulo AC.
    EXPECT_TRUE(
        match(make_tensor_product(ctx, A, B), make_tensor_product(ctx, A, B))
            .has_value());
    // Mismatched node kinds never match.
    EXPECT_FALSE(match(make_dot(ctx, A, B), make_sum(ctx, A, B)).has_value());
    EXPECT_FALSE(
        match(make_tensor_product(ctx, A, B), make_sum(ctx, A, B)).has_value());
}

TEST(MatchNodeKinds, NegateScalarAndKindMismatch)
{
    Context ctx;
    auto const* A = obj(ctx, "A"); // a slot-less name → a subtree variable
    EXPECT_TRUE(match(make_negate(ctx, A), make_negate(ctx, A)).has_value());
    // The operand A is a subtree variable, so it binds B (vibe 000051).
    EXPECT_TRUE(
        match(make_negate(ctx, A), make_negate(ctx, obj(ctx, "B"))).has_value());
    EXPECT_TRUE(
        match(make_scalar(ctx, Rational{2}), make_scalar(ctx, Rational{2}))
            .has_value());
    EXPECT_FALSE(
        match(make_scalar(ctx, Rational{2}), make_scalar(ctx, Rational{3}))
            .has_value());
    // A *literal* (well-known) tensor pattern against a non-object target
    // fails.
    EXPECT_FALSE(
        match(make_identity(ctx), make_scalar(ctx, Rational{2})).has_value());
    // A subtree variable matches any subtree (including a scalar).
    EXPECT_TRUE(match(A, make_scalar(ctx, Rational{2})).has_value());
}

TEST(MatchNodeKinds, SumCommutativeAndBacktrack)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");
    auto const* C = obj(ctx, "C");
    // Order-independent (AC) match.
    EXPECT_TRUE(match(make_sum(ctx, A, B), make_sum(ctx, B, A)).has_value());
    // Different addend counts.
    EXPECT_FALSE(
        match(make_sum(ctx, A, B), make_sum(ctx, A, make_sum(ctx, B, C)))
            .has_value());
    // Backtracking that ultimately fails: [A, A] against [A, B].
    EXPECT_FALSE(match(make_sum(ctx, A, A), make_sum(ctx, A, B)).has_value());
    // A Sum pattern against a non-Sum target.
    EXPECT_FALSE(match(make_sum(ctx, A, B), A).has_value());
}

TEST(MatchNodeKinds, SlotDescriptorMismatches)
{
    Context ctx;
    auto const* sp3 = space_3d();
    auto const* sp2 = space_2d();
    CountableIndex i{ctx.alloc_index_id()};
    auto t = [&](Realm r, Level l, IndexSpace const* sp)
    { return tobj(ctx, "T", {tslot(l, r, sp, i)}); };
    auto const* base = t(Realm::Oblique, Level::Upper, sp3);
    EXPECT_FALSE(match(base, t(Realm::Oblique, Level::Lower, sp3)).has_value());
    EXPECT_FALSE(
        match(base, t(Realm::Orthonormal, Level::Upper, sp3)).has_value());
    EXPECT_FALSE(match(base, t(Realm::Oblique, Level::Upper, sp2)).has_value());
}

TEST(MatchNodeKinds, VoidSlots)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    auto const* voidpat =
        tobj(ctx, "T", {tslot(Level::Upper, Realm::Oblique, sp, std::nullopt)});
    auto const* filled =
        tobj(ctx, "T", {tslot(Level::Upper, Realm::Oblique, sp, i)});
    // A void pattern slot does not match a filled target slot.
    EXPECT_FALSE(match(voidpat, filled).has_value());
    // Two void slots match (nothing to bind).
    EXPECT_TRUE(
        match(
            voidpat,
            tobj(
                ctx,
                "T",
                {tslot(Level::Upper, Realm::Oblique, sp, std::nullopt)}))
            .has_value());
}

TEST(MatchNodeKinds, ConcreteAndLabelIndices)
{
    Context ctx;
    auto const* sp = space_3d();
    auto c = [&](int v)
    {
        return tobj(
            ctx,
            "T",
            {tslot(Level::Upper, Realm::Orthonormal, sp, ConcreteIndex{v})});
    };
    EXPECT_TRUE(match(c(1), c(1)).has_value());  // concrete equal
    EXPECT_FALSE(match(c(1), c(2)).has_value()); // concrete differ
    auto l = [&](char const* nm)
    {
        return tobj(
            ctx,
            "T",
            {tslot(
                Level::Lower,
                Realm::Label,
                nullptr,
                LabelIndex{make_index_name(nm)})});
    };
    EXPECT_TRUE(match(l("vol"), l("vol")).has_value());   // label equal
    EXPECT_FALSE(match(l("vol"), l("surf")).has_value()); // label differ
}

TEST(MatchNodeKinds, ExplicitSumBoundAndNoSum)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex q{ctx.alloc_index_id()};
    auto const* n = make_scalar(ctx, Rational{3});
    // Symbolic bound on one side only: presence mismatch.
    EXPECT_FALSE(match(
                     make_explicit_sum(ctx, p, A, n),
                     make_explicit_sum(ctx, q, A, nullptr))
                     .has_value());
    // Both symbolic-bound: the bound expressions are matched too.
    EXPECT_TRUE(match(
                    make_explicit_sum(ctx, p, A, n),
                    make_explicit_sum(ctx, q, A, make_scalar(ctx, Rational{3})))
                    .has_value());
    // Both symbolic-bound but the bounds differ: no match.
    EXPECT_FALSE(
        match(
            make_explicit_sum(ctx, p, A, n),
            make_explicit_sum(ctx, q, A, make_scalar(ctx, Rational{4})))
            .has_value());
    // Reusing one pattern binder id across nested sums fails on distinct
    // target binders (try_bind consistency).
    CountableIndex r{ctx.alloc_index_id()};
    EXPECT_FALSE(match(
                     make_explicit_sum(ctx, p, make_explicit_sum(ctx, p, A)),
                     make_explicit_sum(ctx, q, make_explicit_sum(ctx, r, A)))
                     .has_value());
    // NoSum binds its index and matches the body.
    EXPECT_TRUE(
        match(make_no_sum(ctx, p, A), make_no_sum(ctx, q, A)).has_value());
    // The body A is a subtree variable, so it binds B (vibe 000051).
    EXPECT_TRUE(
        match(make_no_sum(ctx, p, A), make_no_sum(ctx, q, obj(ctx, "B")))
            .has_value());
    // NoSum pattern against a non-NoSum target.
    EXPECT_FALSE(match(make_no_sum(ctx, p, A), A).has_value());
    // Reusing one NoSum binder id across nesting fails on distinct targets.
    EXPECT_FALSE(match(
                     make_no_sum(ctx, p, make_no_sum(ctx, p, A)),
                     make_no_sum(ctx, q, make_no_sum(ctx, r, A)))
                     .has_value());
}

// ---- instantiate: substitution across node kinds ---------------------------

TEST(Instantiate, SubstitutesIndicesAcrossNodeKinds)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex r{ctx.alloc_index_id()};
    auto const* A = obj(ctx, "A");

    MatchBinding bnd;
    bnd.indices.emplace_back(p.id, IndexAssoc{q});

    // TensorObject slots: δ^p_p → δ^q_q.
    auto const* dpp =
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, p, p);
    auto const* dqq =
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, q, q);
    EXPECT_TRUE(structural_eq(instantiate(ctx, dpp, bnd), dqq));

    // A tensor with no bound index is returned unchanged.
    auto const* drr =
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, r, r);
    EXPECT_TRUE(structural_eq(instantiate(ctx, drr, bnd), drr));

    // Void and non-countable (concrete) slots are skipped during substitution.
    auto const* mixed = tobj(
        ctx,
        "T",
        {tslot(Level::Upper, Realm::Oblique, sp, std::nullopt),
         tslot(Level::Lower, Realm::Orthonormal, sp, ConcreteIndex{1})});
    EXPECT_TRUE(structural_eq(instantiate(ctx, mixed, bnd), mixed));

    // ExplicitSum binder substitution and the unchanged paths.
    auto const* esi = instantiate(ctx, make_explicit_sum(ctx, p, A), bnd);
    auto const* es = std::get_if<ExplicitSum>(&esi->node);
    ASSERT_NE(es, nullptr);
    EXPECT_EQ(es->index.id, q.id);
    EXPECT_TRUE(structural_eq(
        instantiate(ctx, make_explicit_sum(ctx, r, A), bnd),
        make_explicit_sum(ctx, r, A))); // binder absent from binding

    // NoSum binder substitution and the unchanged path.
    auto const* nsi = instantiate(ctx, make_no_sum(ctx, p, A), bnd);
    auto const* ns = std::get_if<NoSum>(&nsi->node);
    ASSERT_NE(ns, nullptr);
    EXPECT_EQ(ns->index.id, q.id);
    EXPECT_TRUE(structural_eq(
        instantiate(ctx, make_no_sum(ctx, r, A), bnd), make_no_sum(ctx, r, A)));
}

TEST(Instantiate, SelfMappingLeavesBindersUnchanged)
{
    Context ctx;
    CountableIndex p{ctx.alloc_index_id()};
    auto const* A = obj(ctx, "A");
    MatchBinding bnd;
    bnd.indices.emplace_back(p.id, IndexAssoc{p}); // p → p

    EXPECT_TRUE(structural_eq(
        instantiate(ctx, make_explicit_sum(ctx, p, A), bnd),
        make_explicit_sum(ctx, p, A)));
    EXPECT_TRUE(structural_eq(
        instantiate(ctx, make_no_sum(ctx, p, A), bnd), make_no_sum(ctx, p, A)));
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
    // A rank-1 subtree variable does not match a rank-2 subtree.
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);
    EXPECT_FALSE(match(a, A).has_value());
    EXPECT_TRUE(match(a, make_tensor_object(ctx, make_tensor_name("b"), {}, 1))
                    .has_value());
}

TEST(SubtreeVars, WellKnownStaysLiteral)
{
    // The identity tensor in an LHS is a literal, not a variable.
    Context ctx;
    auto const* I = make_identity(ctx);
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);
    EXPECT_TRUE(match(I, make_identity(ctx)).has_value());
    EXPECT_FALSE(match(I, A).has_value());
}
