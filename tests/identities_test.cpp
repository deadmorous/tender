#include <gtest/gtest.h>
#include <tender/context.hpp>
#include <tender/derivation.hpp> // contract_eps_pair / contract_delta oracles
#include <tender/engine.hpp>
#include <tender/expr.hpp>
#include <tender/identities.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>
#include <tender/nf_egraph.hpp>
#include <tender/nf_lower.hpp> // nf::raise

#include <string>

using namespace tender;

namespace
{
constexpr Level U = Level::Upper;
constexpr Level L = Level::Lower;

auto eps(
    Context& ctx,
    Level lvl,
    CountableIndex x,
    CountableIndex y,
    CountableIndex z) -> Expr const*
{
    return make_levi_civita(
        ctx,
        Realm::Oblique,
        space_3d(),
        {lvl, lvl, lvl},
        {IndexAssoc{x}, IndexAssoc{y}, IndexAssoc{z}});
}

auto d(Context& ctx, Level la, Level lb, CountableIndex a, CountableIndex b)
    -> Expr const*
{
    return make_delta(ctx, Realm::Oblique, space_3d(), la, lb, a, b);
}

// Saturate `target` with one identity and return the extracted result, raised
// back to a surface `Expr`.
auto run(Context& ctx, Expr const* target, Identity rule) -> Expr const*
{
    nf::NfEGraph eg{ctx};
    auto const root = eg.add(target);
    eg.saturate({std::move(rule)});
    return nf::raise(ctx, *eg.extract(eg.find(root)));
}
} // namespace

TEST(Identities, DeltaContraction)
{
    Context ctx;
    auto const* sp = space_3d();
    auto const q = CountableIndex{ctx.alloc_index_id()};
    auto const m = CountableIndex{ctx.alloc_index_id()};
    auto const n = CountableIndex{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        q,
        make_tensor_product(ctx, d(ctx, U, L, q, m), d(ctx, U, L, q, n)));

    auto const* result =
        run(ctx,
            target,
            identities::delta_contraction(ctx, sp, Realm::Oblique));
    EXPECT_TRUE(algebraic_eq(ctx, result, d(ctx, L, L, m, n)));
}

TEST(Identities, DeltaTrace)
{
    Context ctx;
    auto const* sp = space_3d();
    auto const q = CountableIndex{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(ctx, q, d(ctx, U, L, q, q)); // Σ_q
                                                                        // δ^q_q

    auto const* result =
        run(ctx, target, identities::delta_trace(ctx, sp, Realm::Oblique));
    EXPECT_TRUE(algebraic_eq(ctx, result, make_scalar(ctx, Rational{3})));
}

TEST(Identities, EpsDelta1MatchesOracle)
{
    // The δ-expansion is larger than the ε-form, so this only extracts
    // correctly because the cost function weights Levi-Civita symbols heavily.
    Context ctx;
    auto const a = CountableIndex{ctx.alloc_index_id()};
    auto const b = CountableIndex{ctx.alloc_index_id()};
    auto const c = CountableIndex{ctx.alloc_index_id()};
    auto const dd = CountableIndex{ctx.alloc_index_id()};
    auto const e = CountableIndex{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        a,
        make_tensor_product(ctx, eps(ctx, U, a, b, c), eps(ctx, L, a, dd, e)));

    auto const* oracle = steps::contract_eps_pair(ctx, target);
    auto const* result =
        run(ctx, target, identities::eps_delta_1(ctx, Realm::Oblique));
    EXPECT_TRUE(algebraic_eq(ctx, result, oracle));
}

TEST(Identities, RealmOrthonormalContraction)
{
    // The builder is realm-parameterized (vibe 000047 (a)): an Orthonormal rule
    // contracts an Orthonormal target.
    Context ctx;
    auto const* sp = space_3d();
    auto const q = CountableIndex{ctx.alloc_index_id()};
    auto const m = CountableIndex{ctx.alloc_index_id()};
    auto const n = CountableIndex{ctx.alloc_index_id()};
    // Orthonormal indices are spelled lower by convention (vibe 000047), and
    // the rule produces lower-lower deltas to match.
    auto dn = [&](Level la, Level lb, CountableIndex a, CountableIndex b)
    { return make_delta(ctx, Realm::Orthonormal, sp, la, lb, a, b); };
    auto const* target = make_explicit_sum(
        ctx, q, make_tensor_product(ctx, dn(L, L, q, m), dn(L, L, q, n)));

    auto const* result =
        run(ctx,
            target,
            identities::delta_contraction(ctx, sp, Realm::Orthonormal));
    EXPECT_TRUE(algebraic_eq(ctx, result, dn(L, L, m, n)));
}

TEST(Identities, OrthonormalRuleIsLowerSpelled)
{
    // The Orthonormal rule is lower-lower, so it does NOT fire on an
    // upper-spelled Orthonormal target — pinning the lower-index convention.
    Context ctx;
    auto const* sp = space_3d();
    auto const q = CountableIndex{ctx.alloc_index_id()};
    auto const m = CountableIndex{ctx.alloc_index_id()};
    auto const n = CountableIndex{ctx.alloc_index_id()};
    auto dn = [&](Level la, Level lb, CountableIndex a, CountableIndex b)
    { return make_delta(ctx, Realm::Orthonormal, sp, la, lb, a, b); };
    auto const* upper_target = make_explicit_sum(
        ctx, q, make_tensor_product(ctx, dn(U, L, q, m), dn(U, L, q, n)));

    auto const* result =
        run(ctx,
            upper_target,
            identities::delta_contraction(ctx, sp, Realm::Orthonormal));
    EXPECT_TRUE(algebraic_eq(ctx, result, upper_target)); // unchanged
    EXPECT_FALSE(algebraic_eq(ctx, result, dn(L, L, m, n)));
}

TEST(Identities, RealmMismatchDoesNotFire)
{
    // A rule built in the wrong realm must not match (match_slot is
    // realm-exact) — the target comes back unchanged, not contracted.
    Context ctx;
    auto const* sp = space_3d();
    auto const q = CountableIndex{ctx.alloc_index_id()};
    auto const m = CountableIndex{ctx.alloc_index_id()};
    auto const n = CountableIndex{ctx.alloc_index_id()};
    auto dn = [&](Level la, Level lb, CountableIndex a, CountableIndex b)
    { return make_delta(ctx, Realm::Orthonormal, sp, la, lb, a, b); };
    auto const* target = make_explicit_sum(
        ctx, q, make_tensor_product(ctx, dn(L, L, q, m), dn(L, L, q, n)));

    // Oblique rule against an Orthonormal target.
    auto const* result =
        run(ctx,
            target,
            identities::delta_contraction(ctx, sp, Realm::Oblique));
    EXPECT_FALSE(algebraic_eq(ctx, result, dn(L, L, m, n)));
    EXPECT_TRUE(algebraic_eq(ctx, result, target));
}

TEST(Identities, EpsDelta2MatchesOracle)
{
    Context ctx;
    auto const a = CountableIndex{ctx.alloc_index_id()};
    auto const b = CountableIndex{ctx.alloc_index_id()};
    auto const c = CountableIndex{ctx.alloc_index_id()};
    auto const dd = CountableIndex{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        a,
        make_explicit_sum(
            ctx,
            b,
            make_tensor_product(
                ctx, eps(ctx, U, a, b, c), eps(ctx, L, a, b, dd))));

    auto const* oracle = steps::contract_eps_pair(ctx, target);
    auto const* result =
        run(ctx, target, identities::eps_delta_2(ctx, Realm::Oblique));
    EXPECT_TRUE(algebraic_eq(ctx, result, oracle));
}

// ---- rule library: fire-tests at birth (vibe 000096 increment 2) --------
//
// Every library rule must *fire* on a minimal target.  This is not
// ceremony: canon α-renames dummies, normalizes symmetries, and sorts
// symmetric contraction chains by tensor name, any of which can silently
// make a correctly-stated rule unmatchable (vibe 000040).  A rule that
// cannot fire is worse than no rule — it looks like coverage and is inert.

namespace
{

using namespace tender::engine;

auto v1(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 1);
}
auto v2(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 2);
}

// Assert `lhs == rhs` is provable by `rules`, and that some rule fired.
void expect_proves(
    Context& ctx,
    Expr const* lhs,
    Expr const* rhs,
    std::vector<Identity> const& rules,
    char const* what)
{
    auto const res = prove_equal(ctx, lhs, rhs, rules);
    EXPECT_TRUE(res.proved()) << what << ": not proved (status "
                              << static_cast<int>(res.status) << ")";
    int total = 0;
    for (int f: res.report.fired)
        total += f;
    EXPECT_GT(total, 0) << what
                        << ": proved without firing any rule — the "
                           "rule under test is not what did the work";
}

} // namespace

TEST(RuleLibrary, BacCabFires)
{
    Context ctx;
    auto const* a = v1(ctx, "a");
    auto const* b = v1(ctx, "b");
    auto const* c = v1(ctx, "c");
    expect_proves(
        ctx,
        make_cross(ctx, a, make_cross(ctx, b, c)),
        make_difference(
            ctx,
            make_tensor_product(ctx, b, make_dot(ctx, a, c)),
            make_tensor_product(ctx, c, make_dot(ctx, a, b))),
        {identities::bac_cab(ctx)},
        "bac-cab");
}

TEST(RuleLibrary, BacCabDoesNotFireAcrossARankTwoFence)
{
    // Soundness guard: with a rank-2 middle operand the crosses reassociate
    // around the fence (vibe 000055) and the triple-product identity does NOT
    // hold — a subtree variable binds any factor regardless of rank, so this
    // is the test that the fence, not luck, keeps the rule honest.
    Context ctx;
    auto const* a = v1(ctx, "a");
    auto const* B = v2(ctx, "B");
    auto const* c = v1(ctx, "c");
    auto const* wrong = make_difference(
        ctx,
        make_tensor_product(ctx, B, make_dot(ctx, a, c)),
        make_tensor_product(ctx, c, make_dot(ctx, a, B)));

    auto const res = prove_equal(
        ctx,
        make_cross(ctx, a, make_cross(ctx, B, c)),
        wrong,
        {identities::bac_cab(ctx)});
    EXPECT_FALSE(res.proved())
        << "bac-cab must not apply across a rank-2 fence";
}

TEST(RuleLibrary, CrossIdentityFires)
{
    Context ctx;
    auto const* a = v1(ctx, "a");
    auto const* I = make_identity(ctx);
    expect_proves(
        ctx,
        make_cross(ctx, a, I),
        make_cross(ctx, I, a),
        {identities::cross_identity(ctx)},
        "a×I = I×a");
}

TEST(RuleLibrary, CrossRemovalFires)
{
    // THE vibe-000056 case: a×(b×I) = b⊗a − (a·b)I.
    Context ctx;
    auto const* a = v1(ctx, "a");
    auto const* b = v1(ctx, "b");
    auto const* I = make_identity(ctx);
    expect_proves(
        ctx,
        make_cross(ctx, a, make_cross(ctx, b, I)),
        make_difference(
            ctx,
            make_tensor_product(ctx, b, a),
            make_tensor_product(ctx, make_dot(ctx, a, b), I)),
        {identities::cross_removal(ctx)},
        "cross removal");
}

TEST(RuleLibrary, LagrangeFires)
{
    Context ctx;
    auto const* a = v1(ctx, "a");
    auto const* b = v1(ctx, "b");
    auto const* c = v1(ctx, "c");
    auto const* d = v1(ctx, "d");
    expect_proves(
        ctx,
        make_dot(ctx, make_cross(ctx, a, b), make_cross(ctx, c, d)),
        make_difference(
            ctx,
            make_tensor_product(ctx, make_dot(ctx, a, c), make_dot(ctx, b, d)),
            make_tensor_product(ctx, make_dot(ctx, a, d), make_dot(ctx, b, c))),
        {identities::lagrange(ctx)},
        "Lagrange identity");
}

TEST(RuleLibrary, TraceCyclicFires)
{
    Context ctx;
    auto const* A = v2(ctx, "A");
    auto const* B = v2(ctx, "B");
    expect_proves(
        ctx,
        make_trace(ctx, make_dot(ctx, A, B)),
        make_trace(ctx, make_dot(ctx, B, A)),
        {identities::trace_cyclic(ctx)},
        "tr(A·B) = tr(B·A)");
}

TEST(RuleLibrary, IdentityDotFires)
{
    Context ctx;
    auto const* a = v1(ctx, "a");
    expect_proves(
        ctx,
        make_dot(ctx, make_identity(ctx), a),
        a,
        {identities::identity_dot(ctx)},
        "I·a = a");
}

TEST(RuleLibrary, RulesAreNameRobustAcrossTheAlphabet)
{
    // Canon sorts symmetric contraction chains by tensor name, so a rule can
    // fire for targets named one way and silently miss others.  Every library
    // rule must be insensitive to the target's naming — checked over a spread
    // of letters from both ends of the alphabet.
    for (auto const* nm: {"a", "f", "p", "x"})
    {
        Context ctx;
        std::string const n1 = nm;
        std::string const n2 = n1 == "a" ? "b" : "a";
        std::string const n3 = n1 == "c" ? "d" : "c";
        auto const* a = v1(ctx, n1.c_str());
        auto const* b = v1(ctx, n2.c_str());
        auto const* c = v1(ctx, n3.c_str());

        expect_proves(
            ctx,
            make_cross(ctx, a, make_cross(ctx, b, c)),
            make_difference(
                ctx,
                make_tensor_product(ctx, b, make_dot(ctx, a, c)),
                make_tensor_product(ctx, c, make_dot(ctx, a, b))),
            {identities::bac_cab(ctx)},
            "bac-cab (name sweep)");

        auto const* I = make_identity(ctx);
        expect_proves(
            ctx,
            make_cross(ctx, a, make_cross(ctx, b, I)),
            make_difference(
                ctx,
                make_tensor_product(ctx, b, a),
                make_tensor_product(ctx, make_dot(ctx, a, b), I)),
            {identities::cross_removal(ctx)},
            "cross removal (name sweep)");
    }
}

// ---- groups -------------------------------------------------------------

TEST(RuleLibrary, GroupsAreSelectableAndPopulated)
{
    Context ctx;
    EXPECT_EQ(identities::group_names().size(), 3u);
    EXPECT_EQ(identities::group(ctx, "eps_delta").size(), 4u);
    EXPECT_EQ(identities::group(ctx, "cross").size(), 4u);
    EXPECT_EQ(identities::group(ctx, "dyadic").size(), 2u);
    EXPECT_EQ(identities::all_rules(ctx).size(), 10u);
}

TEST(RuleLibrary, UnknownGroupThrows)
{
    Context ctx;
    EXPECT_THROW(
        (void)identities::group(ctx, "no_such_group"), std::invalid_argument);
}

TEST(RuleLibrary, EveryRuleCompilesForTheEngine)
{
    // A rule the engine cannot compile (multi-term LHS) never fires.  The
    // library must contain none: `skipped` must come back empty.
    Context ctx;
    auto const* a = v1(ctx, "a");
    auto const res = simplify(ctx, a, identities::all_rules(ctx));
    EXPECT_TRUE(res.report.skipped.empty())
        << "library contains " << res.report.skipped.size()
        << " rule(s) the engine silently cannot use";
}
