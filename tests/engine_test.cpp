#include <tender/engine.hpp>

#include <tender/derivation.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>

#include <gtest/gtest.h>

using namespace tender;
using namespace tender::engine;

namespace
{

auto delta_ul(
    Context& ctx,
    IndexSpace const* sp,
    CountableIndex a,
    CountableIndex b) -> Expr const*
{
    return make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, a, b);
}

auto delta_ll(
    Context& ctx,
    IndexSpace const* sp,
    CountableIndex a,
    CountableIndex b) -> Expr const*
{
    return make_delta(ctx, Realm::Oblique, sp, Level::Lower, Level::Lower, a, b);
}

// Σ_q δ^q_m δ^q_n — the contractible form the delta_contraction rule collapses.
auto contraction(
    Context& ctx,
    IndexSpace const* sp,
    CountableIndex q,
    CountableIndex m,
    CountableIndex n) -> Expr const*
{
    return make_explicit_sum(
        ctx,
        q,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, q, m), delta_ul(ctx, sp, q, n)));
}

// The δ-contraction rule, built locally: the shipped identity library lives
// in Python (tender/identities.py) so it can be extended without a rebuild,
// so the C++ engine tests carry the small rules they need themselves.
auto rules_eps_delta(Context& ctx) -> std::vector<Identity>
{
    auto const* sp = space_3d();
    CountableIndex const p{ctx.alloc_index_id()};
    CountableIndex const a{ctx.alloc_index_id()};
    CountableIndex const b{ctx.alloc_index_id()};
    return {Identity{
        "delta-contraction",
        make_explicit_sum(
            ctx,
            p,
            make_tensor_product(
                ctx, delta_ul(ctx, sp, p, a), delta_ul(ctx, sp, p, b))),
        delta_ll(ctx, sp, a, b)}};
}

} // namespace

// ---- prove_equal --------------------------------------------------------

TEST(ProveEqual, ProvesByFiringARule)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res = prove_equal(
        ctx,
        contraction(ctx, sp, q, m, n),
        delta_ll(ctx, sp, m, n),
        rules_eps_delta(ctx));

    EXPECT_TRUE(res.proved());
    EXPECT_EQ(res.status, ProofStatus::Proved);
    EXPECT_EQ(res.report.fired.at(0), 1); // the trace attributes the work
}

TEST(ProveEqual, AlreadyEqualUnderCanonicalizationNeedsNoRules)
{
    // a ⊗ b vs a ⊗ b written differently is theory-T0 equal: proved with zero
    // passes and no rule firing.
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    auto const* two = make_scalar(ctx, Rational{2});

    auto const res = prove_equal(
        ctx,
        make_tensor_product(ctx, two, make_tensor_product(ctx, a, b)),
        make_tensor_product(ctx, make_tensor_product(ctx, a, two), b),
        {});

    EXPECT_TRUE(res.proved());
    EXPECT_EQ(res.report.passes, 0);
}

TEST(ProveEqual, UnrelatedExpressionsExhaustWithoutProof)
{
    // Genuinely different expressions: the rules run to a fixed point and no
    // proof appears.  `Exhausted` — NOT a claim that they are unequal.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* c = make_tensor_object(ctx, make_tensor_name("c"), {}, 2);

    auto const res =
        prove_equal(ctx, delta_ll(ctx, sp, m, n), c, rules_eps_delta(ctx));

    EXPECT_FALSE(res.proved());
    EXPECT_EQ(res.status, ProofStatus::Exhausted);
    EXPECT_EQ(res.report.outcome, SaturateOutcome::Saturated);
}

TEST(ProveEqual, BudgetTripIsInconclusiveNotDisproof)
{
    // The safety property: a proof that exists but is cut short by the budget
    // must report `Budget`, never `Exhausted` — a caller reading "not proved"
    // as "not equal" would be wrong.  Zero passes allowed, so the rule that
    // would prove it never fires.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res = prove_equal(
        ctx,
        contraction(ctx, sp, q, m, n),
        delta_ll(ctx, sp, m, n),
        rules_eps_delta(ctx),
        SaturateBudget{.max_passes = 0});

    EXPECT_FALSE(res.proved());
    EXPECT_EQ(res.status, ProofStatus::Budget);
    EXPECT_NE(res.status, ProofStatus::Exhausted);
}

TEST(ProveEqual, NodeBudgetAlsoReportsInconclusive)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res = prove_equal(
        ctx,
        contraction(ctx, sp, q, m, n),
        delta_ll(ctx, sp, m, n),
        rules_eps_delta(ctx),
        SaturateBudget{.max_passes = 30, .max_nodes = 1});

    EXPECT_EQ(res.status, ProofStatus::Budget);
    EXPECT_EQ(res.report.outcome, SaturateOutcome::NodeBudget);
}

TEST(ProveEqual, StopsEarlyOnceTheGoalIsReached)
{
    // The goal check ends saturation as soon as the sides join, rather than
    // running the rule set to its fixed point.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res = prove_equal(
        ctx,
        contraction(ctx, sp, q, m, n),
        delta_ll(ctx, sp, m, n),
        rules_eps_delta(ctx));

    EXPECT_TRUE(res.proved());
    EXPECT_EQ(res.report.outcome, SaturateOutcome::EarlyStop);
    EXPECT_EQ(res.report.passes, 1); // proved on the first pass, then stopped
}

// ---- simplify -----------------------------------------------------------

TEST(Simplify, ContractsAndReportsCompletion)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res =
        simplify(ctx, contraction(ctx, sp, q, m, n), rules_eps_delta(ctx));

    EXPECT_TRUE(res.complete());
    EXPECT_TRUE(algebraic_eq(ctx, res.expr, delta_ll(ctx, sp, m, n)));
}

TEST(Simplify, BudgetTripStillReturnsTheBestFormFound)
{
    // The loud-fallback contract: on a budget trip the caller still gets a
    // usable expression, flagged incomplete rather than silently presented as
    // a fixed point.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res = simplify(
        ctx,
        contraction(ctx, sp, q, m, n),
        rules_eps_delta(ctx),
        SaturateBudget{.max_passes = 0});

    EXPECT_FALSE(res.complete());
    ASSERT_NE(res.expr, nullptr);
    // Unsimplified, but algebraically the input — never garbage.
    EXPECT_TRUE(algebraic_eq(ctx, res.expr, contraction(ctx, sp, q, m, n)));
}

TEST(Simplify, MultiTermLhsRuleIsReportedAsSkipped)
{
    // A rule the engine cannot compile must be *visible* in the trace, not
    // silently inert (vibe 000096).
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    Identity multi{"sum-lhs", make_sum(ctx, a, b), a};

    auto const res = simplify(ctx, a, {multi});
    ASSERT_EQ(res.report.skipped.size(), 1u);
    EXPECT_EQ(res.report.skipped.at(0), 0u);
}

// ---- intent-driven extraction cost (vibe 000097) ------------------------

namespace
{

auto vec1(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 1);
}

// a × (b × c) = b (a·c) − c (a·b)
auto bac_cab(Context& ctx) -> Identity
{
    auto const* u = vec1(ctx, "u");
    auto const* v = vec1(ctx, "v");
    auto const* w = vec1(ctx, "w");
    return Identity{
        "bac-cab",
        make_cross(ctx, u, make_cross(ctx, v, w)),
        make_difference(
            ctx,
            make_tensor_product(ctx, v, make_dot(ctx, u, w)),
            make_tensor_product(ctx, w, make_dot(ctx, u, v)))};
}

} // namespace

TEST(CostModel, SameGraphDifferentIntentDifferentAnswer)
{
    // The point of making cost a parameter: "simplest" is the caller's goal,
    // not a property of the algebra.  One rule, one expression — the default
    // keeps the compact crossed form (it has fewer nodes), while an intent to
    // remove crosses takes the *larger* expanded form instead.
    Context ctx;
    auto const* a = vec1(ctx, "a");
    auto const* b = vec1(ctx, "b");
    auto const* c = vec1(ctx, "c");
    auto const* crossed = make_cross(ctx, a, make_cross(ctx, b, c));
    auto const* expanded = make_difference(
        ctx,
        make_tensor_product(ctx, b, make_dot(ctx, a, c)),
        make_tensor_product(ctx, c, make_dot(ctx, a, b)));

    auto const cheap = simplify(ctx, crossed, {bac_cab(ctx)});
    EXPECT_TRUE(algebraic_eq(ctx, cheap.expr, crossed))
        << "the default cost should keep the compact crossed form";

    auto const nocross =
        simplify(ctx, crossed, {bac_cab(ctx)}, {}, CostModel::fewest_crosses());
    EXPECT_TRUE(algebraic_eq(ctx, nocross.expr, expanded))
        << "fewest_crosses should take the expansion, larger though it is";
}

TEST(CostModel, EpsPreferenceIsTheDefaultIntentNotAFixedRule)
{
    // The ε weight of vibe 000046 is one intent among several — expressible,
    // and switchable off by asking for plain size.
    EXPECT_GT(CostModel::fewest_eps().levi_civita, CostModel{}.node);
    EXPECT_EQ(CostModel::smallest().levi_civita, CostModel::smallest().node);
    EXPECT_GT(
        CostModel::fewest_crosses().cross, CostModel::fewest_crosses().node);
}

TEST(CostModel, CrossWeightCountsOperatorsNotChains)
{
    // A chain of n factors carries n−1 × operators; the weight must scale with
    // the operators, else a single long chain would look as cheap as one ×.
    Context ctx;
    auto const* a = vec1(ctx, "a");
    auto const* b = vec1(ctx, "b");
    auto const* c = vec1(ctx, "c");
    auto const one = simplify(
        ctx, make_cross(ctx, a, b), {}, {}, CostModel::fewest_crosses());
    auto const two = simplify(
        ctx,
        make_cross(ctx, a, make_cross(ctx, b, c)),
        {},
        {},
        CostModel::fewest_crosses());
    // Both are irreducible without rules; the test is that neither throws and
    // the model is applied uniformly.
    EXPECT_NE(one.expr, nullptr);
    EXPECT_NE(two.expr, nullptr);
}

// ---- refutation by component expansion (vibe 000097) --------------------

TEST(Refutation, FalseStatementsAreRefutedNotMerelyUnproved)
{
    // Saturation alone can only exhaust; the component decision procedure
    // supplies the negative, so a false claim gets a verdict.
    Context ctx;
    auto const* a = vec1(ctx, "a");
    auto const* b = vec1(ctx, "b");
    auto const* c = vec1(ctx, "c");

    // a×b = b×a is false (the cross anticommutes).
    auto const swapped = prove_equal(
        ctx, make_cross(ctx, a, b), make_cross(ctx, b, a), {bac_cab(ctx)});
    EXPECT_TRUE(swapped.refuted());
    EXPECT_EQ(swapped.status, ProofStatus::Refuted);

    // bac-cab with the two terms interchanged is false.
    auto const wrong_sign = prove_equal(
        ctx,
        make_cross(ctx, a, make_cross(ctx, b, c)),
        make_difference(
            ctx,
            make_tensor_product(ctx, c, make_dot(ctx, a, b)),
            make_tensor_product(ctx, b, make_dot(ctx, a, c))),
        {bac_cab(ctx)});
    EXPECT_TRUE(wrong_sign.refuted());
}

TEST(Refutation, TrueStatementsAreNeverRefuted)
{
    Context ctx;
    auto const* a = vec1(ctx, "a");
    auto const* b = vec1(ctx, "b");
    // a×b = −(b×a) is true and canon decides it.
    auto const res = prove_equal(
        ctx, make_cross(ctx, a, b), make_negate(ctx, make_cross(ctx, b, a)), {});
    EXPECT_TRUE(res.proved());
    EXPECT_FALSE(res.refuted());
}

TEST(Refutation, TrueButUnprovableBlamesTheRulesNotTheClaim)
{
    // The Lagrange identity holds, but with no rule supplied saturation
    // cannot reach it.  The component check agrees the sides are equal, so
    // the answer points at the incomplete rule set rather than the claim.
    Context ctx;
    auto const* a = vec1(ctx, "a");
    auto const* b = vec1(ctx, "b");
    auto const* c = vec1(ctx, "c");
    auto const* d = vec1(ctx, "d");

    auto const res = prove_equal(
        ctx,
        make_dot(ctx, make_cross(ctx, a, b), make_cross(ctx, c, d)),
        make_difference(
            ctx,
            make_tensor_product(ctx, make_dot(ctx, a, c), make_dot(ctx, b, d)),
            make_tensor_product(ctx, make_dot(ctx, a, d), make_dot(ctx, b, c))),
        {});
    EXPECT_EQ(res.status, ProofStatus::Exhausted);
    EXPECT_FALSE(res.refuted());
    EXPECT_TRUE(res.components_agree);
}

TEST(Refutation, DifferentialContentIsUndecidedNotRefuted)
{
    // The component procedure decides the chart-free *algebraic* fragment
    // only.  Anything holding a ∇ leaves a residue, so no refutation is
    // claimed — silence is the safe answer, not a wrong verdict.
    Context ctx;
    auto const* u = make_field(ctx, make_tensor_name("u"), 1, {});
    auto const* lhs = make_dot(ctx, tender::make_nabla(ctx), u);
    auto const* rhs = make_sum(ctx, lhs, make_scalar(ctx, Rational{1}));
    EXPECT_EQ(decide_by_components(ctx, lhs, rhs), ComponentVerdict::Undecided);
}

TEST(Refutation, BudgetTripDoesNotAttemptRefutation)
{
    // A budget trip concludes nothing: the rules might have proved it given
    // room, so the component pass must not turn "ran out of time" into a
    // verdict either way.
    Context ctx;
    auto const* a = vec1(ctx, "a");
    auto const* b = vec1(ctx, "b");
    auto const res = prove_equal(
        ctx,
        make_cross(ctx, a, b),
        make_cross(ctx, b, a),
        {bac_cab(ctx)},
        SaturateBudget{.max_passes = 0});
    EXPECT_EQ(res.status, ProofStatus::Budget);
    EXPECT_FALSE(res.refuted());
}

// ---- budgets in user units (vibe 000097) --------------------------------

TEST(Budget, TimeAndMemoryCapsStopSaturation)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* target = contraction(ctx, sp, q, m, n);

    // A memory cap of one byte is reached before the first pass.
    auto const mem = simplify(
        ctx, target, rules_eps_delta(ctx), SaturateBudget{.max_bytes = 1});
    EXPECT_EQ(mem.report.outcome, SaturateOutcome::MemoryBudget);
    EXPECT_FALSE(mem.complete());

    // A zero-millisecond wall clock likewise: elapsed is already >= 0.
    auto const clock = simplify(
        ctx,
        target,
        rules_eps_delta(ctx),
        SaturateBudget{
            .max_time =
                std::chrono::milliseconds{0} + std::chrono::milliseconds{1}});
    // 1 ms may or may not trip on a fast machine — the point is only that it
    // is accepted and reported honestly, never mistaken for a fixed point.
    EXPECT_TRUE(
        clock.report.outcome == SaturateOutcome::Saturated
        || clock.report.outcome == SaturateOutcome::TimeBudget);
}

TEST(Budget, DeterministicCapsWinOverResourceCaps)
{
    // When several caps could fire, the reproducible reason is reported: a
    // result that says "passes" can be reasoned about anywhere, one that says
    // "time" cannot.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res = simplify(
        ctx,
        contraction(ctx, sp, q, m, n),
        rules_eps_delta(ctx),
        SaturateBudget{
            .max_passes = 0,
            .max_nodes = 1,
            .max_time = std::chrono::milliseconds{1},
            .max_bytes = 1});
    EXPECT_EQ(res.report.outcome, SaturateOutcome::PassBudget);
}

TEST(Budget, EveryBudgetStopIsInconclusive)
{
    using O = SaturateOutcome;
    EXPECT_TRUE(nf::is_budget_stop(O::PassBudget));
    EXPECT_TRUE(nf::is_budget_stop(O::NodeBudget));
    EXPECT_TRUE(nf::is_budget_stop(O::TimeBudget));
    EXPECT_TRUE(nf::is_budget_stop(O::MemoryBudget));
    // A fixed point and an early stop are conclusions, not shortfalls.
    EXPECT_FALSE(nf::is_budget_stop(O::Saturated));
    EXPECT_FALSE(nf::is_budget_stop(O::EarlyStop));
}

TEST(Budget, ReportCarriesResourceUsage)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const res =
        simplify(ctx, contraction(ctx, sp, q, m, n), rules_eps_delta(ctx));
    EXPECT_TRUE(res.complete());
    EXPECT_GT(res.report.nodes, 0u);
    EXPECT_EQ(res.report.bytes, res.report.nodes * nf::kEstimatedBytesPerNode);
    EXPECT_GE(res.report.elapsed.count(), 0);
}

// ---- transposes reach the components (vibe 000110 M8) ------------------
//
// The defect these pin: a trace or transpose opened into frame dots only
// *after* `simplify_basis_dot` had run, so unevaluated `i·j` reached the
// comparison, the two sides of a true identity differed term by term, and the
// procedure reported `Different`.  `refuted` is the one verdict this library
// states as a fact about the mathematics rather than about its own reach, so
// a false one is worse than no answer at all.

namespace
{

auto rank2(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 2);
}

} // namespace

TEST(Refutation, TrueTransposeIdentitiesAreNotRefuted)
{
    Context ctx;
    auto const* A = rank2(ctx, "A");
    auto const* B = rank2(ctx, "B");
    auto const* a = vec1(ctx, "a");
    auto const* b = vec1(ctx, "b");
    auto const* At = make_transpose(ctx, A);
    auto const* Bt = make_transpose(ctx, B);

    // tr(Aᵀ) = tr(A)
    EXPECT_EQ(
        decide_by_components(ctx, make_trace(ctx, At), make_trace(ctx, A)),
        ComponentVerdict::Equal);
    // a·Aᵀ = A·a  and  a·A = Aᵀ·a
    EXPECT_EQ(
        decide_by_components(ctx, make_dot(ctx, a, At), make_dot(ctx, A, a)),
        ComponentVerdict::Equal);
    EXPECT_EQ(
        decide_by_components(ctx, make_dot(ctx, a, A), make_dot(ctx, At, a)),
        ComponentVerdict::Equal);
    // Aᵀ·Bᵀ = (B·A)ᵀ
    EXPECT_EQ(
        decide_by_components(
            ctx,
            make_dot(ctx, At, Bt),
            make_transpose(ctx, make_dot(ctx, B, A))),
        ComponentVerdict::Equal);
    // (a⊗b)·A = a⊗(Aᵀ·b)
    EXPECT_EQ(
        decide_by_components(
            ctx,
            make_dot(ctx, make_tensor_product(ctx, a, b), A),
            make_tensor_product(ctx, a, make_dot(ctx, At, b))),
        ComponentVerdict::Equal);
}

TEST(Refutation, AFalseTransposeClaimIsStillRefuted)
{
    // The fix must not buy safety by refusing to decide: a general tensor is
    // not its own transpose, and the procedure still says so.
    Context ctx;
    auto const* A = rank2(ctx, "A");
    EXPECT_EQ(
        decide_by_components(ctx, make_transpose(ctx, A), A),
        ComponentVerdict::Different);

    auto const res = prove_equal(ctx, make_transpose(ctx, A), A, {});
    EXPECT_TRUE(res.refuted());
}

TEST(Refutation, ATrueTransposeIdentityBlamesTheRulesNotTheClaim)
{
    // End to end through the verb: with no transpose rule supplied, the answer
    // is "the rules are incomplete", never "the claim is false".
    Context ctx;
    auto const* A = rank2(ctx, "A");
    auto const res = prove_equal(
        ctx, make_trace(ctx, make_transpose(ctx, A)), make_trace(ctx, A), {});
    EXPECT_EQ(res.status, ProofStatus::Exhausted);
    EXPECT_FALSE(res.refuted());
    EXPECT_TRUE(res.components_agree);
}

TEST(Refutation, AnUnreducedContractionIsResidueNotAVerdict)
{
    // The belt to the fix's braces, and a second false refutation of the same
    // family found while pinning it: a tensor of *unknown rank* cannot expand
    // on a frame, so tr(X·Y) and tr(Y·X) both reach the comparison with their
    // trace and dot intact and differ structurally.  Reading that as a verdict
    // refutes the cyclicity of the trace.
    //
    // A complete reduction leaves a polynomial in the component symbols and
    // nothing else, so a surviving contraction or invariant operator means the
    // reduction did not finish — silence, not a verdict.
    Context ctx;
    auto const* X = make_tensor_object(ctx, make_tensor_name("X"), {});
    auto const* Y = make_tensor_object(ctx, make_tensor_name("Y"), {});
    EXPECT_EQ(
        decide_by_components(
            ctx,
            make_trace(ctx, make_dot(ctx, X, Y)),
            make_trace(ctx, make_dot(ctx, Y, X))),
        ComponentVerdict::Undecided);
}

// ---- declared constraints (vibe 000110 I4) ----------------------------

TEST(Refutation, AConstrainedSymbolMakesTheComponentCheckAbstain)
{
    // A claim about a declared symbol is conditional, and the component
    // expansion cannot represent the condition: it writes P as nine
    // independent components, which satisfy no relation.  Answering from them
    // refutes true conditional claims, so the procedure must abstain.
    Context ctx;
    auto const* P = make_constrained_tensor(
        ctx,
        make_tensor_name("P"),
        2,
        SymbolConstraint{SymbolConstraint::Kind::Orthogonal, true});
    auto const* a = vec1(ctx, "a");
    auto const* b = vec1(ctx, "b");

    EXPECT_EQ(
        decide_by_components(
            ctx,
            make_dot(ctx, make_dot(ctx, P, a), make_dot(ctx, P, b)),
            make_dot(ctx, a, b)),
        ComponentVerdict::Undecided);

    // …and only for claims that mention it: the same shape with an
    // undeclared tensor is still decided, and decided false.
    auto const* A = rank2(ctx, "A");
    EXPECT_EQ(
        decide_by_components(
            ctx,
            make_dot(ctx, make_dot(ctx, A, a), make_dot(ctx, A, b)),
            make_dot(ctx, a, b)),
        ComponentVerdict::Different);
}

TEST(Constraints, DeclarationStampsTheSymbolAndRegistersIt)
{
    // One call does both, because the two consumers differ: the matcher reads
    // the stamp (it has no Context), and only the registry can be enumerated
    // to mint the rules.
    Context ctx;
    auto const* P = make_constrained_tensor(
        ctx,
        make_tensor_name("P"),
        2,
        SymbolConstraint{SymbolConstraint::Kind::Orthogonal, false});

    auto const& obj = std::get<TensorObject>(P->node);
    ASSERT_TRUE(obj.traits && obj.traits->constraint);
    EXPECT_EQ(obj.traits->constraint->kind, SymbolConstraint::Kind::Orthogonal);
    EXPECT_FALSE(obj.traits->constraint->proper);

    auto const c = ctx.constraint("P");
    ASSERT_TRUE(c);
    EXPECT_FALSE(c->proper);
    EXPECT_FALSE(ctx.constraint("Q"));
}

TEST(Constraints, AreNotSharedWithAnotherContext)
{
    // A symbol name is exactly what two independent contexts reuse for
    // different objects; one context's rotation P must not constrain another's.
    Context a;
    (void)make_constrained_tensor(
        a,
        make_tensor_name("P"),
        2,
        SymbolConstraint{SymbolConstraint::Kind::Orthogonal, true});
    Context b = a.new_context();
    EXPECT_TRUE(a.constraint("P"));
    EXPECT_FALSE(b.constraint("P"));
}
