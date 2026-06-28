#include <tender/nf_match.hpp>

#include <tender/derivation.hpp> // steps::canonicalize
#include <tender/expr.hpp>
#include <tender/index_space.hpp> // space_3d
#include <tender/name.hpp>
#include <tender/nf_lower.hpp> // canonicalize_nf

#include <gtest/gtest.h>

using namespace tender;
using namespace tender::nf;

namespace
{

// The canonical `Nf` of `e`: the round-trip property (C12) gives
// `canonicalize_nf(canonicalize(e)) == the canonical Nf of e`, so this needs no
// materialize/float prep of its own.
auto to_nf(Context& ctx, Expr const* e) -> Nf const*
{
    return canonicalize_nf(ctx, steps::canonicalize(ctx, e));
}

auto only_term(Nf const* nf) -> Term const&
{
    EXPECT_EQ(nf->terms.size(), 1u);
    return nf->terms.at(0);
}

// A single lowered factor (the sole tensor of a one-term, one-factor Nf).
auto only_tensor(Context& ctx, Expr const* e) -> Factor const*
{
    Term const& t = only_term(to_nf(ctx, e));
    EXPECT_EQ(t.tensors.size(), 1u);
    return t.tensors.at(0);
}

auto delta_ul(Context& ctx, IndexSpace const* sp, IndexAssoc a, IndexAssoc b)
    -> Expr const*
{
    return make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, a, b);
}

// Σ_p δ^p_a δ^p_b lowered to its single canonical term.
auto contraction_term(
    Context& ctx,
    IndexSpace const* sp,
    CountableIndex a,
    CountableIndex b) -> Term
{
    CountableIndex p{ctx.alloc_index_id()};
    auto const* e = make_explicit_sum(
        ctx,
        p,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, p, a), delta_ul(ctx, sp, p, b)));
    return only_term(to_nf(ctx, e));
}

auto vrank1(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 1);
}

auto rank2(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 2);
}

auto scalar0(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 0);
}

} // namespace

// ---- whole-term matching ---------------------------------------------------

TEST(MatchTerm, ContractionMatchesAlphaRenamed)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    Term const pat = contraction_term(ctx, sp, a, b);
    Term const tgt = contraction_term(ctx, sp, m, n);

    NfBinding bnd;
    ASSERT_TRUE(match_term(pat, tgt, bnd));
    // The pattern's free indices bind to the target's.
    EXPECT_TRUE(bnd.find(a.id).has_value());
    EXPECT_TRUE(bnd.find(b.id).has_value());
}

TEST(MatchTerm, ExtraTargetFactorFailsWholeMatch)
{
    // A whole-term match is exact: the same δ-contraction times an extra,
    // unrelated subtree factor does not match (that is partial matching, C14b).
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    Term const pat = contraction_term(ctx, sp, a, b);
    CountableIndex p{ctx.alloc_index_id()};
    auto const* withextra = make_explicit_sum(
        ctx,
        p,
        make_tensor_product(
            ctx,
            make_tensor_product(
                ctx, delta_ul(ctx, sp, p, m), delta_ul(ctx, sp, p, n)),
            vrank1(ctx, "u")));
    Term const tgt = only_term(to_nf(ctx, withextra));

    NfBinding bnd;
    EXPECT_FALSE(match_term(pat, tgt, bnd));
}

TEST(MatchTerm, ScalarRegionMatchesModuloOrder)
{
    // The two δ factors are commutative scalars; a target built in the opposite
    // factor order still matches (AC over the scalar region).  Canonicalization
    // already sorts them, but the AC matcher must not depend on that.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    Term const pat = contraction_term(ctx, sp, a, b);
    CountableIndex p{ctx.alloc_index_id()};
    auto const* swapped = make_explicit_sum(
        ctx,
        p,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, p, n), delta_ul(ctx, sp, p, m)));
    Term const tgt = only_term(to_nf(ctx, swapped));

    NfBinding bnd;
    EXPECT_TRUE(match_term(pat, tgt, bnd));
}

TEST(MatchTerm, DifferentBoundCountFails)
{
    // A single contraction does not match a bare (unsummed) delta.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    Term const pat = contraction_term(ctx, sp, a, b);
    Term const tgt = only_term(to_nf(ctx, delta_ul(ctx, sp, m, n)));

    NfBinding bnd;
    EXPECT_FALSE(match_term(pat, tgt, bnd));
}

// ---- partial sub-product matching ------------------------------------------

TEST(MatchTermPartial, ContractionInsideLargerTerm)
{
    // Target: Σ_p Σ_q δ^p_m δ^p_n δ^q_r δ^q_s.  The δ-contraction pattern
    // matches one of the two independent contractions, leaving the other as
    // leftover.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    CountableIndex r{ctx.alloc_index_id()};
    CountableIndex s{ctx.alloc_index_id()};

    Term const pat = contraction_term(ctx, sp, a, b);
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex q{ctx.alloc_index_id()};
    auto const* dbl = make_explicit_sum(
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
    Term const tgt = only_term(to_nf(ctx, dbl));
    ASSERT_EQ(tgt.scalars.size(), 4u);
    ASSERT_EQ(tgt.bound.size(), 2u);

    auto pm = match_term_partial(pat, tgt);
    ASSERT_TRUE(pm.has_value());
    EXPECT_EQ(pm->leftover.scalars.size(), 2u); // the other contraction
    EXPECT_EQ(pm->leftover.bound.size(), 1u);   // its dummy survives
    EXPECT_TRUE(pm->leftover.coeff == Rational{1});
}

TEST(MatchTermPartial, SharedDummyInLeftoverRejected)
{
    // Target: Σ_p δ^p_m δ^p_n δ^p_r — one dummy across three factors. Consuming
    // p for a two-factor pattern would leave p dangling in the third factor, so
    // the partial match must be rejected.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    CountableIndex r{ctx.alloc_index_id()};

    Term const pat = contraction_term(ctx, sp, a, b);
    CountableIndex p{ctx.alloc_index_id()};
    auto const* triple = make_explicit_sum(
        ctx,
        p,
        make_tensor_product(
            ctx,
            make_tensor_product(
                ctx, delta_ul(ctx, sp, p, m), delta_ul(ctx, sp, p, n)),
            delta_ul(ctx, sp, p, r)));
    Term const tgt = only_term(to_nf(ctx, triple));
    ASSERT_EQ(tgt.bound.size(), 1u);

    EXPECT_FALSE(match_term_partial(pat, tgt).has_value());
}

TEST(MatchTermPartial, WholeTermLeftoverIsUnit)
{
    // When the pattern is the whole term, the leftover is the empty unit.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    Term const pat = contraction_term(ctx, sp, a, b);
    Term const tgt = contraction_term(ctx, sp, m, n);

    auto pm = match_term_partial(pat, tgt);
    ASSERT_TRUE(pm.has_value());
    EXPECT_TRUE(pm->leftover.scalars.empty());
    EXPECT_TRUE(pm->leftover.tensors.empty());
    EXPECT_TRUE(pm->leftover.bound.empty());
    EXPECT_TRUE(pm->leftover.coeff == Rational{1});
}

TEST(MatchTermPartial, LeftoverCarriesCoeffRatio)
{
    // 2 · (δ-contraction): the leftover carries the coefficient 2.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    Term const pat = contraction_term(ctx, sp, a, b);
    CountableIndex p{ctx.alloc_index_id()};
    auto const* scaled = make_tensor_product(
        ctx,
        make_scalar(ctx, Rational{2}),
        make_explicit_sum(
            ctx,
            p,
            make_tensor_product(
                ctx, delta_ul(ctx, sp, p, m), delta_ul(ctx, sp, p, n))));
    Term const tgt = only_term(to_nf(ctx, scaled));
    ASSERT_TRUE(tgt.coeff == Rational{2});

    auto pm = match_term_partial(pat, tgt);
    ASSERT_TRUE(pm.has_value());
    EXPECT_TRUE(pm->leftover.coeff == Rational{2});
    EXPECT_TRUE(pm->leftover.scalars.empty());
}

// ---- sub-chain rewrite -----------------------------------------------------

TEST(RewriteSubchain, CommuteInsideCrossChain)
{
    // I × x = x × I applied inside a × I × b (a flat Cross chain [a, I, b]):
    // the I × b sub-run rewrites to b × I, giving a × b × I.
    Context ctx;
    auto const* a = vrank1(ctx, "a");
    auto const* b = vrank1(ctx, "b");
    auto const* x = vrank1(ctx, "x"); // subtree variable
    auto const* I = make_identity(ctx);

    Term const pat = only_term(to_nf(ctx, make_cross(ctx, I, x)));
    Nf const* rhs = to_nf(ctx, make_cross(ctx, x, I));
    Term const tgt =
        only_term(to_nf(ctx, make_cross(ctx, make_cross(ctx, a, I), b)));

    auto rt = rewrite_subchain(ctx, pat, rhs, tgt);
    ASSERT_TRUE(rt.has_value());

    std::vector<Term> one{*rt};
    auto const* got = raise(ctx, *make_nf(ctx, std::move(one)));
    auto const* want = make_cross(ctx, a, make_cross(ctx, b, I)); // a × b × I
    EXPECT_TRUE(algebraic_eq(ctx, got, want));
}

TEST(RewriteSubchain, NonChainRuleDeclines)
{
    // A δ-contraction is not a single chain factor, so it is not a chain rule.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    Term const pat = contraction_term(ctx, sp, a, b);
    Nf const* rhs = to_nf(ctx, delta_ul(ctx, sp, a, b));
    Term const tgt = contraction_term(ctx, sp, m, n);

    EXPECT_FALSE(rewrite_subchain(ctx, pat, rhs, tgt).has_value());
}

TEST(RewriteSubchain, CommuteInsideNestedContractionChain)
{
    // I · x = x · I as a *contraction* chain rule, applied to a · (I · b): the
    // run hides one level down, so rewrite recurses into the nested Contraction
    // factor (the Contraction analog of the Cross-chain payoff).  Built with
    // the Nf factor builders to control the chain shape directly.
    Context ctx;
    auto const* a = only_tensor(ctx, vrank1(ctx, "a"));
    auto const* b = only_tensor(ctx, vrank1(ctx, "b"));
    auto const* x = only_tensor(ctx, vrank1(ctx, "x")); // subtree variable
    auto const* I = only_tensor(ctx, make_identity(ctx));

    Term pat;
    pat.coeff = Rational{1};
    pat.tensors = {make_contraction(ctx, {I, x}, {COp::Dot})};

    Term rep;
    rep.coeff = Rational{1};
    rep.tensors = {make_contraction(ctx, {x, I}, {COp::Dot})};
    auto const* rhs = make_nf(ctx, {rep});

    Term tgt;
    tgt.coeff = Rational{1};
    tgt.tensors = {make_contraction(
        ctx, {a, make_contraction(ctx, {I, b}, {COp::Dot})}, {COp::Dot})};

    auto rt = rewrite_subchain(ctx, pat, rhs, tgt);
    ASSERT_TRUE(rt.has_value());
    // The inner I · b became b · I.
    auto const* want = make_contraction(
        ctx, {a, make_contraction(ctx, {b, I}, {COp::Dot})}, {COp::Dot});
    ASSERT_EQ(rt->tensors.size(), 1u);
    EXPECT_TRUE(equal(rt->tensors.front(), want));
}

TEST(MatchTermPartial, LeftoverCollectsIdsOverCompoundFactors)
{
    // A single subtree-variable pattern matches one tensor of a target term
    // whose other factors are every compound kind; the soundness pass that
    // collects the leftover's index ids must recurse through them all.
    Context ctx;
    auto const* fa = only_tensor(ctx, vrank1(ctx, "a"));
    auto const* fb = only_tensor(ctx, vrank1(ctx, "b"));
    auto const* sum =
        to_nf(ctx, make_sum(ctx, scalar0(ctx, "p"), scalar0(ctx, "q")));
    auto const* xnf = to_nf(ctx, scalar0(ctx, "x"));

    Term pat;
    pat.coeff = Rational{1};
    pat.tensors = {only_tensor(ctx, vrank1(ctx, "v"))}; // subtree variable

    Term tgt;
    tgt.coeff = Rational{1};
    tgt.scalars = {make_div(ctx, xnf, sum)};
    tgt.tensors = {
        make_cross(ctx, {fa, fb}),
        make_contraction(ctx, {fa, fb}, {COp::Dot}),
        make_unary(ctx, UnaryOp::Transpose, fa),
        make_paren(ctx, sum)};

    auto pm = match_term_partial(pat, tgt);
    ASSERT_TRUE(pm.has_value());
    // One tensor consumed; the three other tensors carry through as leftover.
    EXPECT_EQ(pm->leftover.tensors.size(), 3u);
}

// ---- factor matching -------------------------------------------------------

TEST(MatchFactor, SubtreeVarBindsAnyFactor)
{
    Context ctx;
    auto const* pat = only_tensor(ctx, vrank1(ctx, "a")); // subtree variable
    auto const* tgt = only_tensor(ctx, vrank1(ctx, "x"));

    NfBinding bnd;
    EXPECT_TRUE(match_factor(pat, tgt, bnd));
    EXPECT_EQ(bnd.find_subtree("a"), tgt);
}

TEST(MatchFactor, SubtreeVarConsistencyAcrossUses)
{
    Context ctx;
    auto const* a = only_tensor(ctx, vrank1(ctx, "a"));
    auto const* x = only_tensor(ctx, vrank1(ctx, "x"));
    auto const* y = only_tensor(ctx, vrank1(ctx, "y"));

    NfBinding bnd;
    ASSERT_TRUE(match_factor(a, x, bnd));
    // Second use of `a` must bind the same factor.
    EXPECT_FALSE(match_factor(a, y, bnd));
    EXPECT_TRUE(match_factor(a, x, bnd));
}

TEST(MatchFactor, WellKnownStaysLiteral)
{
    // The identity tensor is well-known, so it is a literal, not a variable.
    Context ctx;
    auto const* I = only_tensor(ctx, make_identity(ctx));
    auto const* A =
        only_tensor(ctx, make_tensor_object(ctx, make_tensor_name("A"), {}, 2));

    NfBinding bnd;
    EXPECT_TRUE(match_factor(I, I, bnd));
    EXPECT_FALSE(match_factor(I, A, bnd));
}

// ---- match_factor over every Factor kind -----------------------------------

TEST(MatchFactor, CompoundKindsMatchThemselves)
{
    // Each non-Atom Factor kind matches an identical copy (exercising the
    // Contraction / Cross / Unary / Paren / Div arms and the match_nf recursion
    // the Paren / Div bottom out in).  Built directly with the Nf factor
    // builders to sidestep lowering's distribution / collapse rules.
    Context ctx;
    auto const* fa = only_tensor(ctx, vrank1(ctx, "a"));
    auto const* fb = only_tensor(ctx, vrank1(ctx, "b"));
    auto const* sum =
        to_nf(ctx, make_sum(ctx, scalar0(ctx, "p"), scalar0(ctx, "q")));
    auto const* xnf = to_nf(ctx, scalar0(ctx, "x"));

    auto const* contraction = make_contraction(ctx, {fa, fb}, {COp::Dot});
    auto const* cross = make_cross(ctx, {fa, fb});
    auto const* paren = make_paren(ctx, sum);
    auto const* unary = make_unary(ctx, UnaryOp::Transpose, fa);
    auto const* div = make_div(ctx, xnf, sum);

    for (auto const* f: {contraction, cross, paren, unary, div})
    {
        NfBinding bnd;
        EXPECT_TRUE(match_factor(f, f, bnd));
    }
}

TEST(MatchFactor, SubtreeVarRankGateOverFactorKinds)
{
    // A rank-1 subtree variable binds a target factor only when that factor's
    // inferred rank is 1 (or unknown).  This exercises factor_rank / nf_rank
    // over every Factor kind: Atom, Contraction, Cross, Paren, Unary (all three
    // ops), and Div (both the scalar- and tensor-region nf_rank loops).
    Context ctx;
    auto const* var = only_tensor(ctx, vrank1(ctx, "a")); // rank-1 subtree var
    auto const* fa = only_tensor(ctx, vrank1(ctx, "u"));
    auto const* fb = only_tensor(ctx, vrank1(ctx, "v"));
    auto const* m2 = only_tensor(ctx, rank2(ctx, "M"));
    auto const* scalarsum =
        to_nf(ctx, make_sum(ctx, scalar0(ctx, "p"), scalar0(ctx, "q")));
    auto const* vecnf = to_nf(ctx, vrank1(ctx, "w"));
    auto const* den = to_nf(ctx, scalar0(ctx, "d"));

    auto rank_of = [&](Factor const* tgt)
    {
        NfBinding bnd;
        return match_factor(var, tgt, bnd); // true ⇔ inferred rank 1 (or
                                            // unknown)
    };

    // Rank ≠ 1 → the gate rejects the bind.
    EXPECT_FALSE(rank_of(only_tensor(ctx, rank2(ctx, "X")))); // Atom (2)
    EXPECT_FALSE(rank_of(make_contraction(ctx, {fa, fb}, {COp::Dot}))); // 0
    EXPECT_FALSE(rank_of(make_paren(ctx, scalarsum)));                  // 0
    EXPECT_FALSE(rank_of(make_unary(ctx, UnaryOp::Trace, m2)));         // 0

    // Rank == 1 → the gate passes and the variable binds.
    EXPECT_TRUE(rank_of(make_cross(ctx, {fa, fb})));                     // 1
    EXPECT_TRUE(rank_of(make_unary(ctx, UnaryOp::Transpose, fa)));       // 1
    EXPECT_TRUE(rank_of(make_unary(ctx, UnaryOp::VectorInvariant, m2))); // 1
    EXPECT_TRUE(rank_of(make_div(ctx, vecnf, den)));                     // 1
}

TEST(MatchFactor, DistinctKindsDoNotMatch)
{
    Context ctx;
    auto const* fa = only_tensor(ctx, vrank1(ctx, "a"));
    auto const* fb = only_tensor(ctx, vrank1(ctx, "b"));
    auto const* cross = make_cross(ctx, {fa, fb});
    auto const* contraction = make_contraction(ctx, {fa, fb}, {COp::Dot});
    auto const* atom = only_tensor(ctx, make_identity(ctx)); // literal Atom

    NfBinding bnd;
    EXPECT_FALSE(match_factor(cross, contraction, bnd));
    EXPECT_FALSE(match_factor(contraction, cross, bnd));
    EXPECT_FALSE(match_factor(atom, cross, bnd));
}

TEST(MatchFactor, ContractionOpsAndArityMismatch)
{
    Context ctx;
    auto const* fa = only_tensor(ctx, vrank1(ctx, "a"));
    auto const* fb = only_tensor(ctx, vrank1(ctx, "b"));
    auto const* fc = only_tensor(ctx, vrank1(ctx, "c"));
    auto const* dot = make_contraction(ctx, {fa, fb}, {COp::Dot});
    auto const* ddot = make_contraction(ctx, {fa, fb}, {COp::DDot});
    auto const* three =
        make_contraction(ctx, {fa, fb, fc}, {COp::Dot, COp::Dot});

    NfBinding bnd;
    EXPECT_FALSE(match_factor(dot, ddot, bnd));  // op sequence differs
    EXPECT_FALSE(match_factor(dot, three, bnd)); // factor count differs
    EXPECT_FALSE(match_factor(
        make_cross(ctx, {fa, fb}),
        make_cross(ctx, {fa, fb, fc}),
        bnd)); // cross
               // arity
}

TEST(MatchFactor, SlotDescriptorMismatches)
{
    // Two slotted tensors differing only in a slot descriptor (level / realm /
    // space, or index presence) must not match — match_slot is exact on each.
    Context ctx;
    auto const* sp3 = space_3d();
    auto const* sp2 = space_2d();
    CountableIndex i{ctx.alloc_index_id()};

    auto t = [&](Realm r, Level l, IndexSpace const* sp, bool filled)
    {
        return only_tensor(
            ctx,
            make_tensor_object(
                ctx,
                make_tensor_name("T"),
                {filled ? SlotBinding{IndexSlot{l, r, sp}, IndexAssoc{i}} :
                          SlotBinding{IndexSlot{l, r, sp}, std::nullopt}}));
    };

    auto const* base = t(Realm::Oblique, Level::Upper, sp3, true);
    NfBinding bnd;
    EXPECT_TRUE(match_factor(base, base, bnd));
    EXPECT_FALSE(
        match_factor(base, t(Realm::Oblique, Level::Lower, sp3, true), bnd));
    EXPECT_FALSE(
        match_factor(base, t(Realm::Orthonormal, Level::Upper, sp3, true), bnd));
    EXPECT_FALSE(
        match_factor(base, t(Realm::Oblique, Level::Upper, sp2, true), bnd));
    // Index-presence mismatch: filled pattern slot vs void target slot.
    EXPECT_FALSE(
        match_factor(base, t(Realm::Oblique, Level::Upper, sp3, false), bnd));
}

TEST(MatchFactor, BasisIdDontCareWhenPatternUnset)
{
    // basis_id (vibe 000067): a pattern that leaves basis_id unset (0) is
    // basis-generic and matches any tagged target; a pattern that pins a basis
    // matches only that basis.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};

    auto vec = [&](int basis_id)
    {
        return only_tensor(
            ctx,
            make_tensor_object(
                ctx,
                make_tensor_name("e"),
                {SlotBinding{
                    IndexSlot{Level::Lower, Realm::Orthonormal, sp, basis_id},
                    IndexAssoc{i}}},
                1));
    };

    NfBinding bnd;
    // Unset (0) pattern is don't-care: matches both untagged and tagged.
    EXPECT_TRUE(match_factor(vec(0), vec(0), bnd));
    EXPECT_TRUE(match_factor(vec(0), vec(5), bnd));
    // A pinned pattern matches its own basis only.
    EXPECT_TRUE(match_factor(vec(5), vec(5), bnd));
    EXPECT_FALSE(match_factor(vec(5), vec(6), bnd));
    // A pinned pattern does not match an untagged target.
    EXPECT_FALSE(match_factor(vec(5), vec(0), bnd));
}

TEST(MatchFactor, ConcreteAndLabelSlotIndices)
{
    // A concrete (numeric) and a label index in a slot match by value/name and
    // mismatch otherwise (assoc_eq's Concrete / Label arms).
    Context ctx;
    auto const* sp = space_3d();
    auto cidx = [&](int v)
    {
        return only_tensor(
            ctx,
            make_tensor_object(
                ctx,
                make_tensor_name("T"),
                {SlotBinding{
                    IndexSlot{Level::Upper, Realm::Orthonormal, sp},
                    ConcreteIndex{v}}}));
    };
    auto lidx = [&](char const* nm)
    {
        return only_tensor(
            ctx,
            make_tensor_object(
                ctx,
                make_tensor_name("T"),
                {SlotBinding{
                    IndexSlot{Level::Lower, Realm::Label, nullptr},
                    LabelIndex{make_index_name(nm)}}}));
    };

    NfBinding bnd;
    EXPECT_TRUE(match_factor(cidx(1), cidx(1), bnd));
    EXPECT_FALSE(match_factor(cidx(1), cidx(2), bnd));
    EXPECT_TRUE(match_factor(lidx("vol"), lidx("vol"), bnd));
    EXPECT_FALSE(match_factor(lidx("vol"), lidx("surf"), bnd));
}

// ---- match_term negative paths ---------------------------------------------

TEST(MatchTerm, CoefficientMismatchFails)
{
    Context ctx;
    Term const pat = only_term(to_nf(
        ctx,
        make_tensor_product(
            ctx, make_scalar(ctx, Rational{2}), vrank1(ctx, "x"))));
    Term const tgt = only_term(to_nf(
        ctx,
        make_tensor_product(
            ctx, make_scalar(ctx, Rational{3}), vrank1(ctx, "x"))));
    NfBinding bnd;
    EXPECT_FALSE(match_term(pat, tgt, bnd));
}

TEST(MatchTerm, DifferentTensorCountFails)
{
    Context ctx;
    Term const pat = only_term(to_nf(ctx, vrank1(ctx, "x")));
    Term const tgt = only_term(to_nf(
        ctx, make_tensor_product(ctx, vrank1(ctx, "x"), vrank1(ctx, "y"))));
    NfBinding bnd;
    EXPECT_FALSE(match_term(pat, tgt, bnd));
}

// ---- instantiate_nf over every Factor kind ---------------------------------

TEST(InstantiateNf, RemapsIndicesAndWalksAllKinds)
{
    // Instantiating the δ-contraction RHS-style term freshens the bound dummy
    // and remaps a bound free index; instantiating a term carrying compound
    // factors walks every inst_factor arm.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    // bind a→m, b→n on the contraction term, then instantiate.
    Term const pat = contraction_term(ctx, sp, a, b);
    Term const tgt = contraction_term(ctx, sp, m, n);
    NfBinding bnd;
    ASSERT_TRUE(match_term(pat, tgt, bnd));

    auto const* patnf = make_nf(ctx, {pat});
    auto const* inst = instantiate_nf(ctx, patnf, bnd);
    EXPECT_EQ(inst->terms.size(), 1u);

    // A term with a cross, a contraction-with-paren, a transpose, a trace, and
    // a scalar division — instantiate (empty binding) rebuilds each factor
    // kind.
    auto const* A = rank2(ctx, "A");
    auto const* compound = make_tensor_product(
        ctx,
        make_cross(ctx, vrank1(ctx, "a"), vrank1(ctx, "b")),
        make_dot(
            ctx,
            vrank1(ctx, "c"),
            make_sum(ctx, vrank1(ctx, "d"), vrank1(ctx, "e"))));
    auto const* withscalars = make_tensor_product(
        ctx,
        make_tensor_product(
            ctx,
            make_scalar_div(
                ctx,
                scalar0(ctx, "x"),
                make_sum(ctx, scalar0(ctx, "p"), scalar0(ctx, "q"))),
            make_trace(ctx, A)),
        make_tensor_product(ctx, make_transpose(ctx, A), compound));
    auto const* cnf = to_nf(ctx, withscalars);
    auto const* cinst = instantiate_nf(ctx, cnf, NfBinding{});
    EXPECT_TRUE(equal(cinst, cnf)); // empty binding is an identity rebuild
}
