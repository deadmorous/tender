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
