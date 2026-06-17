#include <gtest/gtest.h>
#include <tender/derivation.hpp>
#include <tender/egraph.hpp>
#include <tender/expr.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>

#include <optional>

using namespace tender;

namespace
{
auto obj(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n));
}

auto delta_ul(Context& ctx, CountableIndex a, CountableIndex b) -> Expr const*
{
    return make_delta(
        ctx, Realm::Oblique, space_3d(), Level::Upper, Level::Lower, a, b);
}
} // namespace

// ---- hash-consing and round-trip -------------------------------------------

TEST(EGraph, RoundTripsCanonicalExpression)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");
    auto const* e = make_sum(ctx, make_tensor_product(ctx, A, B), A);

    EGraph eg{ctx};
    auto root = eg.add(e);
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(root), e));
}

TEST(EGraph, StructurallyEqualInputsShareAClass)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");

    EGraph eg{ctx};
    auto c1 = eg.add(make_sum(ctx, A, B));
    auto c2 = eg.add(make_sum(ctx, A, B));
    EXPECT_EQ(eg.find(c1), eg.find(c2));
}

TEST(EGraph, ACEquivalentInputsShareAClass)
{
    // Canonicalization on insertion means A + B and B + A hash-cons together.
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");

    EGraph eg{ctx};
    EXPECT_EQ(
        eg.find(eg.add(make_sum(ctx, A, B))),
        eg.find(eg.add(make_sum(ctx, B, A))));
}

TEST(EGraph, AlphaEquivalentSumsShareAClass)
{
    // Σ_i δ^i_i and Σ_j δ^j_j are α-equivalent; canonicalization aligns them.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto const* ei = make_explicit_sum(ctx, i, delta_ul(ctx, i, i));
    auto const* ej = make_explicit_sum(ctx, j, delta_ul(ctx, j, j));

    EGraph eg{ctx};
    EXPECT_EQ(eg.find(eg.add(ei)), eg.find(eg.add(ej)));
}

TEST(EGraph, SharedSubtermIsNotDuplicated)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");
    auto const* C = obj(ctx, "C");

    EGraph eg{ctx};
    (void)eg.add(make_dot(ctx, A, B));                   // e-nodes: A, B, Dot
    (void)eg.add(make_sum(ctx, make_dot(ctx, A, B), C)); // reuse Dot; add C,
                                                         // Sum
    // A, B, C, Dot(A,B), Sum — five distinct e-nodes (Dot hash-cons'd once).
    EXPECT_EQ(eg.node_count(), 5u);
}

// ---- merge and congruence --------------------------------------------------

TEST(EGraph, MergePropagatesByCongruence)
{
    // Merge A == B; then f(A) and f(B) must become equal after rebuild.
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");
    auto const* C = obj(ctx, "C");

    EGraph eg{ctx};
    auto ca = eg.add(A);
    auto cb = eg.add(B);
    auto fa = eg.add(make_dot(ctx, A, C));
    auto fb = eg.add(make_dot(ctx, B, C));
    EXPECT_NE(eg.find(fa), eg.find(fb));

    eg.merge(ca, cb);
    eg.rebuild();
    EXPECT_EQ(eg.find(fa), eg.find(fb)); // congruence closure
}

TEST(EGraph, ExtractChoosesCheapestRepresentative)
{
    // After proving a big expression equals a small one, extraction yields the
    // small one.
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");
    auto const* big = make_sum(ctx, make_dot(ctx, A, B), make_dot(ctx, B, A));
    auto const* small = obj(ctx, "C");

    EGraph eg{ctx};
    auto cbig = eg.add(big);
    auto csmall = eg.add(small);
    eg.merge(cbig, csmall);
    eg.rebuild();

    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.find(cbig)), small));
}

TEST(EGraph, ClassCountDropsAfterMerge)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");

    EGraph eg{ctx};
    auto ca = eg.add(A);
    auto cb = eg.add(B);
    auto before = eg.class_count();
    eg.merge(ca, cb);
    eg.rebuild();
    EXPECT_EQ(eg.class_count(), before - 1);
}

// ---- round-trips across node kinds (add + extract) -------------------------

TEST(EGraph, RoundTripsEveryReachableNodeKind)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");

    EGraph eg{ctx};
    for (auto const* e: {
             make_negate(ctx, A),
             make_scalar_div(ctx, A, B),
             make_dot(ctx, A, B),
             make_ddot(ctx, A, B),
             make_ddot_alt(ctx, A, B),
             make_cross(ctx, A, B),
             make_sum(ctx, A, A), // → 2·A, exercises a ScalarLiteral leaf
         })
        EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.add(e)), e));
}

TEST(EGraph, RoundTripsBoundNodes)
{
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto const* body = delta_ul(ctx, i, j);

    EGraph eg{ctx};
    // ExplicitSum with a symbolic bound (two children), and NoSum.
    auto const* es =
        make_explicit_sum(ctx, i, body, make_scalar(ctx, Rational{3}));
    auto const* ns = make_no_sum(ctx, i, body);
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.add(es)), es));
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.add(ns)), ns));
}

TEST(EGraph, IsMovable)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    EGraph eg{ctx};
    auto c = eg.add(A);

    EGraph moved{std::move(eg)}; // move-construct
    EXPECT_TRUE(algebraic_eq(ctx, moved.extract(moved.find(c)), A));

    EGraph eg2{ctx};
    eg2 = std::move(moved); // move-assign
    EXPECT_TRUE(algebraic_eq(ctx, eg2.extract(eg2.find(c)), A));
}

TEST(EGraph, MergingAClassWithItselfIsANoOp)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    EGraph eg{ctx};
    auto c = eg.add(A);
    EXPECT_EQ(eg.merge(c, c), eg.find(c));
}

TEST(EGraph, RoundTripsTensorWithVoidSlot)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    auto const* t = make_tensor_object(
        ctx,
        make_tensor_name("T"),
        {SlotBinding{IndexSlot{Level::Upper, Realm::Oblique, sp}, std::nullopt},
         SlotBinding{IndexSlot{Level::Lower, Realm::Oblique, sp}, i}},
        2);
    EGraph eg{ctx};
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.add(t)), t));
}

TEST(EGraph, ConcreteAndLabelLeaves)
{
    Context ctx;
    auto const* sp = space_3d();
    auto const* tc = make_tensor_object(
        ctx,
        make_tensor_name("v"),
        {SlotBinding{
            IndexSlot{Level::Upper, Realm::Orthonormal, sp}, ConcreteIndex{2}}},
        1);
    auto const* tl = make_tensor_object(
        ctx,
        make_tensor_name("A"),
        {SlotBinding{
            IndexSlot{Level::Lower, Realm::Label, nullptr},
            LabelIndex{make_index_name("vol")}}});

    EGraph eg{ctx};
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.add(tc)), tc));
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.add(tl)), tl));
    // A different concrete value is a distinct leaf (distinct e-class).
    auto const* tc3 = make_tensor_object(
        ctx,
        make_tensor_name("v"),
        {SlotBinding{
            IndexSlot{Level::Upper, Realm::Orthonormal, sp}, ConcreteIndex{3}}},
        1);
    EXPECT_NE(eg.find(eg.add(tc)), eg.find(eg.add(tc3)));
}
