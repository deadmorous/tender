#include <tender/basis.hpp>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/index_space.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <variant>

using namespace tender;

namespace
{

// A bare rank-1 named vector.
auto vec(Context& ctx, char const* name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 1);
}

// A standard 3D orthonormal basis i, j, k.
auto wcs_basis(Context& ctx) -> Basis
{
    return make_orthonormal_basis(
        space_3d(), {vec(ctx, "i"), vec(ctx, "j"), vec(ctx, "k")});
}

auto idx_of(SlotBinding const& sb) -> int
{
    return std::get<CountableIndex>(*sb.index).id;
}

} // namespace

TEST(Basis, OrthonormalWcsAccessors)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    auto const* k = vec(ctx, "k");
    auto b = make_orthonormal_basis(space_3d(), {i, j, k});

    EXPECT_EQ(b.realm(), Realm::Orthonormal);
    EXPECT_TRUE(b.is_orthonormal());
    EXPECT_EQ(b.space(), space_3d());
    EXPECT_EQ(b.dim(), 3);
    EXPECT_EQ(b.basis(0), i);
    EXPECT_EQ(b.basis(2), k);
    // Orthonormal: the cobasis coincides with the basis.
    EXPECT_EQ(b.cobasis(0), i);
    EXPECT_EQ(b.cobasis(1), j);
}

TEST(Basis, SubspaceBasisIsAllowed)
{
    // Two vectors over a 2-value space: a (sub)space basis of cardinality 2.
    // No ambient-dimension check is made (vibe 000049 §1).
    Context ctx;
    auto const* u = vec(ctx, "u");
    auto const* v = vec(ctx, "v");
    auto b = make_orthonormal_basis(space_2d(), {u, v});
    EXPECT_EQ(b.dim(), 2);
    EXPECT_EQ(b.basis(1), v);
    EXPECT_EQ(b.cobasis(1), v);
}

TEST(Basis, CardinalityMismatchThrows)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    // Two vectors but a 3-value space.
    EXPECT_THROW(
        (void)make_orthonormal_basis(space_3d(), {i, j}),
        std::invalid_argument);
}

TEST(Basis, EmptyVectorsThrow)
{
    EXPECT_THROW(
        (void)make_orthonormal_basis(space_3d(), {}), std::invalid_argument);
}

TEST(Basis, NullSpaceThrows)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    EXPECT_THROW(
        (void)make_orthonormal_basis(nullptr, {i}), std::invalid_argument);
}

TEST(Basis, NonRankOneVectorThrows)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    auto const* rank2 = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);
    EXPECT_THROW(
        (void)make_orthonormal_basis(space_3d(), {i, j, rank2}),
        std::invalid_argument);
}

TEST(Basis, NullVectorThrows)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    EXPECT_THROW(
        (void)make_orthonormal_basis(space_3d(), {i, j, nullptr}),
        std::invalid_argument);
}

TEST(Basis, OutOfRangeAccessThrows)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    auto const* k = vec(ctx, "k");
    auto b = make_orthonormal_basis(space_3d(), {i, j, k});
    EXPECT_THROW((void)b.basis(3), std::out_of_range);
    EXPECT_THROW((void)b.cobasis(5), std::out_of_range);
}

TEST(Basis, DefaultVectorSymbolIsE)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    auto const* k = vec(ctx, "k");
    auto b = make_orthonormal_basis(space_3d(), {i, j, k});
    EXPECT_EQ(b.vector_symbol().v.view(), "e");
}

TEST(Basis, SymbolicEmissionCovariant)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    auto const* k = vec(ctx, "k");
    auto b = make_orthonormal_basis(space_3d(), {i, j, k});

    auto m = CountableIndex{ctx.alloc_index_id()};
    auto const* e_m = b.covariant_vector(ctx, m);
    auto const& t = std::get<TensorObject>(e_m->node);
    EXPECT_EQ(t.name.v.view(), "e");
    EXPECT_EQ(t.rank, std::optional<int>{1});
    ASSERT_EQ(t.slots.size(), 1u);
    EXPECT_EQ(t.slots[0].slot.level, Level::Lower);
    EXPECT_EQ(t.slots[0].slot.realm, Realm::Orthonormal);
    EXPECT_EQ(t.slots[0].slot.space, space_3d());
    ASSERT_TRUE(t.slots[0].index.has_value());
    EXPECT_EQ(std::get<CountableIndex>(*t.slots[0].index).id, m.id);
}

TEST(Basis, SymbolicEmissionContravariantIsLowerForOrthonormal)
{
    // Orthonormal: e^i is spelled lower (upper/lower coincide), so covariant
    // and contravariant emissions agree.
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    auto const* k = vec(ctx, "k");
    auto b = make_orthonormal_basis(space_3d(), {i, j, k});

    auto m = CountableIndex{ctx.alloc_index_id()};
    auto const& t =
        std::get<TensorObject>(b.contravariant_vector(ctx, m)->node);
    EXPECT_EQ(t.slots[0].slot.level, Level::Lower);
}

TEST(Basis, CustomVectorSymbol)
{
    Context ctx;
    auto const* u = vec(ctx, "u");
    auto const* v = vec(ctx, "v");
    auto b = make_orthonormal_basis(space_2d(), {u, v}, "g");
    EXPECT_EQ(b.vector_symbol().v.view(), "g");
    auto m = CountableIndex{ctx.alloc_index_id()};
    EXPECT_EQ(
        std::get<TensorObject>(b.covariant_vector(ctx, m)->node).name.v.view(),
        "g");
}

// ---- expand_in_basis ---------------------------------------------------

TEST(ExpandInBasis, VectorCovariant)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);

    auto const* ex = expand_in_basis(ctx, a, b, Variance::Covariant);
    auto const* tp = std::get_if<TensorProduct>(&ex->node);
    ASSERT_NE(tp, nullptr);

    auto const& coord = std::get<TensorObject>(tp->left->node);
    auto const& evec = std::get<TensorObject>(tp->right->node);
    // coordinate a_i: rank 0, one lower slot.
    EXPECT_EQ(coord.name.v.view(), "a");
    EXPECT_EQ(coord.rank, std::optional<int>{0});
    ASSERT_EQ(coord.slots.size(), 1u);
    EXPECT_EQ(coord.slots[0].slot.level, Level::Lower);
    // basis vector e_i: rank 1, one lower slot, generic symbol.
    EXPECT_EQ(evec.name.v.view(), "e");
    EXPECT_EQ(evec.rank, std::optional<int>{1});
    ASSERT_EQ(evec.slots.size(), 1u);
    EXPECT_EQ(evec.slots[0].slot.level, Level::Lower);
    // The shared Einstein index.
    EXPECT_EQ(idx_of(coord.slots[0]), idx_of(evec.slots[0]));
}

TEST(ExpandInBasis, CanonicalizesToImplicitSum)
{
    // The repeated index makes the expansion an implicit Einstein sum, which
    // canonicalize materializes into an ExplicitSum.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);

    auto const* ex = expand_in_basis(ctx, a, b, Variance::Covariant);
    auto const* canon = steps::canonicalize(ctx, ex);
    EXPECT_TRUE(std::holds_alternative<ExplicitSum>(canon->node));
}

TEST(ExpandInBasis, Rank2)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);

    auto const* ex = expand_in_basis(ctx, A, b, Variance::Covariant);
    auto const* tp = std::get_if<TensorProduct>(&ex->node);
    ASSERT_NE(tp, nullptr);
    // coordinate A_{ij}: rank 0, two slots.
    auto const& coord = std::get<TensorObject>(tp->left->node);
    ASSERT_EQ(coord.slots.size(), 2u);
    // polyad e_i ⊗ e_j.
    auto const* poly = std::get_if<TensorProduct>(&tp->right->node);
    ASSERT_NE(poly, nullptr);
    auto const& ei = std::get<TensorObject>(poly->left->node);
    auto const& ej = std::get<TensorObject>(poly->right->node);
    // Indices pair up positionally: coord slot k shares with polyad vector k.
    EXPECT_EQ(idx_of(coord.slots[0]), idx_of(ei.slots[0]));
    EXPECT_EQ(idx_of(coord.slots[1]), idx_of(ej.slots[0]));
    EXPECT_NE(idx_of(coord.slots[0]), idx_of(coord.slots[1]));
}

TEST(ExpandInBasis, RecursesIntoDotWithDistinctIndices)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* c = make_tensor_object(ctx, make_tensor_name("c"), {}, 1);

    auto const* ex =
        expand_in_basis(ctx, make_dot(ctx, a, c), b, Variance::Covariant);
    auto const* dot = std::get_if<Dot>(&ex->node);
    ASSERT_NE(dot, nullptr);
    auto const* lhs = std::get_if<TensorProduct>(&dot->left->node);
    auto const* rhs = std::get_if<TensorProduct>(&dot->right->node);
    ASSERT_NE(lhs, nullptr);
    ASSERT_NE(rhs, nullptr);
    // a and c expand with independent dummy indices, so they do not contract.
    auto const& acoord = std::get<TensorObject>(lhs->left->node);
    auto const& ccoord = std::get<TensorObject>(rhs->left->node);
    EXPECT_NE(idx_of(acoord.slots[0]), idx_of(ccoord.slots[0]));
}

TEST(ExpandInBasis, LeavesWellKnownUnchanged)
{
    // The identity's coordinates are δ, not generic — it must be left alone.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* I = make_identity(ctx);
    EXPECT_EQ(expand_in_basis(ctx, I, b, Variance::Covariant), I);
}

TEST(ExpandInBasis, LeavesIndexedCoordinateUnchanged)
{
    // An already-indexed object (non-empty slots) is not an invariant.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto m = CountableIndex{ctx.alloc_index_id()};
    auto const* a_i = make_tensor_object(
        ctx,
        make_tensor_name("a"),
        {SlotBinding{
            IndexSlot{Level::Lower, Realm::Orthonormal, space_3d()},
            IndexAssoc{m}}},
        0);
    EXPECT_EQ(expand_in_basis(ctx, a_i, b, Variance::Covariant), a_i);
}

// ---- simplify_basis_dot ------------------------------------------------

TEST(SimplifyBasisDot, BasisVectorPairBecomesDelta)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* dot =
        make_dot(ctx, b.covariant_vector(ctx, i), b.covariant_vector(ctx, j));

    auto const* res = simplify_basis_dot(ctx, dot, b);
    auto const& d = std::get<TensorObject>(res->node);
    ASSERT_TRUE(d.traits.has_value());
    EXPECT_EQ(
        d.traits->well_known,
        std::optional<WellKnownKind>{WellKnownKind::Delta});
    ASSERT_EQ(d.slots.size(), 2u);
    EXPECT_EQ(d.slots[0].slot.level, Level::Lower);
    EXPECT_EQ(d.slots[1].slot.level, Level::Lower);
    EXPECT_EQ(idx_of(d.slots[0]), i.id);
    EXPECT_EQ(idx_of(d.slots[1]), j.id);
}

TEST(SimplifyBasisDot, PullsCoordinatesOutOfContraction)
{
    // The expansion of a·c, simplified: a_i ⊗ (c_j ⊗ δ_{ij}).
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* c = make_tensor_object(ctx, make_tensor_name("c"), {}, 1);
    auto const* expanded =
        expand_in_basis(ctx, make_dot(ctx, a, c), b, Variance::Covariant);

    auto const* res = simplify_basis_dot(ctx, expanded, b);
    auto const* top = std::get_if<TensorProduct>(&res->node);
    ASSERT_NE(top, nullptr);
    EXPECT_EQ(std::get<TensorObject>(top->left->node).name.v.view(), "a");
    auto const* inner = std::get_if<TensorProduct>(&top->right->node);
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(std::get<TensorObject>(inner->left->node).name.v.view(), "c");
    auto const& delta = std::get<TensorObject>(inner->right->node);
    ASSERT_TRUE(delta.traits.has_value());
    EXPECT_EQ(
        delta.traits->well_known,
        std::optional<WellKnownKind>{WellKnownKind::Delta});
}

TEST(SimplifyBasisDot, NonBasisDotUnchanged)
{
    // Invariant vectors that are not this basis's vectors: left alone.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* c = make_tensor_object(ctx, make_tensor_name("c"), {}, 1);
    auto const* dot = make_dot(ctx, a, c);
    EXPECT_EQ(simplify_basis_dot(ctx, dot, b), dot);
}

TEST(SimplifyBasisDot, DifferentVectorSymbolUnchanged)
{
    // e-symbol vectors are not recognized by a basis whose symbol is g.
    Context ctx;
    auto e_basis = wcs_basis(ctx); // symbol "e"
    auto g_basis = make_orthonormal_basis(
        space_3d(), {vec(ctx, "p"), vec(ctx, "q"), vec(ctx, "r")}, "g");
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* dot = make_dot(
        ctx,
        e_basis.covariant_vector(ctx, i),
        e_basis.covariant_vector(ctx, j));
    EXPECT_EQ(simplify_basis_dot(ctx, dot, g_basis), dot);
}
