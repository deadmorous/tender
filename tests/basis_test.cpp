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
        ctx, space_3d(), {vec(ctx, "i"), vec(ctx, "j"), vec(ctx, "k")});
}

// A 3D oblique basis with covariant vectors a, b, c.
auto oblique_basis(Context& ctx) -> Basis
{
    return make_oblique_basis(
        ctx, space_3d(), {vec(ctx, "a"), vec(ctx, "b"), vec(ctx, "c")});
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
    auto b = make_orthonormal_basis(ctx, space_3d(), {i, j, k});

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
    auto b = make_orthonormal_basis(ctx, space_2d(), {u, v});
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
        (void)make_orthonormal_basis(ctx, space_3d(), {i, j}),
        std::invalid_argument);
}

TEST(Basis, EmptyVectorsThrow)
{
    Context ctx;
    EXPECT_THROW(
        (void)make_orthonormal_basis(ctx, space_3d(), {}),
        std::invalid_argument);
}

TEST(Basis, NullSpaceThrows)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    EXPECT_THROW(
        (void)make_orthonormal_basis(ctx, nullptr, {i}), std::invalid_argument);
}

TEST(Basis, NonRankOneVectorThrows)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    auto const* rank2 = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);
    EXPECT_THROW(
        (void)make_orthonormal_basis(ctx, space_3d(), {i, j, rank2}),
        std::invalid_argument);
}

TEST(Basis, NullVectorThrows)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    EXPECT_THROW(
        (void)make_orthonormal_basis(ctx, space_3d(), {i, j, nullptr}),
        std::invalid_argument);
}

TEST(Basis, OutOfRangeAccessThrows)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    auto const* k = vec(ctx, "k");
    auto b = make_orthonormal_basis(ctx, space_3d(), {i, j, k});
    EXPECT_THROW((void)b.basis(3), std::out_of_range);
    EXPECT_THROW((void)b.cobasis(5), std::out_of_range);
}

TEST(Basis, DefaultVectorSymbolIsE)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    auto const* k = vec(ctx, "k");
    auto b = make_orthonormal_basis(ctx, space_3d(), {i, j, k});
    EXPECT_EQ(b.vector_symbol().v.view(), "e");
}

TEST(Basis, SymbolicEmissionCovariant)
{
    Context ctx;
    auto const* i = vec(ctx, "i");
    auto const* j = vec(ctx, "j");
    auto const* k = vec(ctx, "k");
    auto b = make_orthonormal_basis(ctx, space_3d(), {i, j, k});

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
    auto b = make_orthonormal_basis(ctx, space_3d(), {i, j, k});

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
    auto b =
        make_orthonormal_basis(ctx, space_2d(), {u, v}, make_tensor_name("g"));
    EXPECT_EQ(b.vector_symbol().v.view(), "g");
    auto m = CountableIndex{ctx.alloc_index_id()};
    EXPECT_EQ(
        std::get<TensorObject>(b.covariant_vector(ctx, m)->node).name.v.view(),
        "g");
}

TEST(Basis, ObliqueRealmAndEmission)
{
    Context ctx;
    auto b = oblique_basis(ctx);
    EXPECT_EQ(b.realm(), Realm::Oblique);
    EXPECT_FALSE(b.is_orthonormal());
    auto m = CountableIndex{ctx.alloc_index_id()};
    // Covariant lower, contravariant upper — distinct for an oblique basis.
    EXPECT_EQ(
        std::get<TensorObject>(b.covariant_vector(ctx, m)->node)
            .slots[0]
            .slot.level,
        Level::Lower);
    EXPECT_EQ(
        std::get<TensorObject>(b.contravariant_vector(ctx, m)->node)
            .slots[0]
            .slot.level,
        Level::Upper);
}

TEST(Basis, ObliqueCobasisIsReciprocal)
{
    // cobasis(0) = (e_1 × e_2) / V — a scalar division of a cross product.
    Context ctx;
    auto b = oblique_basis(ctx);
    EXPECT_TRUE(std::holds_alternative<ScalarDiv>(b.cobasis(0)->node));
}

TEST(Basis, ObliqueRequires3D)
{
    Context ctx;
    EXPECT_THROW(
        (void)
            make_oblique_basis(ctx, space_2d(), {vec(ctx, "u"), vec(ctx, "v")}),
        std::invalid_argument);
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

TEST(ExpandInBasis, LeavesWellKnownSymbolUnchanged)
{
    // A bare δ symbol is well-known with non-generic coordinates and no
    // resolution rule, so expand leaves it untouched (unlike the identity,
    // which now resolves to Σ_i e_i ⊗ e^i).
    Context ctx;
    auto b = wcs_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* d = make_delta(
        ctx,
        Realm::Orthonormal,
        space_3d(),
        Level::Lower,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    EXPECT_EQ(expand_in_basis(ctx, d, b, Variance::Covariant), d);
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

TEST(ExpandInBasis, PerSlotVarianceList)
{
    // Mixed variance A^i{}_j: covariant slot 0, contravariant slot 1.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);
    auto const* ex = expand_in_basis(
        ctx,
        A,
        b,
        std::vector<Variance>{Variance::Covariant, Variance::Contravariant});
    auto const* tp = std::get_if<TensorProduct>(&ex->node);
    ASSERT_NE(tp, nullptr);
    EXPECT_EQ(std::get<TensorObject>(tp->left->node).slots.size(), 2u);
    EXPECT_TRUE(std::holds_alternative<TensorProduct>(tp->right->node));
}

TEST(ExpandInBasis, MixedVarianceRoundTrips)
{
    // reassemble matches basis vectors by symbol, not variance, so a mixed
    // expansion still folds back to the invariant.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);
    auto const* ex = steps::canonicalize(
        ctx,
        expand_in_basis(
            ctx,
            A,
            b,
            std::vector<Variance>{
                Variance::Covariant, Variance::Contravariant}));
    EXPECT_TRUE(structural_eq(reassemble(ctx, ex, b), A));
}

TEST(ExpandInBasis, SingleVarianceBroadcastsToRank2)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);
    auto const* ex =
        expand_in_basis(ctx, A, b, std::vector<Variance>{Variance::Covariant});
    auto const* tp = std::get_if<TensorProduct>(&ex->node);
    ASSERT_NE(tp, nullptr);
    EXPECT_EQ(std::get<TensorObject>(tp->left->node).slots.size(), 2u);
}

TEST(ExpandInBasis, VarianceCountMismatchThrows)
{
    // A rank-1 tensor cannot take a length-2 variance pattern.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    EXPECT_THROW(
        (void)expand_in_basis(
            ctx,
            a,
            b,
            std::vector<Variance>{Variance::Covariant, Variance::Contravariant}),
        std::invalid_argument);
}

TEST(ExpandInBasis, EmptyVarianceThrows)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    EXPECT_THROW(
        (void)expand_in_basis(ctx, a, b, std::vector<Variance>{}),
        std::invalid_argument);
}

TEST(ExpandInBasis, IdentityResolvesToBasisVectorSum)
{
    // I expands to the resolution of identity Σ_i e_i ⊗ e^i — orthonormal: a
    // product of two e-vectors sharing one (implicitly summed) index.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* ex =
        expand_in_basis(ctx, make_identity(ctx), b, Variance::Covariant);
    auto const* tp = std::get_if<TensorProduct>(&ex->node);
    ASSERT_NE(tp, nullptr);
    auto const& l = std::get<TensorObject>(tp->left->node);
    auto const& r = std::get<TensorObject>(tp->right->node);
    EXPECT_EQ(l.name.v.view(), "e");
    EXPECT_EQ(r.name.v.view(), "e");
    EXPECT_EQ(idx_of(l.slots[0]), idx_of(r.slots[0]));
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
    // a·c simplifies to a_i ⊗ c_j ⊗ δ_{ij} — the scalar coordinates pulled out
    // of the contraction (factor order is up to canonicalization).
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* c = make_tensor_object(ctx, make_tensor_name("c"), {}, 1);
    auto const* res = simplify_basis_dot(
        ctx,
        expand_in_basis(ctx, make_dot(ctx, a, c), b, Variance::Covariant),
        b);

    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto coord = [&](char const* nm, CountableIndex idx)
    {
        return make_tensor_object(
            ctx,
            make_tensor_name(nm),
            {SlotBinding{
                IndexSlot{Level::Lower, Realm::Orthonormal, space_3d()},
                IndexAssoc{idx}}},
            0);
    };
    auto const* expected = make_tensor_product(
        ctx,
        coord("a", i),
        make_tensor_product(
            ctx,
            coord("c", j),
            make_delta(
                ctx,
                Realm::Orthonormal,
                space_3d(),
                Level::Lower,
                Level::Lower,
                IndexAssoc{i},
                IndexAssoc{j})));
    EXPECT_TRUE(algebraic_eq(ctx, res, expected));
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

TEST(SimplifyBasisDot, ObliqueSameLevelGivesMetric)
{
    // e_i · e_j (both covariant) → g_{ij} for an oblique basis.
    Context ctx;
    auto b = oblique_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* dot =
        make_dot(ctx, b.covariant_vector(ctx, i), b.covariant_vector(ctx, j));
    auto const& g =
        std::get<TensorObject>(simplify_basis_dot(ctx, dot, b)->node);
    ASSERT_TRUE(g.traits.has_value());
    EXPECT_EQ(
        g.traits->well_known,
        std::optional<WellKnownKind>{WellKnownKind::Metric});
    EXPECT_EQ(g.slots[0].slot.level, Level::Lower);
    EXPECT_EQ(g.slots[1].slot.level, Level::Lower);
}

TEST(SimplifyBasisDot, ObliqueUpperUpperGivesInverseMetric)
{
    // e^i · e^j → g^{ij}.
    Context ctx;
    auto b = oblique_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* dot = make_dot(
        ctx, b.contravariant_vector(ctx, i), b.contravariant_vector(ctx, j));
    auto const& g =
        std::get<TensorObject>(simplify_basis_dot(ctx, dot, b)->node);
    EXPECT_EQ(
        g.traits->well_known,
        std::optional<WellKnownKind>{WellKnownKind::Metric});
    EXPECT_EQ(g.slots[0].slot.level, Level::Upper);
    EXPECT_EQ(g.slots[1].slot.level, Level::Upper);
}

TEST(SimplifyBasisDot, ObliqueMixedLevelGivesDelta)
{
    // e_i · e^j → δ_i^j (Kronecker) even for an oblique basis.
    Context ctx;
    auto b = oblique_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* dot = make_dot(
        ctx, b.covariant_vector(ctx, i), b.contravariant_vector(ctx, j));
    auto const& d =
        std::get<TensorObject>(simplify_basis_dot(ctx, dot, b)->node);
    EXPECT_EQ(
        d.traits->well_known,
        std::optional<WellKnownKind>{WellKnownKind::Delta});
    EXPECT_EQ(d.slots[0].slot.level, Level::Lower);
    EXPECT_EQ(d.slots[1].slot.level, Level::Upper);
}

TEST(SimplifyBasisDot, OrthonormalSameLevelStaysDelta)
{
    // Orthonormal: two lower vectors still give δ (its metric is the identity).
    Context ctx;
    auto b = wcs_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* dot =
        make_dot(ctx, b.covariant_vector(ctx, i), b.covariant_vector(ctx, j));
    auto const& d =
        std::get<TensorObject>(simplify_basis_dot(ctx, dot, b)->node);
    EXPECT_EQ(
        d.traits->well_known,
        std::optional<WellKnownKind>{WellKnownKind::Delta});
}

TEST(SimplifyBasisDot, DifferentVectorSymbolUnchanged)
{
    // e-symbol vectors are not recognized by a basis whose symbol is g.
    Context ctx;
    auto e_basis = wcs_basis(ctx); // symbol "e"
    auto g_basis = make_orthonormal_basis(
        ctx,
        space_3d(),
        {vec(ctx, "p"), vec(ctx, "q"), vec(ctx, "r")},
        make_tensor_name("g"));
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* dot = make_dot(
        ctx,
        e_basis.covariant_vector(ctx, i),
        e_basis.covariant_vector(ctx, j));
    EXPECT_EQ(simplify_basis_dot(ctx, dot, g_basis), dot);
}

// ---- simplify_basis_cross ----------------------------------------------

TEST(SimplifyBasisCross, OrthonormalGivesEpsilonTimesBasisVector)
{
    // e_i × e_j → ε_{ijk} e_k (orthonormal: no √g, e^k spelled lower).
    Context ctx;
    auto b = wcs_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* cr =
        make_cross(ctx, b.covariant_vector(ctx, i), b.covariant_vector(ctx, j));

    auto const* tp =
        std::get_if<TensorProduct>(&simplify_basis_cross(ctx, cr, b)->node);
    ASSERT_NE(tp, nullptr);
    auto const& eps = std::get<TensorObject>(tp->left->node);
    auto const& evec = std::get<TensorObject>(tp->right->node);
    ASSERT_TRUE(eps.traits.has_value());
    EXPECT_EQ(
        eps.traits->well_known,
        std::optional<WellKnownKind>{WellKnownKind::LeviCivita});
    ASSERT_EQ(eps.slots.size(), 3u);
    EXPECT_EQ(evec.name.v.view(), "e");
    ASSERT_EQ(evec.slots.size(), 1u);
    // ε's first two slots are i and j; its third is the summed index, shared
    // with the basis vector e^k.
    EXPECT_EQ(idx_of(eps.slots[0]), i.id);
    EXPECT_EQ(idx_of(eps.slots[1]), j.id);
    EXPECT_EQ(idx_of(eps.slots[2]), idx_of(evec.slots[0]));
    EXPECT_EQ(evec.slots[0].slot.level, Level::Lower);
}

TEST(SimplifyBasisCross, ObliqueCarriesVolumeAndContravariantVector)
{
    // e_i × e_j → √g ε_{ijk} e^k: a √g (scalar triple product) factor, and the
    // output basis vector is contravariant (upper).
    Context ctx;
    auto b = oblique_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* cr =
        make_cross(ctx, b.covariant_vector(ctx, i), b.covariant_vector(ctx, j));

    auto const* tp =
        std::get_if<TensorProduct>(&simplify_basis_cross(ctx, cr, b)->node);
    ASSERT_NE(tp, nullptr);
    // result = √g ⊗ (ε ⊗ e^k); the √g factor is the volume e_0·(e_1×e_2).
    EXPECT_TRUE(std::holds_alternative<Dot>(tp->left->node));
    auto const* inner = std::get_if<TensorProduct>(&tp->right->node);
    ASSERT_NE(inner, nullptr);
    auto const& evec = std::get<TensorObject>(inner->right->node);
    EXPECT_EQ(evec.name.v.view(), "e");
    EXPECT_EQ(evec.slots[0].slot.level, Level::Upper);
}

TEST(SimplifyBasisCross, OrthonormalVolumeIsPlusOne)
{
    // A right-handed orthonormal basis has √g = +1 (the scalar literal 1).
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* vol = b.volume();
    ASSERT_NE(vol, nullptr);
    auto const* s = std::get_if<ScalarLiteral>(&vol->node);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->value, Rational{1});
}

TEST(SimplifyBasisCross, LeftHandedFlipsSign)
{
    // A left-handed orthonormal basis: √g = -1, so e_i × e_j = -ε_{ijk} e_k.
    Context ctx;
    auto b = make_orthonormal_basis(
        ctx,
        space_3d(),
        {vec(ctx, "i"), vec(ctx, "j"), vec(ctx, "k")},
        make_tensor_name("e"),
        Handedness::Left);
    auto const* vol = b.volume();
    auto const* s = std::get_if<ScalarLiteral>(&vol->node);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->value, Rational{-1});

    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* res = simplify_basis_cross(
        ctx,
        make_cross(ctx, b.covariant_vector(ctx, i), b.covariant_vector(ctx, j)),
        b);
    // The -1 volume rides along as the leading factor: -1 ⊗ (ε ⊗ e_k).
    auto const* tp = std::get_if<TensorProduct>(&res->node);
    ASSERT_NE(tp, nullptr);
    auto const* lead = std::get_if<ScalarLiteral>(&tp->left->node);
    ASSERT_NE(lead, nullptr);
    EXPECT_EQ(lead->value, Rational{-1});
}

TEST(SimplifyBasisCross, NonBasisCrossUnchanged)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* c = make_tensor_object(ctx, make_tensor_name("c"), {}, 1);
    auto const* cr = make_cross(ctx, a, c);
    EXPECT_EQ(simplify_basis_cross(ctx, cr, b), cr);
}

TEST(SimplifyBasisCross, ContravariantInputUnchanged)
{
    // e^i × e^j (both upper) is not handled yet → left unchanged.
    Context ctx;
    auto b = oblique_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* cr = make_cross(
        ctx, b.contravariant_vector(ctx, i), b.contravariant_vector(ctx, j));
    EXPECT_EQ(simplify_basis_cross(ctx, cr, b), cr);
}

// ---- reassemble (fold-back) --------------------------------------------

TEST(Reassemble, VectorRoundTrip)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);

    auto const* expanded = steps::canonicalize(
        ctx, expand_in_basis(ctx, a, b, Variance::Covariant));
    auto const* back = reassemble(ctx, expanded, b);
    EXPECT_TRUE(structural_eq(back, a));
}

TEST(Reassemble, Rank2RoundTrip)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"), {}, 2);

    auto const* expanded = steps::canonicalize(
        ctx, expand_in_basis(ctx, A, b, Variance::Covariant));
    auto const* back = reassemble(ctx, expanded, b);
    EXPECT_TRUE(structural_eq(back, A));
}

TEST(Reassemble, TwoVectorsFoldIndividuallyIntoDyad)
{
    // u ⊗ v expands to u_i v_j e_i e_j (two distinct coordinate vectors).  Each
    // coordinate vector reassembles on its own, so the dyad reappears without
    // any special dyad-assembly — the same mechanism would handle u⊗v⊗w.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* u = make_tensor_object(ctx, make_tensor_name("u"), {}, 1);
    auto const* v = make_tensor_object(ctx, make_tensor_name("v"), {}, 1);
    auto const* dyad = make_tensor_product(ctx, u, v);

    auto const* expanded = steps::canonicalize(
        ctx, expand_in_basis(ctx, dyad, b, Variance::Covariant));
    EXPECT_TRUE(structural_eq(reassemble(ctx, expanded, b), dyad));
}

TEST(Reassemble, ContractedCoordsFoldToDot)
{
    // Two coordinate components sharing a summed index, with no basis vector,
    // reassemble to the invariant dot: Σ_i u_i v_i → u · v.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* u = make_tensor_object(ctx, make_tensor_name("u"), {}, 1);
    auto const* v = make_tensor_object(ctx, make_tensor_name("v"), {}, 1);

    // u · v expanded and reduced to component form u_i v_i.
    auto const* comp = steps::contract_delta(
        ctx,
        simplify_basis_dot(
            ctx,
            expand_in_basis(ctx, make_dot(ctx, u, v), b, Variance::Covariant),
            b));
    auto const* back = reassemble(ctx, steps::canonicalize(ctx, comp), b);
    EXPECT_TRUE(algebraic_eq(ctx, back, make_dot(ctx, u, v)));
}

TEST(Reassemble, ContravariantRoundTrip)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);

    auto const* expanded = steps::canonicalize(
        ctx, expand_in_basis(ctx, a, b, Variance::Contravariant));
    EXPECT_TRUE(structural_eq(reassemble(ctx, expanded, b), a));
}

TEST(Reassemble, NonExpansionUnchanged)
{
    // A lone Kronecker trace Σ_i δ_{ii} is not a basis expansion.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto const* trace = make_explicit_sum(
        ctx,
        i,
        make_delta(
            ctx,
            Realm::Orthonormal,
            space_3d(),
            Level::Lower,
            Level::Lower,
            IndexAssoc{i},
            IndexAssoc{i}));
    EXPECT_EQ(reassemble(ctx, trace, b), trace);
}

TEST(Reassemble, NoSumUnchanged)
{
    // No surrounding sum: nothing to fold.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    EXPECT_EQ(reassemble(ctx, a, b), a);
}

TEST(Reassemble, ForeignBasisUnchanged)
{
    // An expansion in the e-basis is not recognized by the g-basis.
    Context ctx;
    auto e_basis = wcs_basis(ctx);
    auto g_basis = make_orthonormal_basis(
        ctx,
        space_3d(),
        {vec(ctx, "p"), vec(ctx, "q"), vec(ctx, "r")},
        make_tensor_name("g"));
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* expanded = steps::canonicalize(
        ctx, expand_in_basis(ctx, a, e_basis, Variance::Covariant));
    EXPECT_EQ(reassemble(ctx, expanded, g_basis), expanded);
}

TEST(Reassemble, IdentityRoundTrip)
{
    // I  →  Σ_i e_i ⊗ e^i  →  I.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* I = make_identity(ctx);
    auto const* ex = steps::canonicalize(
        ctx, expand_in_basis(ctx, I, b, Variance::Covariant));
    EXPECT_TRUE(structural_eq(reassemble(ctx, ex, b), I));
}

// Completeness, shape A: Σ_i (a·e_i) e_i = a·I = a.
TEST(ReassembleCompleteness, ContractionFoldsToVector)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto const* e_i = b.covariant_vector(ctx, i);
    auto const* term = make_explicit_sum(
        ctx, i, make_tensor_product(ctx, make_dot(ctx, a, e_i), e_i));
    auto const* back = steps::canonicalize(
        ctx, reassemble_completeness(ctx, steps::canonicalize(ctx, term), b));
    EXPECT_TRUE(structural_eq(back, a));
}

// Completeness, shape A inside a dyad: Σ_i (a·e_i)(b ⊗ e_i) = b ⊗ a — the leg
// the cross product writes a onto is the RIGHT one (guards against transpose).
TEST(ReassembleCompleteness, ContractionFoldsRightLegOfDyad)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* av = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* bv = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto const* e_i = b.covariant_vector(ctx, i);
    auto const* term = make_explicit_sum(
        ctx,
        i,
        make_tensor_product(
            ctx, make_dot(ctx, av, e_i), make_tensor_product(ctx, bv, e_i)));
    auto const* back = steps::canonicalize(
        ctx, reassemble_completeness(ctx, steps::canonicalize(ctx, term), b));
    EXPECT_TRUE(structural_eq(back, make_tensor_product(ctx, bv, av))); // b ⊗ a
    EXPECT_FALSE(structural_eq(back, make_tensor_product(ctx, av, bv))); // not
                                                                         // a ⊗
                                                                         // b
}

// Completeness, shape B with a scalar coefficient: Σ_i (a·b) e_i⊗e_i = (a·b) I.
TEST(ReassembleCompleteness, ScaledResolutionFoldsToIdentity)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* av = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* bv = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto const* e_i = b.covariant_vector(ctx, i);
    auto const* term = make_explicit_sum(
        ctx,
        i,
        make_tensor_product(
            ctx, make_dot(ctx, av, bv), make_tensor_product(ctx, e_i, e_i)));
    auto const* back = steps::canonicalize(
        ctx, reassemble_completeness(ctx, steps::canonicalize(ctx, term), b));
    auto const* want = steps::canonicalize(
        ctx,
        make_tensor_product(ctx, make_dot(ctx, av, bv), make_identity(ctx)));
    EXPECT_TRUE(structural_eq(back, want));
}

// Completeness, shape A with a rank-2 X: Σ_i (T·e_i) ⊗ e_i = T·I = T.  The dot
// contracts T's last slot and the leg sits on that (right) side, so the legs
// reassemble to T, not Tᵀ.
TEST(ReassembleCompleteness, Rank2ContractionFoldsToTensor)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* T = make_tensor_object(ctx, make_tensor_name("T"), {}, 2);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto const* e_i = b.covariant_vector(ctx, i);
    auto const* term = make_explicit_sum(
        ctx, i, make_tensor_product(ctx, make_dot(ctx, T, e_i), e_i));
    auto const* back = steps::canonicalize(
        ctx, reassemble_completeness(ctx, steps::canonicalize(ctx, term), b));
    EXPECT_TRUE(structural_eq(back, T));
}

// Mirror of the above: the leg on the LEFT with the dot contracting T's first
// slot, Σ_i e_i ⊗ (e_i·T) = I·T = T.
TEST(ReassembleCompleteness, Rank2ContractionLeftLegFoldsToTensor)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* T = make_tensor_object(ctx, make_tensor_name("T"), {}, 2);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto const* e_i = b.covariant_vector(ctx, i);
    auto const* term = make_explicit_sum(
        ctx, i, make_tensor_product(ctx, e_i, make_dot(ctx, e_i, T)));
    auto const* back = steps::canonicalize(
        ctx, reassemble_completeness(ctx, steps::canonicalize(ctx, term), b));
    EXPECT_TRUE(structural_eq(back, T));
}

// A non-scalar factor between the dot and the leg blocks the fold: there is no
// atomic direct-notation form for Σ_i (T·e_i) ⊗ b ⊗ e_i (= T_{ji} e_j⊗b⊗e_i),
// so the step refuses rather than mis-reassemble.
TEST(ReassembleCompleteness, Rank2NonScalarBetweenIsNoOp)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* T = make_tensor_object(ctx, make_tensor_name("T"), {}, 2);
    auto const* bv = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto const* e_i = b.covariant_vector(ctx, i);
    auto const* term = steps::canonicalize(
        ctx,
        make_explicit_sum(
            ctx,
            i,
            make_tensor_product(
                ctx, make_dot(ctx, T, e_i), make_tensor_product(ctx, bv, e_i))));
    EXPECT_EQ(reassemble_completeness(ctx, term, b), term);
}

// Wrong-side guard: Σ_i (e_i·T) ⊗ e_i would reassemble to Tᵀ (the dot contracts
// T's FIRST slot but the leg is on the right), which the step has no atomic
// form for, so it refuses — and in particular must not silently yield T.
TEST(ReassembleCompleteness, Rank2WrongSideIsNoOp)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* T = make_tensor_object(ctx, make_tensor_name("T"), {}, 2);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto const* e_i = b.covariant_vector(ctx, i);
    auto const* term = steps::canonicalize(
        ctx,
        make_explicit_sum(
            ctx, i, make_tensor_product(ctx, make_dot(ctx, e_i, T), e_i)));
    auto const* back = reassemble_completeness(ctx, term, b);
    EXPECT_EQ(back, term);
    EXPECT_FALSE(structural_eq(steps::canonicalize(ctx, back), T));
}

// No completeness pattern (a lone resolution with no contraction or coefficient
// is reassemble's job, not this step's): left unchanged.
TEST(ReassembleCompleteness, PlainResolutionUnchanged)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto const* e_i = b.covariant_vector(ctx, i);
    auto const* term = steps::canonicalize(
        ctx, make_explicit_sum(ctx, i, make_tensor_product(ctx, e_i, e_i)));
    // Σ_i e_i⊗e_i has two bare legs and no scalar coefficient: shape B with an
    // empty coefficient still folds it to I (the resolution of identity).
    EXPECT_TRUE(structural_eq(
        reassemble_completeness(ctx, term, b), make_identity(ctx)));
}

// A genuine non-pattern (Kronecker trace) is a no-op.
TEST(ReassembleCompleteness, NonPatternUnchanged)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto const* trace = make_explicit_sum(
        ctx,
        i,
        make_delta(
            ctx,
            Realm::Orthonormal,
            space_3d(),
            Level::Lower,
            Level::Lower,
            IndexAssoc{i},
            IndexAssoc{i}));
    EXPECT_EQ(reassemble_completeness(ctx, trace, b), trace);
}
