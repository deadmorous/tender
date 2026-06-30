#include <tender/basis.hpp>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/identity.hpp>
#include <tender/index_space.hpp>
#include <tender/rewrite.hpp>

#include <gtest/gtest.h>

#include <set>
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

// A rank-0 coordinate component name_{ids…} (the way expand_in_basis emits
// one): orthonormal lower slots carrying the given countable indices, tagged
// with the basis id so reassemble recognizes them as `b`'s (vibe 000067).
auto coord(
    Context& ctx,
    Basis const& b,
    char const* name,
    std::vector<CountableIndex> ids) -> Expr const*
{
    std::vector<SlotBinding> slots;
    for (auto id: ids)
        slots.push_back(SlotBinding{
            IndexSlot{
                Level::Lower, Realm::Orthonormal, space_3d(), b.basis_id()},
            IndexAssoc{id}});
    return make_tensor_object(ctx, make_tensor_name(name), std::move(slots), 0);
}

// Expand to symbolic coordinate components and contract the basis dots to
// deltas — grounds tr(B), Bᵀ and dyads to comparable component sums.
auto to_components(Context& ctx, Basis const& b, Expr const* x) -> Expr const*
{
    x = expand_in_basis(ctx, x, b, Variance::Covariant);
    for (int i = 0; i < 4; ++i)
    {
        x = simplify_basis_dot(ctx, x, b);
        x = steps::contract_delta(ctx, x);
    }
    return steps::canonicalize(ctx, x);
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
        // Coordinates carry the basis tag (vibe 000067); the δ stays neutral.
        return make_tensor_object(
            ctx,
            make_tensor_name(nm),
            {SlotBinding{
                IndexSlot{
                    Level::Lower, Realm::Orthonormal, space_3d(), b.basis_id()},
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

// ---- basis registry + id stamping (vibe 000067, increment 2) --------------

TEST(BasisRegistry, FactoriesAssignDistinctIdsAndResolve)
{
    Context ctx;
    auto b1 = wcs_basis(ctx);
    auto b2 = oblique_basis(ctx);
    EXPECT_GT(b1.basis_id(), 0);
    EXPECT_GT(b2.basis_id(), 0);
    EXPECT_NE(b1.basis_id(), b2.basis_id());
    ASSERT_NE(ctx.basis(b1.basis_id()), nullptr);
    EXPECT_EQ(ctx.basis(b1.basis_id())->basis_id(), b1.basis_id());
    // 0 (basis-unaware) and out-of-range resolve to nullptr.
    EXPECT_EQ(ctx.basis(0), nullptr);
    EXPECT_EQ(ctx.basis(9999), nullptr);
}

TEST(BasisRegistry, EmissionsAndExpansionCarryBasisId)
{
    Context ctx;
    auto b = wcs_basis(ctx);

    // A symbolic basis vector is stamped with the basis id.
    CountableIndex const i{ctx.alloc_index_id()};
    auto const& cov = std::get<TensorObject>(b.covariant_vector(ctx, i)->node);
    EXPECT_EQ(cov.slots.at(0).slot.basis_id, b.basis_id());

    // expand_in_basis stamps every emitted slot (coordinate and vector) with
    // it.
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* exp = expand_in_basis(ctx, a, b, Variance::Covariant);
    std::set<int> ids;
    rewrite_tree(
        ctx,
        exp,
        [&](Context&, Expr const* n) -> Expr const*
        {
            if (auto const* to = std::get_if<TensorObject>(&n->node))
                for (auto const& sb: to->slots)
                    ids.insert(sb.slot.basis_id);
            return n;
        });
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(*ids.begin(), b.basis_id());
}

TEST(BasisRegistry, DifferentBasesProduceDistinctExpansions)
{
    // The same invariant expanded in two bases differs only by basis_id, so the
    // two coordinate forms are not algebraically equal (vibe 000067).
    Context ctx;
    auto b1 = wcs_basis(ctx);
    auto b2 = make_orthonormal_basis(
        ctx, space_3d(), {vec(ctx, "p"), vec(ctx, "q"), vec(ctx, "r")});
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* e1 = expand_in_basis(ctx, a, b1, Variance::Covariant);
    auto const* e2 = expand_in_basis(ctx, a, b2, Variance::Covariant);
    EXPECT_FALSE(algebraic_eq(ctx, e1, e2));
}

// ---- basis-aware step filtering (vibe 000067, increment 3) -----------------

namespace
{
// A second 3D orthonormal basis (p, q, r), distinct from wcs_basis.
auto other_basis(Context& ctx) -> Basis
{
    return make_orthonormal_basis(
        ctx, space_3d(), {vec(ctx, "p"), vec(ctx, "q"), vec(ctx, "r")});
}

// True if a tensor named `name` appears anywhere in the tree.
auto contains_named(Context& ctx, Expr const* e, char const* name) -> bool
{
    bool found = false;
    rewrite_tree(
        ctx,
        e,
        [&](Context&, Expr const* n) -> Expr const*
        {
            if (auto const* t = std::get_if<TensorObject>(&n->node))
                if (t->name.v.view() == name)
                    found = true;
            return n;
        });
    return found;
}
} // namespace

TEST(BasisFilter, DotContractsOnlyWithinOneBasis)
{
    // e_i^A · e_j^A → δ, but e_i^A · e_j^B (the overlap of two frames) is left
    // untouched by simplify_basis_dot(A) (vibe 000067).
    Context ctx;
    auto a = wcs_basis(ctx);
    auto b = other_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};

    // Same basis: contracts to a δ (top node no longer a Dot).
    auto const* same = simplify_basis_dot(
        ctx,
        make_dot(ctx, a.covariant_vector(ctx, i), a.covariant_vector(ctx, j)),
        a);
    EXPECT_FALSE(std::holds_alternative<Dot>(same->node));

    // Cross basis: the overlap is left as the dot (no δ produced).
    auto const* cross = simplify_basis_dot(
        ctx,
        make_dot(ctx, a.covariant_vector(ctx, i), b.covariant_vector(ctx, j)),
        a);
    EXPECT_TRUE(std::holds_alternative<Dot>(cross->node));
}

TEST(BasisFilter, CrossReducesOnlyWithinOneBasis)
{
    // e_i^A × e_j^B is left as a Cross by simplify_basis_cross(A).
    Context ctx;
    auto a = wcs_basis(ctx);
    auto b = other_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};

    auto const* same = simplify_basis_cross(
        ctx,
        make_cross(ctx, a.covariant_vector(ctx, i), a.covariant_vector(ctx, j)),
        a);
    EXPECT_FALSE(std::holds_alternative<Cross>(same->node));

    auto const* cross = simplify_basis_cross(
        ctx,
        make_cross(ctx, a.covariant_vector(ctx, i), b.covariant_vector(ctx, j)),
        a);
    EXPECT_TRUE(std::holds_alternative<Cross>(cross->node));
}

TEST(BasisFilter, ReassembleIgnoresForeignBasis)
{
    // An expansion in basis B is not reassembled by basis A; the correct basis
    // does fold it back (vibe 000067).
    Context ctx;
    auto a = wcs_basis(ctx);
    auto b = other_basis(ctx);
    auto const* v = make_tensor_object(ctx, make_tensor_name("v"), {}, 1);
    auto const* expB = steps::canonicalize(
        ctx, expand_in_basis(ctx, v, b, Variance::Covariant));

    // Foreign basis: no fold (still carries the basis vector "e").
    auto const* wrong = reassemble(ctx, expB, a);
    EXPECT_TRUE(contains_named(ctx, wrong, "e"));
    EXPECT_FALSE(structural_eq(steps::canonicalize(ctx, wrong), v));

    // Right basis: folds back to the invariant v.
    EXPECT_TRUE(structural_eq(reassemble(ctx, expB, b), v));
}

TEST(BasisFilter, TwoPointCoordinateNotReassembled)
{
    // F_{iJ} with i in A and J in B is a two-point coordinate, not a clean A
    // reassembly: reassemble(A) leaves the basis vectors unfolded.
    Context ctx;
    auto a = wcs_basis(ctx);
    auto b = other_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* F = make_tensor_object(
        ctx,
        make_tensor_name("F"),
        {SlotBinding{
             IndexSlot{
                 Level::Lower, Realm::Orthonormal, space_3d(), a.basis_id()},
             IndexAssoc{i}},
         SlotBinding{
             IndexSlot{
                 Level::Lower, Realm::Orthonormal, space_3d(), b.basis_id()},
             IndexAssoc{j}}},
        0);
    auto const* term = make_explicit_sum(
        ctx,
        i,
        make_explicit_sum(
            ctx,
            j,
            make_tensor_product(
                ctx,
                F,
                make_tensor_product(
                    ctx,
                    a.covariant_vector(ctx, i),
                    a.covariant_vector(ctx, j)))));

    // F is not an A coordinate (its J slot is B), so the A vectors stay.
    auto const* res = steps::canonicalize(ctx, reassemble(ctx, term, a));
    EXPECT_TRUE(contains_named(ctx, res, "e"));
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

TEST(Reassemble, TraceFold)
{
    // Σ_k B_{kk} (a coordinate component with a repeated summed index) → tr(B),
    // the way the index appears after an ε-pair contraction collapses two
    // slots.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto k = CountableIndex{ctx.alloc_index_id()};
    auto const* term = make_explicit_sum(ctx, k, coord(ctx, b, "B", {k, k}));

    auto const* back = steps::canonicalize(ctx, reassemble(ctx, term, b));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"), {}, 2);
    EXPECT_TRUE(structural_eq(back, make_trace(ctx, B)));
}

TEST(Reassemble, TraceFoldInBiggerTerm)
{
    // The trace folds even as one factor of a larger product: Σ_k B_{kk} u ⊗ v
    // → tr(B) u ⊗ v (the dyad u ⊗ v is carried through untouched).
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* u = vec(ctx, "u");
    auto const* v = vec(ctx, "v");
    auto k = CountableIndex{ctx.alloc_index_id()};
    auto const* term = make_explicit_sum(
        ctx,
        k,
        make_tensor_product(
            ctx, coord(ctx, b, "B", {k, k}), make_tensor_product(ctx, u, v)));

    auto const* back = steps::canonicalize(ctx, reassemble(ctx, term, b));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"), {}, 2);
    auto const* want = steps::canonicalize(
        ctx,
        make_tensor_product(
            ctx, make_trace(ctx, B), make_tensor_product(ctx, u, v)));
    EXPECT_TRUE(structural_eq(back, want));
}

TEST(Reassemble, BilinearFold)
{
    // Σ_{ij} B_{ij} a_i c_j → a contraction of B by two coordinate vectors on
    // both legs, a scalar invariant (a·B·c, here read out as (Bᵀ·a)·c).
    Context ctx;
    auto b = wcs_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* term = make_explicit_sum(
        ctx,
        i,
        make_explicit_sum(
            ctx,
            j,
            make_tensor_product(
                ctx,
                coord(ctx, b, "B", {i, j}),
                make_tensor_product(
                    ctx, coord(ctx, b, "a", {i}), coord(ctx, b, "c", {j})))));

    // Contracting i (B's first leg, with a) transposes B, then contracting j
    // with c gives the scalar c·(Bᵀ·a) — numerically B_ij a_i c_j = a·B·c.
    auto const* back = steps::canonicalize(ctx, reassemble(ctx, term, b));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"), {}, 2);
    auto const* want = steps::canonicalize(
        ctx,
        make_dot(
            ctx,
            vec(ctx, "c"),
            make_dot(ctx, make_transpose(ctx, B), vec(ctx, "a"))));
    EXPECT_TRUE(structural_eq(back, want));
}

TEST(Reassemble, TensorTensorContraction)
{
    // Σ_{ijk} B_{ij} D_{jk} e_i e_k → B·D: two rank-2 coordinate tensors share
    // a summed index and contract into one, whose two free legs reassemble (in
    // basis order) to the rank-2 invariant B·D.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto k = CountableIndex{ctx.alloc_index_id()};
    auto const* term = make_explicit_sum(
        ctx,
        i,
        make_explicit_sum(
            ctx,
            j,
            make_explicit_sum(
                ctx,
                k,
                make_tensor_product(
                    ctx,
                    make_tensor_product(
                        ctx,
                        coord(ctx, b, "B", {i, j}),
                        coord(ctx, b, "D", {j, k})),
                    make_tensor_product(
                        ctx,
                        b.covariant_vector(ctx, i),
                        b.covariant_vector(ctx, k))))));

    auto const* back = steps::canonicalize(ctx, reassemble(ctx, term, b));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"), {}, 2);
    auto const* D = make_tensor_object(ctx, make_tensor_name("D"), {}, 2);
    EXPECT_TRUE(
        structural_eq(back, steps::canonicalize(ctx, make_dot(ctx, B, D))));
}

TEST(Reassemble, Rank2TransposeRoundTrip)
{
    // Bᵀ expands to B_{ij} e_j e_i (basis order reversed against the slots),
    // and the leg-order/basis-order mismatch folds back to the transpose, not
    // B.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* Bt = make_transpose(
        ctx, make_tensor_object(ctx, make_tensor_name("B"), {}, 2));

    auto const* expanded = steps::canonicalize(
        ctx, expand_in_basis(ctx, Bt, b, Variance::Covariant));
    EXPECT_TRUE(structural_eq(reassemble(ctx, expanded, b), Bt));
}

TEST(Reassemble, CompositeDyadLegFold)
{
    // c ⊗ (B·a): one dyad leg is itself a tensor·vector contraction.  Its
    // coordinate (B·a)_k pairs with a basis vector and reassembles to the
    // invariant B·a, sitting beside the already-invariant c.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* a = vec(ctx, "a");
    auto const* c = vec(ctx, "c");
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"), {}, 2);
    auto const* dyad = make_tensor_product(ctx, c, make_dot(ctx, B, a));

    auto const* back = steps::canonicalize(
        ctx, reassemble(ctx, to_components(ctx, b, dyad), b));
    EXPECT_TRUE(structural_eq(back, steps::canonicalize(ctx, dyad)));
}

TEST(Reassemble, SignBetweenSumBinders)
{
    // A subtracted term may carry its sign as a Negate sitting *between* its
    // two summation binders: Σ_j −(Σ_i B_{ji} c_j e_i ⊗ a).  The interleaved
    // ExplicitSum/Negate peel collects both binders past the sign, so this
    // folds to −(Bᵀ·c)⊗a (B_{ji} c_j is the i-th component of Bᵀ·c; e_i
    // realizes that leg).  Note: standalone, reassemble's self-prep
    // canonicalize already adjacency-normalizes Σ_j −Σ_i → Σ_j Σ_i −, so this
    // shape folds with or without the interleaved peel; the peel is what
    // matters when the term is an addend inside a larger Sum (see
    // BasisFeasibility.CrossTensorCrossOnePass).
    Context ctx;
    auto b = wcs_basis(ctx);
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* a = vec(ctx, "a");
    auto const* term = make_explicit_sum(
        ctx,
        j,
        make_negate(
            ctx,
            make_explicit_sum(
                ctx,
                i,
                make_tensor_product(
                    ctx,
                    make_tensor_product(
                        ctx,
                        make_tensor_product(
                            ctx,
                            coord(ctx, b, "B", {j, i}),
                            coord(ctx, b, "c", {j})),
                        b.covariant_vector(ctx, i)),
                    a))));

    auto const* back = steps::canonicalize(ctx, reassemble(ctx, term, b));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"), {}, 2);
    auto const* c = make_tensor_object(ctx, make_tensor_name("c"), {}, 1);
    auto const* want = make_negate(
        ctx,
        make_tensor_product(ctx, make_dot(ctx, make_transpose(ctx, B), c), a));
    EXPECT_TRUE(structural_eq(back, steps::canonicalize(ctx, want)));
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
    // A lone Kronecker trace Σ_i δ_{ii} is not a basis expansion: no reassembly
    // fold fires.  reassemble still self-preps (canonicalize) and strips the
    // materialized Σ on return (vibe 000064 #3/#6), so the result is exactly
    // the prepared input — not folded into any invariant.
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
    auto const* prepped =
        steps::implicitize(ctx, steps::canonicalize(ctx, trace));
    EXPECT_TRUE(structural_eq(reassemble(ctx, trace, b), prepped));
}

TEST(Reassemble, SurfacesPrepCancellation)
{
    // vibe 000064 #6: reassemble's self-prep canonicalize can simplify the
    // input (here cancel Y − Y) even when no reassembly fold then fires.  The
    // old no-op guard discarded that and returned the raw input; reassemble
    // must surface the prepared (simplified) result instead.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* X = make_tensor_product(ctx, vec(ctx, "b"), vec(ctx, "c"));
    auto const* Y = make_tensor_product(ctx, vec(ctx, "a"), vec(ctx, "a"));
    auto const* e = make_difference(ctx, make_sum(ctx, X, Y), Y);

    auto const* got = reassemble(ctx, e, b);
    EXPECT_TRUE(structural_eq(got, steps::canonicalize(ctx, X)));
    EXPECT_FALSE(structural_eq(got, e)); // really simplified, not the raw input
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

// A genuine non-pattern (Kronecker trace): no completeness fold fires, so the
// result is just the self-prep (canonicalize + implicitize) of the input.
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
    auto const* prepped =
        steps::implicitize(ctx, steps::canonicalize(ctx, trace));
    EXPECT_TRUE(structural_eq(reassemble_completeness(ctx, trace, b), prepped));
}

TEST(Reassemble, AlsoDoesCompletenessFold)
{
    // vibe 000068 P2: reassemble now finishes the resolution-of-identity-with-
    // contraction on its own.  I·e_1 = (e_i ⊗ e_i)·e_1 reassembles straight to
    // e_1 (= basis(0)) — previously only reassemble_completeness did this.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* I = make_identity(ctx);
    auto const* term = make_dot(
        ctx, expand_in_basis(ctx, I, b, Variance::Covariant), b.basis(0));
    EXPECT_TRUE(structural_eq(reassemble(ctx, term, b), b.basis(0)));
}

// ---- concrete resolution of identity (vibe 000070 Phase 0) ----------------

namespace
{
// The concrete dyad sum Σ_k c·u_k⊗u_k over `b`'s frame vectors (the shape the
// differential operators emit), with an optional common scalar coefficient.
auto dyad_sum(Context& ctx, Basis const& b, Expr const* coeff = nullptr)
    -> Expr const*
{
    Expr const* s = nullptr;
    for (int k = 0; k < b.dim(); ++k)
    {
        Expr const* d = make_tensor_product(ctx, b.basis(k), b.basis(k));
        if (coeff)
            d = make_tensor_product(ctx, coeff, d);
        s = s ? make_sum(ctx, s, d) : d;
    }
    return s;
}
} // namespace

// i⊗i + j⊗j + k⊗k folds back to the identity tensor I (the headline ∇R = I).
TEST(ResolutionOfIdentity, ConcreteDyadSumFoldsToIdentity)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* got = fold_resolution_of_identity(ctx, dyad_sum(ctx, b), b);
    EXPECT_TRUE(structural_eq(got, make_identity(ctx)));
}

// A common scalar coefficient survives the fold: 2(i⊗i + …) → 2 I, and the
// negated group −(i⊗i + …) → −I.
TEST(ResolutionOfIdentity, ScaledAndNegatedGroupsFold)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* two = make_scalar(ctx, Rational{2});
    auto const* scaled =
        fold_resolution_of_identity(ctx, dyad_sum(ctx, b, two), b);
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, scaled),
        steps::canonicalize(
            ctx, make_tensor_product(ctx, two, make_identity(ctx)))));

    auto const* neg = make_negate(ctx, dyad_sum(ctx, b));
    auto const* folded = fold_resolution_of_identity(ctx, neg, b);
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, folded),
        steps::canonicalize(ctx, make_negate(ctx, make_identity(ctx)))));
}

// An incomplete group (a direction missing) does not fold — completeness needs
// every frame vector present.
TEST(ResolutionOfIdentity, IncompleteGroupUnchanged)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* partial = make_sum(
        ctx,
        make_tensor_product(ctx, b.basis(0), b.basis(0)),
        make_tensor_product(ctx, b.basis(1), b.basis(1))); // no k⊗k
    auto const* got = fold_resolution_of_identity(ctx, partial, b);
    EXPECT_FALSE(
        structural_eq(steps::canonicalize(ctx, got), make_identity(ctx)));
}

// A non-orthonormal basis has no resolution of identity: the fold is a no-op
// and the forward expansion throws.
TEST(ResolutionOfIdentity, ObliqueBasisIsNoOp)
{
    Context ctx;
    auto i = vec(ctx, "i");
    auto j = vec(ctx, "j");
    auto k = vec(ctx, "k");
    auto ob = make_oblique_basis(ctx, space_3d(), {i, j, k});
    auto const* sum = dyad_sum(ctx, ob);
    EXPECT_EQ(fold_resolution_of_identity(ctx, sum, ob), sum);
    EXPECT_THROW(
        (void)expand_identity(ctx, make_identity(ctx), ob),
        std::invalid_argument);
}

// expand_identity is the inverse direction: I → Σ_k u_k⊗u_k, and folding it
// back returns I (round-trip).
TEST(ResolutionOfIdentity, ExpandThenFoldRoundTrips)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* expanded = expand_identity(ctx, make_identity(ctx), b);
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, expanded),
        steps::canonicalize(ctx, dyad_sum(ctx, b))));
    EXPECT_TRUE(structural_eq(
        fold_resolution_of_identity(ctx, expanded, b), make_identity(ctx)));
}

// A complete dyad group inside a Difference still folds (the fold gathers
// signed addends across subtraction): (i⊗i+j⊗j+k⊗k) − X folds the group to I −
// X.
TEST(ResolutionOfIdentity, FoldsAcrossDifference)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* X = make_tensor_object(ctx, make_tensor_name("X"), {}, 2);
    auto const* term = make_difference(ctx, dyad_sum(ctx, b), X);
    auto const* got = fold_resolution_of_identity(ctx, term, b);
    auto const* want =
        steps::canonicalize(ctx, make_difference(ctx, make_identity(ctx), X));
    EXPECT_TRUE(structural_eq(steps::canonicalize(ctx, got), want));
}

// Sum addends that are not c·u_k⊗u_k are parsed but rejected: a mixed dyad i⊗j
// (two distinct directions) and a triad i⊗i⊗a (a non-scalar extra leg) leave
// the fold a no-op when no complete group is present.
TEST(ResolutionOfIdentity, NonDyadAddendsIgnored)
{
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* mixed = make_tensor_product(ctx, b.basis(0), b.basis(1)); // i⊗j
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* triad = make_tensor_product(
        ctx, make_tensor_product(ctx, b.basis(0), b.basis(0)), a); // i⊗i⊗a
    // A Sum so fold_identity_dyads parses each addend (hits the reject paths);
    // no complete dyad group, so nothing folds and no I appears.
    auto const* sum = make_sum(ctx, mixed, triad);
    auto const* got = fold_resolution_of_identity(ctx, sum, b);
    EXPECT_FALSE(structural_eq(
        steps::canonicalize(ctx, got),
        steps::canonicalize(ctx, make_identity(ctx))));
}

// ---- concrete-direction basis dots (vibe 000068 P1/P3) --------------------

namespace
{
// The symbolic-concrete basis vector e_value — `e` + ConcreteIndex tagged with
// b — which is what a contracted concrete dot reduces to (it renders as the
// frame letter via vector_symbols, but is structurally distinct from the frame
// vector b.basis(k)).
auto e_dir(Context& ctx, Basis const& b, int value) -> Expr const*
{
    return make_tensor_object(
        ctx,
        make_tensor_name("e"),
        {SlotBinding{
            IndexSlot{
                Level::Lower, Realm::Orthonormal, space_3d(), b.basis_id()},
            ConcreteIndex{value}}},
        1);
}
} // namespace

TEST(BasisConcreteDot, DotWithFrameVectorContractsToDelta)
{
    // P1: (e_i ⊗ e_i)·e_1 now contracts — simplify_basis_dot yields e_i δ_{i1}
    // (the frame vector b.basis(0) is recognised as the concrete direction 1),
    // and contract_delta finishes it to the direction-1 vector e_1.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* I = make_identity(ctx);
    auto const* term = make_dot(
        ctx, expand_in_basis(ctx, I, b, Variance::Covariant), b.basis(0));
    auto const* done =
        steps::contract_delta(ctx, simplify_basis_dot(ctx, term, b));
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, done),
        steps::canonicalize(ctx, e_dir(ctx, b, 1))));
}

TEST(BasisConcreteDot, UnrolledDotEvaluatesConcretely)
{
    // P3: after unrolling, simplify_basis_dot turns each e_k·e_1 into δ_{k1},
    // which eval_delta_concrete reduces (1/0), leaving the direction-1 vector.
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* I = make_identity(ctx);
    auto const* term = steps::unroll_sums(
        ctx,
        make_dot(
            ctx, expand_in_basis(ctx, I, b, Variance::Covariant), b.basis(0)));
    auto const* done = steps::fold_arithmetic(
        ctx, steps::eval_delta_concrete(ctx, simplify_basis_dot(ctx, term, b)));
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, done),
        steps::canonicalize(ctx, e_dir(ctx, b, 1))));
}

TEST(BasisConcreteDot, FrameVectorSelfAndCrossDots)
{
    // The reverse-looked-up frame vectors contract among themselves: e_1·e_1 =
    // 1 and e_1·e_2 = 0 (concrete–concrete δ via eval_delta_concrete).
    Context ctx;
    auto b = wcs_basis(ctx);
    auto const* one = steps::eval_delta_concrete(
        ctx, simplify_basis_dot(ctx, make_dot(ctx, b.basis(0), b.basis(0)), b));
    auto const* zero = steps::eval_delta_concrete(
        ctx, simplify_basis_dot(ctx, make_dot(ctx, b.basis(0), b.basis(1)), b));
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, one), make_scalar(ctx, Rational{1})));
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, zero), make_scalar(ctx, Rational{0})));
}
