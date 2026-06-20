#include <tender/expr.hpp>

#include <gtest/gtest.h>

using namespace tender;

// ---- make_scalar -------------------------------------------------------

TEST(MakeScalar, StoresValue)
{
    Context ctx;
    auto* e = make_scalar(ctx, Rational{3, 2});
    ASSERT_NE(e, nullptr);
    ASSERT_TRUE(std::holds_alternative<ScalarLiteral>(e->node));
    EXPECT_EQ(std::get<ScalarLiteral>(e->node).value, (Rational{3, 2}));
}

// ---- make_tensor_object ------------------------------------------------

TEST(MakeTensorObject, AbstractNoSlots)
{
    Context ctx;
    auto* e = make_tensor_object(ctx, make_tensor_name("f"));
    ASSERT_NE(e, nullptr);
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(t.name.v.view(), "f");
    EXPECT_FALSE(t.rank.has_value());
    EXPECT_FALSE(t.traits.has_value());
    EXPECT_TRUE(t.slots.empty());
}

TEST(MakeTensorObject, AbstractWithRank)
{
    Context ctx;
    auto* e = make_tensor_object(ctx, make_tensor_name("T"), {}, 2);
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(t.rank, std::optional<int>{2});
    EXPECT_TRUE(t.slots.empty());
}

TEST(MakeTensorObject, SlotWithFilledIndex)
{
    Context ctx;
    auto id = ctx.alloc_index_id();
    std::vector<SlotBinding> slots = {SlotBinding{
        IndexSlot{Level::Upper, Realm::Oblique, space_3d()},
        CountableIndex{id}}};
    auto* e = make_tensor_object(ctx, make_tensor_name("v"), std::move(slots));
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(t.name.v.view(), "v");
    ASSERT_EQ(t.slots.size(), 1u);
    EXPECT_EQ(t.slots[0].slot.level, Level::Upper);
    EXPECT_EQ(t.slots[0].slot.realm, Realm::Oblique);
    ASSERT_TRUE(t.slots[0].index.has_value());
    EXPECT_EQ(std::get<CountableIndex>(*t.slots[0].index).id, id);
}

TEST(MakeTensorObject, SlotWithNullIndex)
{
    Context ctx;
    std::vector<SlotBinding> slots = {SlotBinding{
        IndexSlot{Level::Upper, Realm::Oblique, space_3d()}, std::nullopt}};
    auto* e = make_tensor_object(ctx, make_tensor_name("A"), std::move(slots));
    auto const& t = std::get<TensorObject>(e->node);
    ASSERT_EQ(t.slots.size(), 1u);
    EXPECT_FALSE(t.slots[0].index.has_value());
}

TEST(MakeTensorObject, SlotCountIndependentOfRank)
{
    // A rank-3 tensor with only 1 slot currently bound — valid by design.
    Context ctx;
    std::vector<SlotBinding> slots = {SlotBinding{
        IndexSlot{Level::Lower, Realm::Oblique, space_3d()},
        CountableIndex{ctx.alloc_index_id()}}};
    auto* e =
        make_tensor_object(ctx, make_tensor_name("T"), std::move(slots), 3);
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(*t.rank, 3);
    EXPECT_EQ(t.slots.size(), 1u);
}

// ---- Unary: make_negate ------------------------------------------------

TEST(MakeNegate, StoresOperand)
{
    Context ctx;
    auto* operand = make_scalar(ctx, Rational{1});
    auto* e = make_negate(ctx, operand);
    ASSERT_TRUE(std::holds_alternative<Negate>(e->node));
    EXPECT_EQ(std::get<Negate>(e->node).operand, operand);
}

// ---- Binary operations -------------------------------------------------

TEST(MakeBinary, SumStoresChildren)
{
    Context ctx;
    auto* l = make_scalar(ctx, Rational{1});
    auto* r = make_scalar(ctx, Rational{2});
    auto* e = make_sum(ctx, l, r);
    ASSERT_TRUE(std::holds_alternative<Sum>(e->node));
    auto const& s = std::get<Sum>(e->node);
    EXPECT_EQ(s.left, l);
    EXPECT_EQ(s.right, r);
}

TEST(MakeBinary, DifferenceStoresChildren)
{
    Context ctx;
    auto* e = make_difference(
        ctx, make_scalar(ctx, Rational{3}), make_scalar(ctx, Rational{1}));
    ASSERT_TRUE(std::holds_alternative<Difference>(e->node));
}

TEST(MakeBinary, TensorProductStoresChildren)
{
    Context ctx;
    auto* a = make_tensor_object(ctx, make_tensor_name("a"));
    auto* b = make_tensor_object(ctx, make_tensor_name("b"));
    auto* e = make_tensor_product(ctx, a, b);
    ASSERT_TRUE(std::holds_alternative<TensorProduct>(e->node));
    EXPECT_EQ(std::get<TensorProduct>(e->node).left, a);
    EXPECT_EQ(std::get<TensorProduct>(e->node).right, b);
}

TEST(MakeBinary, ScalarDivStoresChildren)
{
    Context ctx;
    auto* e = make_scalar_div(
        ctx, make_scalar(ctx, Rational{1}), make_scalar(ctx, Rational{2}));
    ASSERT_TRUE(std::holds_alternative<ScalarDiv>(e->node));
}

TEST(MakeBinary, DotStoresChildren)
{
    Context ctx;
    auto* e = make_dot(
        ctx,
        make_tensor_object(ctx, make_tensor_name("u")),
        make_tensor_object(ctx, make_tensor_name("v")));
    ASSERT_TRUE(std::holds_alternative<Dot>(e->node));
}

TEST(MakeBinary, DDotStoresChildren)
{
    Context ctx;
    auto* e = make_ddot(
        ctx,
        make_tensor_object(ctx, make_tensor_name("A")),
        make_tensor_object(ctx, make_tensor_name("B")));
    ASSERT_TRUE(std::holds_alternative<DDot>(e->node));
}

TEST(MakeBinary, DDotAltStoresChildren)
{
    Context ctx;
    auto* e = make_ddot_alt(
        ctx,
        make_tensor_object(ctx, make_tensor_name("A")),
        make_tensor_object(ctx, make_tensor_name("B")));
    ASSERT_TRUE(std::holds_alternative<DDotAlt>(e->node));
}

TEST(MakeBinary, CrossStoresChildren)
{
    Context ctx;
    auto* e = make_cross(
        ctx,
        make_tensor_object(ctx, make_tensor_name("u")),
        make_tensor_object(ctx, make_tensor_name("v")));
    ASSERT_TRUE(std::holds_alternative<Cross>(e->node));
}

// ---- Summation annotations ---------------------------------------------

TEST(MakeExplicitSum, ConcreteRangeNullBound)
{
    Context ctx;
    CountableIndex idx{ctx.alloc_index_id()};
    auto* body = make_scalar(ctx, Rational{1});
    auto* e = make_explicit_sum(ctx, idx, body);
    ASSERT_TRUE(std::holds_alternative<ExplicitSum>(e->node));
    auto const& es = std::get<ExplicitSum>(e->node);
    EXPECT_EQ(es.index.id, idx.id);
    EXPECT_EQ(es.body, body);
    EXPECT_EQ(es.bound, nullptr);
}

TEST(MakeExplicitSum, SymbolicBound)
{
    Context ctx;
    CountableIndex idx{ctx.alloc_index_id()};
    auto* body = make_scalar(ctx, Rational{1});
    auto* bound = make_tensor_object(ctx, make_tensor_name("N"));
    auto* e = make_explicit_sum(ctx, idx, body, bound);
    EXPECT_EQ(std::get<ExplicitSum>(e->node).bound, bound);
}

TEST(MakeNoSum, StoresIndexAndBody)
{
    Context ctx;
    CountableIndex idx{ctx.alloc_index_id()};
    auto* body = make_scalar(ctx, Rational{2});
    auto* e = make_no_sum(ctx, idx, body);
    ASSERT_TRUE(std::holds_alternative<NoSum>(e->node));
    auto const& ns = std::get<NoSum>(e->node);
    EXPECT_EQ(ns.index.id, idx.id);
    EXPECT_EQ(ns.body, body);
}

// ---- Well-known tensors: identity --------------------------------------

TEST(MakeIdentity, AbstractForm)
{
    Context ctx;
    auto* e = make_identity(ctx);
    ASSERT_NE(e, nullptr);
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(t.name.v.view(), "I");
    EXPECT_EQ(t.rank, std::optional<int>{2});
    ASSERT_TRUE(t.traits.has_value());
    EXPECT_EQ(
        t.traits->well_known,
        std::optional<WellKnownKind>{WellKnownKind::Identity});
    EXPECT_FALSE(t.traits->render_hints.any());
    EXPECT_TRUE(t.slots.empty());
}

// ---- Well-known tensors: Kronecker delta -------------------------------

TEST(MakeDelta, SlotLayout)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto* e = make_delta(
        ctx, Realm::Oblique, space_3d(), Level::Upper, Level::Lower, i, j);
    ASSERT_NE(e, nullptr);
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(t.name.v.view(), "\\delta");
    EXPECT_EQ(t.rank, std::optional<int>{0});
    ASSERT_TRUE(t.traits.has_value());
    EXPECT_EQ(
        t.traits->well_known,
        std::optional<WellKnownKind>{WellKnownKind::Delta});
    EXPECT_TRUE(
        t.traits->render_hints.contains(RenderHint::OmitVoidIndexPlaceholders));
    ASSERT_EQ(t.slots.size(), 2u);
    EXPECT_EQ(t.slots[0].slot.level, Level::Upper);
    EXPECT_EQ(t.slots[0].slot.realm, Realm::Oblique);
    EXPECT_EQ(t.slots[0].slot.space, space_3d());
    EXPECT_EQ(t.slots[1].slot.level, Level::Lower);
    ASSERT_TRUE(t.slots[0].index.has_value());
    ASSERT_TRUE(t.slots[1].index.has_value());
    EXPECT_EQ(std::get<CountableIndex>(*t.slots[0].index).id, i.id);
    EXPECT_EQ(std::get<CountableIndex>(*t.slots[1].index).id, j.id);
}

TEST(MakeDelta, BothLevelsUpper)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto* e = make_delta(
        ctx, Realm::Oblique, space_3d(), Level::Upper, Level::Upper, i, j);
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(t.slots[0].slot.level, Level::Upper);
    EXPECT_EQ(t.slots[1].slot.level, Level::Upper);
}

// ---- Well-known tensors: Levi-Civita -----------------------------------

TEST(MakeLeviCivita, Rank3)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto k = CountableIndex{ctx.alloc_index_id()};
    auto* e = make_levi_civita(
        ctx,
        Realm::Orthonormal,
        space_3d(),
        {Level::Lower, Level::Lower, Level::Lower},
        {i, j, k});
    ASSERT_NE(e, nullptr);
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(t.name.v.view(), "\\varepsilon");
    EXPECT_EQ(t.rank, std::optional<int>{0});
    ASSERT_TRUE(t.traits.has_value());
    EXPECT_EQ(
        t.traits->well_known,
        std::optional<WellKnownKind>{WellKnownKind::LeviCivita});
    ASSERT_EQ(t.slots.size(), 3u);
    for (auto const& sb: t.slots)
    {
        EXPECT_EQ(sb.slot.level, Level::Lower);
        EXPECT_EQ(sb.slot.realm, Realm::Orthonormal);
        EXPECT_EQ(sb.slot.space, space_3d());
        EXPECT_TRUE(sb.index.has_value());
    }
    EXPECT_EQ(std::get<CountableIndex>(*t.slots[0].index).id, i.id);
    EXPECT_EQ(std::get<CountableIndex>(*t.slots[1].index).id, j.id);
    EXPECT_EQ(std::get<CountableIndex>(*t.slots[2].index).id, k.id);
}

TEST(MakeLeviCivita, Rank2)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto* e = make_levi_civita(
        ctx, Realm::Oblique, space_2d(), {Level::Upper, Level::Upper}, {i, j});
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(t.slots.size(), 2u);
    EXPECT_EQ(t.rank, std::optional<int>{0});
    // Rank-2 ε is antisymmetric: it must carry the single transposition
    // generator, not be left silently symmetry-less.
    ASSERT_TRUE(t.traits.has_value());
    EXPECT_TRUE(t.traits->symmetry.generators.empty());
    EXPECT_EQ(t.traits->antisymmetry.generators.size(), 1u);
}

TEST(MakeLeviCivita, UnsupportedRankThrows)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto k = CountableIndex{ctx.alloc_index_id()};
    auto l = CountableIndex{ctx.alloc_index_id()};
    // Rank 4 has no realised antisymmetry generators: reject rather than build
    // an ε with a silently-empty (hence wrong) symmetry.
    EXPECT_THROW(
        {
            (void)make_levi_civita(
                ctx,
                Realm::Oblique,
                space_4d(),
                {Level::Lower, Level::Lower, Level::Lower, Level::Lower},
                {i, j, k, l});
        },
        std::invalid_argument);
}

TEST(MakeLeviCivita, LevelsMismatchThrows)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto k = CountableIndex{ctx.alloc_index_id()};
    EXPECT_THROW(
        {
            (void)make_levi_civita(
                ctx,
                Realm::Orthonormal,
                space_3d(),
                {Level::Lower, Level::Lower},
                {i, j, k});
        },
        std::invalid_argument);
}

TEST(MakeLeviCivita, IndicesMismatchThrows)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    EXPECT_THROW(
        {
            (void)make_levi_civita(
                ctx,
                Realm::Orthonormal,
                space_3d(),
                {Level::Lower, Level::Lower, Level::Lower},
                {i});
        },
        std::invalid_argument);
}

// ---- make_metric -------------------------------------------------------

TEST(MakeMetric, NameKindAndSymmetry)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto const* g = make_metric(
        ctx,
        Realm::Oblique,
        space_3d(),
        Level::Lower,
        Level::Lower,
        IndexAssoc{i},
        IndexAssoc{j});
    auto const& t = std::get<TensorObject>(g->node);
    EXPECT_EQ(t.name.v.view(), "g");
    ASSERT_TRUE(t.traits.has_value());
    EXPECT_EQ(
        t.traits->well_known,
        std::optional<WellKnownKind>{WellKnownKind::Metric});
    // Symmetric: one value-preserving slot-swap generator, like δ.
    EXPECT_EQ(t.traits->symmetry.generators.size(), 1u);
    ASSERT_EQ(t.slots.size(), 2u);
    EXPECT_EQ(t.slots[0].slot.level, Level::Lower);
    EXPECT_EQ(t.slots[1].slot.level, Level::Lower);
}

// ---- tender::visit -----------------------------------------------------

TEST(Visit, DispatchesCorrectAlternative)
{
    Context ctx;
    auto* e = make_scalar(ctx, Rational{7});
    bool scalar_seen = false;
    visit(
        mpk::mix::Overloads{
            [&](ScalarLiteral const& s)
            {
                scalar_seen = true;
                EXPECT_EQ(s.value, (Rational{7}));
            },
            [](TensorObject const&) {},
            [](Negate const&) {},
            [](Sum const&) {},
            [](Difference const&) {},
            [](TensorProduct const&) {},
            [](ScalarDiv const&) {},
            [](Dot const&) {},
            [](DDot const&) {},
            [](DDotAlt const&) {},
            [](Cross const&) {},
            [](ExplicitSum const&) {},
            [](NoSum const&) {},
        },
        *e);
    EXPECT_TRUE(scalar_seen);
}

TEST(Visit, TreeDepthCounter)
{
    struct DepthCounter
    {
        auto operator()(TensorObject const&) const -> int
        {
            return 0;
        }
        auto operator()(ScalarLiteral const&) const -> int
        {
            return 0;
        }
        auto operator()(Negate const& n) const -> int
        {
            return 1 + visit(*this, *n.operand);
        }
        auto operator()(Sum const& s) const -> int
        {
            return 1 + std::max(visit(*this, *s.left), visit(*this, *s.right));
        }
        auto operator()(Difference const& d) const -> int
        {
            return 1 + std::max(visit(*this, *d.left), visit(*this, *d.right));
        }
        auto operator()(TensorProduct const& p) const -> int
        {
            return 1 + std::max(visit(*this, *p.left), visit(*this, *p.right));
        }
        auto operator()(ScalarDiv const& d) const -> int
        {
            return 1 + std::max(visit(*this, *d.left), visit(*this, *d.right));
        }
        auto operator()(Dot const& d) const -> int
        {
            return 1 + std::max(visit(*this, *d.left), visit(*this, *d.right));
        }
        auto operator()(DDot const& d) const -> int
        {
            return 1 + std::max(visit(*this, *d.left), visit(*this, *d.right));
        }
        auto operator()(DDotAlt const& d) const -> int
        {
            return 1 + std::max(visit(*this, *d.left), visit(*this, *d.right));
        }
        auto operator()(Cross const& c) const -> int
        {
            return 1 + std::max(visit(*this, *c.left), visit(*this, *c.right));
        }
        auto operator()(ExplicitSum const& s) const -> int
        {
            return 1 + visit(*this, *s.body);
        }
        auto operator()(NoSum const& s) const -> int
        {
            return 1 + visit(*this, *s.body);
        }
    };

    Context ctx;
    // negate(sum(scalar(1), scalar(2))) → depth 2
    auto* a = make_scalar(ctx, Rational{1});
    auto* b = make_scalar(ctx, Rational{2});
    auto* s = make_sum(ctx, a, b);
    auto* n = make_negate(ctx, s);

    EXPECT_EQ(visit(DepthCounter{}, *a), 0);
    EXPECT_EQ(visit(DepthCounter{}, *s), 1);
    EXPECT_EQ(visit(DepthCounter{}, *n), 2);
}

// ---- Context allocates multiple nodes ----------------------------------

TEST(ExprContext, MultipleNodesInOneContext)
{
    Context ctx;
    auto* a = make_scalar(ctx, Rational{1});
    auto* b = make_scalar(ctx, Rational{2});
    auto* c = make_sum(ctx, a, b);
    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
    EXPECT_EQ(std::get<Sum>(c->node).left, a);
    EXPECT_EQ(std::get<Sum>(c->node).right, b);
}
