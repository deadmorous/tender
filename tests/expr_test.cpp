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
    auto const& s = std::get<ScalarLiteral>(e->node);
    EXPECT_EQ(s.value, (Rational{3, 2}));
}

// ---- make_scalar_object ------------------------------------------------

TEST(MakeScalarObject, ZeroSlots)
{
    Context ctx;
    auto* e = make_scalar_object(ctx, make_tensor_name("f"));
    ASSERT_NE(e, nullptr);
    ASSERT_TRUE(std::holds_alternative<TensorObject>(e->node));
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(t.name.v.view(), "f");
    EXPECT_TRUE(t.slots.empty());
    EXPECT_TRUE(t.indices.empty());
}

// ---- make_tensor_object ------------------------------------------------

TEST(MakeTensorObject, MatchingSlotsAndIndices)
{
    Context ctx;
    auto idx = ctx.alloc_index_id();
    std::vector<Slot> slots = {
        IndexSlot{Level::Upper, Realm::Oblique, space_3d()}};
    std::vector<IndexAssoc> indices = {CountableIndex{idx}};
    auto* e = make_tensor_object(
        ctx, make_tensor_name("v"), std::move(slots), std::move(indices));
    ASSERT_NE(e, nullptr);
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(t.name.v.view(), "v");
    ASSERT_EQ(t.slots.size(), 1u);
    ASSERT_EQ(t.indices.size(), 1u);
    EXPECT_EQ(std::get<CountableIndex>(t.indices[0]).id, idx);
}

TEST(MakeTensorObject, VoidSlotNotCounted)
{
    Context ctx;
    auto idx = ctx.alloc_index_id();
    std::vector<Slot> slots = {
        VoidSlot{Level::Upper},
        IndexSlot{Level::Lower, Realm::Orthonormal, space_3d()}};
    std::vector<IndexAssoc> indices = {CountableIndex{idx}};
    auto* e = make_tensor_object(
        ctx, make_tensor_name("A"), std::move(slots), std::move(indices));
    ASSERT_NE(e, nullptr);
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(t.slots.size(), 2u);
    EXPECT_EQ(t.indices.size(), 1u);
}

TEST(MakeTensorObject, MismatchedIndicesThrows)
{
    Context ctx;
    std::vector<Slot> slots = {
        IndexSlot{Level::Upper, Realm::Oblique, space_3d()},
        IndexSlot{Level::Upper, Realm::Oblique, space_3d()}};
    std::vector<IndexAssoc> indices = {CountableIndex{0}};
    EXPECT_THROW(
        {
            (void)make_tensor_object(
                ctx,
                make_tensor_name("T"),
                std::move(slots),
                std::move(indices));
        },
        std::invalid_argument);
}

// ---- Unary: make_negate ------------------------------------------------

TEST(MakeNegate, StoresOperand)
{
    Context ctx;
    auto* operand = make_scalar(ctx, Rational{1});
    auto* e = make_negate(ctx, operand);
    ASSERT_NE(e, nullptr);
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
    auto* l = make_scalar(ctx, Rational{3});
    auto* r = make_scalar(ctx, Rational{1});
    auto* e = make_difference(ctx, l, r);
    ASSERT_TRUE(std::holds_alternative<Difference>(e->node));
}

TEST(MakeBinary, TensorProductStoresChildren)
{
    Context ctx;
    auto* a = make_scalar_object(ctx, make_tensor_name("a"));
    auto* b = make_scalar_object(ctx, make_tensor_name("b"));
    auto* e = make_tensor_product(ctx, a, b);
    ASSERT_TRUE(std::holds_alternative<TensorProduct>(e->node));
    EXPECT_EQ(std::get<TensorProduct>(e->node).left, a);
    EXPECT_EQ(std::get<TensorProduct>(e->node).right, b);
}

TEST(MakeBinary, ScalarDivStoresChildren)
{
    Context ctx;
    auto* n = make_scalar(ctx, Rational{1});
    auto* d = make_scalar(ctx, Rational{2});
    auto* e = make_scalar_div(ctx, n, d);
    ASSERT_TRUE(std::holds_alternative<ScalarDiv>(e->node));
}

TEST(MakeBinary, DotStoresChildren)
{
    Context ctx;
    auto* l = make_scalar_object(ctx, make_tensor_name("u"));
    auto* r = make_scalar_object(ctx, make_tensor_name("v"));
    auto* e = make_dot(ctx, l, r);
    ASSERT_TRUE(std::holds_alternative<Dot>(e->node));
}

TEST(MakeBinary, DDotStoresChildren)
{
    Context ctx;
    auto* l = make_scalar_object(ctx, make_tensor_name("A"));
    auto* r = make_scalar_object(ctx, make_tensor_name("B"));
    auto* e = make_ddot(ctx, l, r);
    ASSERT_TRUE(std::holds_alternative<DDot>(e->node));
}

TEST(MakeBinary, DDotAltStoresChildren)
{
    Context ctx;
    auto* l = make_scalar_object(ctx, make_tensor_name("A"));
    auto* r = make_scalar_object(ctx, make_tensor_name("B"));
    auto* e = make_ddot_alt(ctx, l, r);
    ASSERT_TRUE(std::holds_alternative<DDotAlt>(e->node));
}

TEST(MakeBinary, CrossStoresChildren)
{
    Context ctx;
    auto* l = make_scalar_object(ctx, make_tensor_name("u"));
    auto* r = make_scalar_object(ctx, make_tensor_name("v"));
    auto* e = make_cross(ctx, l, r);
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
    auto* bound = make_scalar_object(ctx, make_tensor_name("N"));
    auto* e = make_explicit_sum(ctx, idx, body, bound);
    auto const& es = std::get<ExplicitSum>(e->node);
    EXPECT_EQ(es.bound, bound);
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

// ---- Well-known tensors ------------------------------------------------

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
    ASSERT_EQ(t.slots.size(), 2u);
    ASSERT_EQ(t.indices.size(), 2u);
    auto const& s0 = std::get<IndexSlot>(t.slots[0]);
    auto const& s1 = std::get<IndexSlot>(t.slots[1]);
    EXPECT_EQ(s0.level, Level::Upper);
    EXPECT_EQ(s0.realm, Realm::Oblique);
    EXPECT_EQ(s0.space, space_3d());
    EXPECT_EQ(s1.level, Level::Lower);
    EXPECT_EQ(std::get<CountableIndex>(t.indices[0]).id, i.id);
    EXPECT_EQ(std::get<CountableIndex>(t.indices[1]).id, j.id);
}

TEST(MakeDelta, BothLevelsUpper)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto* e = make_delta(
        ctx, Realm::Oblique, space_3d(), Level::Upper, Level::Upper, i, j);
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(std::get<IndexSlot>(t.slots[0]).level, Level::Upper);
    EXPECT_EQ(std::get<IndexSlot>(t.slots[1]).level, Level::Upper);
}

TEST(MakeIdentity, NameIsI)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    auto* e = make_identity(
        ctx, Realm::Orthonormal, space_3d(), Level::Lower, Level::Lower, i, j);
    ASSERT_NE(e, nullptr);
    auto const& t = std::get<TensorObject>(e->node);
    EXPECT_EQ(t.name.v.view(), "I");
    ASSERT_EQ(t.slots.size(), 2u);
}

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
    ASSERT_EQ(t.slots.size(), 3u);
    ASSERT_EQ(t.indices.size(), 3u);
    for (auto const& slot: t.slots)
    {
        auto const& is = std::get<IndexSlot>(slot);
        EXPECT_EQ(is.level, Level::Lower);
        EXPECT_EQ(is.realm, Realm::Orthonormal);
        EXPECT_EQ(is.space, space_3d());
    }
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
}

TEST(MakeLeviCivita, LevelsMismatchThrows)
{
    Context ctx;
    auto i = CountableIndex{ctx.alloc_index_id()};
    auto j = CountableIndex{ctx.alloc_index_id()};
    EXPECT_THROW(
        {
            (void)make_levi_civita(
                ctx,
                Realm::Orthonormal,
                space_3d(),
                {Level::Lower, Level::Lower}, // only 2 for rank-3 space
                {i, j, CountableIndex{ctx.alloc_index_id()}});
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
                {i}); // only 1 index for rank-3 space
        },
        std::invalid_argument);
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
    // Visitor that returns the depth of the expression tree.
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
