#include <tender/expr.hpp>
#include <tender/index.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>
#include <tender/tensor_order.hpp>

#include <gtest/gtest.h>

#include <optional>

using namespace tender;

namespace
{

auto sign(int c) -> int
{
    return (c > 0) - (c < 0);
}

// A bare rank-r TensorObject (the value the comparators take), optionally with
// field-derivative directions.
auto obj(
    char const* name,
    std::optional<int> rank,
    std::vector<SlotBinding> slots = {},
    std::vector<DerivMark> derivs = {}) -> TensorObject
{
    return TensorObject{
        make_tensor_name(name),
        rank,
        std::nullopt,
        std::move(slots),
        std::move(derivs)};
}

auto slot(
    Level level,
    Realm realm,
    IndexSpace const* space,
    int basis_id,
    std::optional<IndexAssoc> idx) -> SlotBinding
{
    return SlotBinding{IndexSlot{level, realm, space, basis_id}, idx};
}

} // namespace

TEST(TensorOrder, NameViewCmp)
{
    EXPECT_EQ(name_view_cmp("a", "a"), 0);
    EXPECT_LT(name_view_cmp("a", "b"), 0);
    EXPECT_GT(name_view_cmp("b", "a"), 0);
}

TEST(TensorOrder, SpaceCmp)
{
    EXPECT_EQ(space_cmp(space_3d(), space_3d()), 0);       // same pointer
    EXPECT_LT(sign(space_cmp(space_2d(), space_3d())), 1); // 2 < 3 dims
    EXPECT_EQ(sign(space_cmp(space_2d(), space_3d())), -1);
    EXPECT_EQ(sign(space_cmp(space_3d(), space_2d())), 1);
}

TEST(TensorOrder, IndexAssocCmp)
{
    std::optional<IndexAssoc> none;
    std::optional<IndexAssoc> ci0 = IndexAssoc{CountableIndex{0}};
    std::optional<IndexAssoc> ci1 = IndexAssoc{CountableIndex{1}};
    std::optional<IndexAssoc> conc = IndexAssoc{ConcreteIndex{2}};
    std::optional<IndexAssoc> conc2 = IndexAssoc{ConcreteIndex{5}};
    std::optional<IndexAssoc> lab =
        IndexAssoc{LabelIndex{make_index_name("i")}};
    std::optional<IndexAssoc> lab2 =
        IndexAssoc{LabelIndex{make_index_name("j")}};

    EXPECT_EQ(index_assoc_cmp(none, none), 0); // both empty
    EXPECT_LT(index_assoc_cmp(none, ci0), 0);  // empty < filled
    EXPECT_GT(index_assoc_cmp(ci0, none), 0);  // filled > empty
    // Same kind: by id / value / name.
    EXPECT_EQ(index_assoc_cmp(ci0, ci0), 0);
    EXPECT_EQ(sign(index_assoc_cmp(ci0, ci1)), -1);
    EXPECT_EQ(sign(index_assoc_cmp(ci1, ci0)), 1);
    EXPECT_EQ(sign(index_assoc_cmp(conc, conc2)), -1);
    EXPECT_EQ(index_assoc_cmp(conc, conc), 0);
    EXPECT_EQ(sign(index_assoc_cmp(lab, lab2)), -1);
    EXPECT_EQ(index_assoc_cmp(lab, lab), 0);
    // Different kinds order by the variant tag.
    EXPECT_NE(index_assoc_cmp(ci0, conc), 0);
    EXPECT_NE(index_assoc_cmp(conc, lab), 0);
}

TEST(TensorOrder, TensorObjectCmpScalarFields)
{
    EXPECT_EQ(sign(tensor_object_cmp(obj("a", 0), obj("b", 0))), -1); // name
    EXPECT_EQ(sign(tensor_object_cmp(obj("a", 0), obj("a", 1))), -1); // rank
    // slot count
    auto s =
        slot(Level::Lower, Realm::Orthonormal, space_3d(), 0, std::nullopt);
    EXPECT_EQ(sign(tensor_object_cmp(obj("a", 1), obj("a", 1, {s}))), -1);
    EXPECT_EQ(tensor_object_cmp(obj("a", 1), obj("a", 1)), 0);
}

TEST(TensorOrder, TensorObjectCmpPerSlot)
{
    auto lower =
        slot(Level::Lower, Realm::Orthonormal, space_3d(), 0, std::nullopt);
    auto upper =
        slot(Level::Upper, Realm::Orthonormal, space_3d(), 0, std::nullopt);
    auto oblique =
        slot(Level::Lower, Realm::Oblique, space_3d(), 0, std::nullopt);
    auto basis1 =
        slot(Level::Lower, Realm::Orthonormal, space_3d(), 1, std::nullopt);
    auto idx = slot(
        Level::Lower,
        Realm::Orthonormal,
        space_3d(),
        0,
        IndexAssoc{ConcreteIndex{2}});

    EXPECT_NE(tensor_object_cmp(obj("a", 1, {lower}), obj("a", 1, {upper})), 0);
    EXPECT_NE(
        tensor_object_cmp(obj("a", 1, {lower}), obj("a", 1, {oblique})), 0);
    EXPECT_NE(tensor_object_cmp(obj("a", 1, {lower}), obj("a", 1, {basis1})), 0);
    EXPECT_NE(tensor_object_cmp(obj("a", 1, {lower}), obj("a", 1, {idx})), 0);
}

// The vibe-000070 field-derivative directions are part of the order: a base
// field, ∂_x of it, and ∂_x∂_y of it are all distinct, and the sorted
// multi-index makes the comparison symmetric.
TEST(TensorOrder, TensorObjectCmpFieldDerivs)
{
    DerivMark dx{make_tensor_name("x"), CoordinateRef{1, 0, false}};
    DerivMark dy{make_tensor_name("y"), CoordinateRef{1, 1, false}};

    auto base = obj("T", 2);
    auto d_x = obj("T", 2, {}, {dx});
    auto d_xy = obj("T", 2, {}, {dx, dy});
    auto d_yx = obj("T", 2, {}, {dx, dy}); // already sorted form

    EXPECT_EQ(sign(tensor_object_cmp(base, d_x)), -1); // size differs
    EXPECT_EQ(sign(tensor_object_cmp(d_x, d_xy)), -1); // size differs
    EXPECT_EQ(tensor_object_cmp(d_xy, d_yx), 0);       // same sorted index
    // Differ by a single direction's (chart_id, slot).
    auto d_y = obj("T", 2, {}, {dy});
    EXPECT_NE(tensor_object_cmp(d_x, d_y), 0);
    DerivMark dx_other{make_tensor_name("x"), CoordinateRef{2, 0, false}};
    auto d_x2 = obj("T", 2, {}, {dx_other});
    EXPECT_NE(tensor_object_cmp(d_x, d_x2), 0); // chart_id differs
}
