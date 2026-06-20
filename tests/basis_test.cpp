#include <tender/basis.hpp>
#include <tender/expr.hpp>
#include <tender/index_space.hpp>

#include <gtest/gtest.h>

#include <stdexcept>

using namespace tender;

namespace
{

// A bare rank-1 named vector.
auto vec(Context& ctx, char const* name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 1);
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
