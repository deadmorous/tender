#include <tender/basis.hpp>
#include <tender/coord_system.hpp>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/index_space.hpp>

#include <gtest/gtest.h>

#include <variant>

using namespace tender;

namespace
{

auto name_of(Expr const* e) -> std::string_view
{
    return std::get<TensorObject>(e->node).name.v.view();
}

} // namespace

TEST(CoordSystem, Wcs)
{
    Context ctx;
    auto b = wcs(ctx);
    EXPECT_TRUE(b.is_orthonormal());
    EXPECT_EQ(b.space(), space_3d());
    ASSERT_EQ(b.dim(), 3);
    EXPECT_EQ(name_of(b.basis(0)), "i");
    EXPECT_EQ(name_of(b.basis(1)), "j");
    EXPECT_EQ(name_of(b.basis(2)), "k");
    // Orthonormal: cobasis coincides with basis.
    EXPECT_EQ(b.cobasis(0), b.basis(0));
}

TEST(CoordSystem, Cylindrical)
{
    Context ctx;
    auto b = cylindrical(ctx);
    EXPECT_TRUE(b.is_orthonormal());
    ASSERT_EQ(b.dim(), 3);
    EXPECT_EQ(name_of(b.basis(0)), "r");
    EXPECT_EQ(name_of(b.basis(1)), "\\theta");
    EXPECT_EQ(name_of(b.basis(2)), "z");
}

TEST(CoordSystem, Spherical)
{
    Context ctx;
    auto b = spherical(ctx);
    EXPECT_TRUE(b.is_orthonormal());
    ASSERT_EQ(b.dim(), 3);
    EXPECT_EQ(name_of(b.basis(0)), "r");
    EXPECT_EQ(name_of(b.basis(1)), "\\theta");
    EXPECT_EQ(name_of(b.basis(2)), "\\phi");
}

TEST(CoordSystem, Polar2d)
{
    Context ctx;
    auto b = polar_2d(ctx);
    EXPECT_TRUE(b.is_orthonormal());
    EXPECT_EQ(b.space(), space_2d());
    ASSERT_EQ(b.dim(), 2);
    EXPECT_EQ(name_of(b.basis(0)), "r");
    EXPECT_EQ(name_of(b.basis(1)), "\\theta");
}

TEST(CoordSystem, ProducedBasisDrivesExpansion)
{
    // A coordinate system's basis works with the basis-parameterized steps.
    Context ctx;
    auto b = wcs(ctx);
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* expanded = expand_in_basis(ctx, a, b, Variance::Covariant);
    EXPECT_TRUE(structural_eq(
        reassemble(ctx, steps::canonicalize(ctx, expanded), b), a));
}
