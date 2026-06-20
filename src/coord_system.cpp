#include <tender/coord_system.hpp>

namespace tender
{

namespace
{

// A bare rank-1 named frame vector.
auto frame_vector(Context& ctx, std::string_view name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 1);
}

} // namespace

auto wcs(Context& ctx) -> Basis
{
    return make_orthonormal_basis(
        space_3d(),
        {frame_vector(ctx, "i"),
         frame_vector(ctx, "j"),
         frame_vector(ctx, "k")});
}

auto cylindrical(Context& ctx) -> Basis
{
    return make_orthonormal_basis(
        space_3d(),
        {frame_vector(ctx, "r"),
         frame_vector(ctx, "\\theta"),
         frame_vector(ctx, "z")});
}

auto spherical(Context& ctx) -> Basis
{
    return make_orthonormal_basis(
        space_3d(),
        {frame_vector(ctx, "r"),
         frame_vector(ctx, "\\theta"),
         frame_vector(ctx, "\\phi")});
}

auto polar_2d(Context& ctx) -> Basis
{
    return make_orthonormal_basis(
        space_2d(), {frame_vector(ctx, "r"), frame_vector(ctx, "\\theta")});
}

} // namespace tender
