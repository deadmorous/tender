#include <tender/index.hpp>

#include <stdexcept>

namespace tender
{

// ===========================================================================
// IndexSpace
// ===========================================================================

IndexSpace::IndexSpace(std::string name, IndexRange range, bool auto_sum) :
  name_(std::move(name)), range_(std::move(range)), auto_sum_(auto_sum)
{
}

// ===========================================================================
// Index
// ===========================================================================

Index::Index(std::string letter, IndexSpace* space) :
  letter_(std::move(letter)), space_(space)
{
    if (!space_)
        throw std::invalid_argument("Index: space must not be null");
}

// ===========================================================================
// Built-in singleton spaces
// ===========================================================================

auto spatial_3d_space() noexcept -> IndexSpace*
{
    static IndexSpace s{"spatial_3d", int64_t{3}, true};
    return &s;
}

auto spatial_2d_space() noexcept -> IndexSpace*
{
    static IndexSpace s{"spatial_2d", int64_t{2}, true};
    return &s;
}

// ===========================================================================
// Index factory functions
// ===========================================================================

auto make_index(ResourceList& rl, std::string letter, IndexSpace* space)
    -> Index*
{
    return rl.make<Index>(std::move(letter), space);
}

auto auto_sum_index_3d(ResourceList& rl, std::string letter) -> Index*
{
    return rl.make<Index>(std::move(letter), spatial_3d_space());
}

auto auto_sum_index_2d(ResourceList& rl, std::string letter) -> Index*
{
    return rl.make<Index>(std::move(letter), spatial_2d_space());
}

} // namespace tender
