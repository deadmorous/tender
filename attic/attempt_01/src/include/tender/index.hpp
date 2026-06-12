#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <tender/fwd.hpp>

namespace tender
{

// ===========================================================================
// IndexRange: concrete cardinality or symbolic name (e.g. "N" for FEM count)
// ===========================================================================

using IndexRange = std::variant<int64_t, std::string>;

// ===========================================================================
// IndexSpace
// ===========================================================================

class IndexSpace
{
public:
    IndexSpace(std::string name, IndexRange range, bool auto_sum);

    [[nodiscard]] auto name() const noexcept -> std::string const&
    {
        return name_;
    }
    [[nodiscard]] auto range() const noexcept -> IndexRange const&
    {
        return range_;
    }
    [[nodiscard]] auto auto_sum() const noexcept -> bool
    {
        return auto_sum_;
    }

private:
    std::string name_;
    IndexRange range_;
    bool auto_sum_;
};

// ===========================================================================
// Index
// ===========================================================================

class Index
{
public:
    Index(std::string letter, IndexSpace* space);

    [[nodiscard]] auto letter() const noexcept -> std::string const&
    {
        return letter_;
    }
    [[nodiscard]] auto space() const noexcept -> IndexSpace*
    {
        return space_;
    }

private:
    std::string letter_;
    IndexSpace* space_;
};

// ===========================================================================
// Slot
// ===========================================================================

enum class SlotLevel
{
    Upper,
    Lower
};

struct Slot
{
    SlotLevel level;
    std::string display; // display name for printing; may be empty
    Index* index{};      // nullptr → anonymous (unbound)
};

using SlotList = std::vector<Slot>;

// ===========================================================================
// Built-in index spaces (singleton objects; addresses are stable)
// ===========================================================================

auto spatial_3d_space() noexcept -> IndexSpace*; // range=3, auto_sum=true
auto spatial_2d_space() noexcept -> IndexSpace*; // range=2, auto_sum=true

// ===========================================================================
// Index factory functions
// ===========================================================================

auto make_index(ResourceList& rl, std::string letter, IndexSpace* space)
    -> Index*;
auto auto_sum_index_3d(ResourceList& rl, std::string letter) -> Index*;
auto auto_sum_index_2d(ResourceList& rl, std::string letter) -> Index*;

} // namespace tender
