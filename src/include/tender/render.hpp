#pragma once

#include <tender/expr.hpp>

#include <map>
#include <optional>
#include <string>

namespace tender
{

// Bidirectional map between CountableIndex ids and display names.
//
// Pass the same map across multiple render_latex calls to get consistent
// naming: an index that appeared as "i" in the first call will still be
// "i" in the second call. The user may also pre-seed the map with
// assign() to force specific names before rendering begins.
class IndexNameMap final
{
public:
    // Explicitly assign a display name to an index id.
    // Removes any previous binding on either side to keep the map bijective.
    void assign(CountableIndex, IndexName);

    // Forward lookup: return the name for id.
    // If not yet assigned, allocate the next available name from space's
    // schema (skipping any names already claimed by other indices).
    // Throws std::out_of_range if the schema is exhausted.
    auto name_for(CountableIndex, IndexSpace const*) -> IndexName;

    // Forward lookup without allocation.
    auto lookup(CountableIndex) const -> std::optional<IndexName>;

    // Reverse lookup: return the id assigned this name, if any.
    auto index_for(IndexName) const -> std::optional<CountableIndex>;

private:
    std::map<int, IndexName> id_to_name_;
    std::map<std::string, int> name_to_id_; // key = string content of IndexName
    std::map<IndexSpace const*, int> next_schema_pos_;
};

// Render expr to a LaTeX math string (no surrounding $..$ delimiters).
// map is updated in-place as new dummy-index names are allocated.
[[nodiscard]] auto render_latex(Expr const&, IndexNameMap&) -> std::string;

} // namespace tender
