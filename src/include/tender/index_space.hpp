#pragma once

#include <tender/context.hpp>
#include <tender/name.hpp>

#include <span>
#include <stdexcept>
#include <vector>

namespace tender
{

// An index space defines the set of integer values an index ranges over
// during summation, and the naming schema used to assign display names
// to dummy indices at render time.
//
// The value set is explicit and need not be contiguous (e.g. {1,3} is valid).
//
// Identity is by pointer. Two IndexSpace objects with identical value sets
// are still distinct if they are different instances — an index may only be
// attached to slots that refer to the same IndexSpace instance.
class IndexSpace final
{
public:
    IndexSpace(std::vector<int> values, std::vector<IndexName> schema);

    // All index values, in the order they were supplied.
    auto values() const noexcept -> std::span<int const>
    {
        return values_;
    }

    // Return the n-th dummy index name from the naming schema (0-based).
    // Throws std::out_of_range if n >= schema size.
    auto dummy_name(int n) const -> IndexName;

private:
    std::vector<int> values_;
    std::vector<IndexName> schema_;
};

// Allocate an IndexSpace in ctx and return a plain pointer.
// Lifetime is tied to ctx.
[[nodiscard]] inline auto make_index_space(
    Context& ctx,
    std::vector<int> values,
    std::vector<IndexName> schema) -> IndexSpace const*
{
    return ctx.make<IndexSpace>(std::move(values), std::move(schema));
}

// ---- Well-known index spaces -------------------------------------------
//
// These are process-lifetime singletons. Use pointer equality to compare
// identity. They are not owned by any Context.

// 3D spatial: values {1,2,3}, schema i,j,k,l,m,n,p,q,r,s,t,u,v,w,x,y,z
auto space_3d() -> IndexSpace const*;

// 2D spatial: values {1,2}, schema \alpha..\theta
auto space_2d() -> IndexSpace const*;

// 4D spacetime: values {0,1,2,3}, schema \mu..\omega (12 names)
auto space_4d() -> IndexSpace const*;

} // namespace tender
