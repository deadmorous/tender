#pragma once
#include <tender/fwd.hpp>
namespace tender
{
struct Context
{
    ResourceList rl;
    auto alloc_index_id() -> int
    {
        return next_id_++;
    }

private:
    int next_id_ = 0;
};
} // namespace tender
