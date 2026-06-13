#pragma once

#include <mpk/mix/util/resource_list.hpp>

#include <memory>

namespace tender
{

// Context is the single argument passed to every factory function. It owns:
//   - a ResourceList (arena-style lifetime for all allocated objects)
//   - a shared id factory (produces contiguous CountableIndex ids)
//
// Derived contexts created via new_context() share the id factory so that ids
// are globally unique within a derivation session, but each has its own
// resource list so its allocations can be released independently.
//
// Plain pointers returned by make<T>() are valid for the lifetime of the
// Context that allocated them; data structures store plain pointers, not
// shared_ptr.
class Context final
{
public:
    Context() : id_factory_{std::make_shared<IdFactory>()}
    {
    }

    Context(Context&&) noexcept = default;
    auto operator=(Context&&) noexcept -> Context& = default;

    auto operator=(Context const&) -> Context& = delete;

    // Allocate an object of type T, owned by this context's resource list.
    // Returns a plain pointer; lifetime is tied to this Context.
    template <typename T, typename... Args>
    auto make(Args&&... args) -> T*
    {
        return rl_.make<T>(std::forward<Args>(args)...);
    }

    // Allocate a fresh CountableIndex id, unique within the sharing group.
    auto alloc_index_id() -> int
    {
        return id_factory_->next++;
    }

    // Create a child context: shares the id factory (ids remain contiguous)
    // but starts with an empty resource list (independent allocation scope).
    auto new_context() -> Context
    {
        return Context{*this};
    }

private:
    struct IdFactory final
    {
        int next = 0;
    };

    mpk::mix::ResourceList rl_;
    std::shared_ptr<IdFactory> id_factory_;

    // Private copy constructor: shares id factory, fresh empty resource list.
    // ResourceList is non-copyable so rl_ is default-initialised here.
    explicit Context(Context const& other) : id_factory_{other.id_factory_}
    {
    }
};

} // namespace tender
