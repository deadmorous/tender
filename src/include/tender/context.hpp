#pragma once

#include <mpk/mix/util/resource_list.hpp>

#include <map>
#include <memory>
#include <vector>

namespace tender
{

// Forward declaration: the Context owns a registry of bases (vibe 000067) but
// only stores pointers, so it need not see the full Basis definition.
class Basis;

// Forward declaration: the Context also owns a registry of per-chart connection
// tables (vibe 000071) by basis id, storing pointers only (defined in
// derivation.hpp).
struct BasisConnection;

// Forward declaration: the chart-frame cache stores a fingerprint of Expr
// pointers (vibe 000072 Obs 2); pointers only, so the full definition (in
// expr.hpp) is not needed here.
struct Expr;

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
    Context() :
      id_factory_{std::make_shared<IdFactory>()},
      basis_registry_{std::make_shared<std::vector<Basis const*>>()}
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

    // Register a basis and return its id (vibe 000067).  The id is the slot
    // tag (IndexSlot::basis_id); 1-based, unique within the sharing group, so 0
    // stays reserved for "basis-unaware".  The registry stores the pointer only
    // (the Basis must outlive the expressions that reference its id).
    auto register_basis(Basis const* b) -> int
    {
        basis_registry_->push_back(b);
        return static_cast<int>(basis_registry_->size());
    }

    // Resolve a basis id back to its Basis, or nullptr for 0 / out-of-range.
    auto basis(int id) const -> Basis const*
    {
        if (id < 1 || id > static_cast<int>(basis_registry_->size()))
            return nullptr;
        return (*basis_registry_)[static_cast<std::size_t>(id - 1)];
    }

    // Register a chart's connection table (vibe 000071), keyed by the physical
    // basis's id, so the differentiator can resolve ∂_{q^j} e_i.  Overwrites
    // any previous entry for that basis id.  Stores the pointer only (the
    // connection must outlive the expressions differentiated against it).
    void register_connection(BasisConnection const* c, int basis_id)
    {
        (*connection_registry_)[basis_id] = c;
    }

    // Resolve a basis id to its connection table, or nullptr if none.
    auto connection(int basis_id) const -> BasisConnection const*
    {
        auto it = connection_registry_->find(basis_id);
        return it == connection_registry_->end() ? nullptr : it->second;
    }

    // A cached physical frame: its basis id plus a structural fingerprint of
    // the chart that built it (the coords + embedding Expr pointers, stable
    // under hash-consing).  The fingerprint lets physical_frame detect a
    // chart_id reused for a *different* geometry and rebuild instead of
    // returning a stale frame (vibe 000072 Obs 2).
    struct ChartFrame final
    {
        int basis_id = 0;
        std::vector<Expr const*> fingerprint;
    };

    // Cache the physical-frame basis id for a chart (vibe 000071), so
    // physical_frame is idempotent per chart and every caller — the user
    // building fields and the operators — resolves the *same* frame (hence the
    // same e_i atoms and connection).  Keyed by the chart's id, validated by
    // the geometry fingerprint (vibe 000072 Obs 2).
    void register_chart_frame(
        int chart_id, int basis_id, std::vector<Expr const*> fingerprint)
    {
        (*chart_frame_registry_)[chart_id] =
            ChartFrame{basis_id, std::move(fingerprint)};
    }

    // The cached physical frame for a chart, or nullptr if none built yet.
    auto chart_frame(int chart_id) const -> ChartFrame const*
    {
        auto it = chart_frame_registry_->find(chart_id);
        return it == chart_frame_registry_->end() ? nullptr : &it->second;
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
    // Basis registry, shared across the sharing group like the id factory so a
    // child context resolves bases registered by its parent (vibe 000067).
    std::shared_ptr<std::vector<Basis const*>> basis_registry_;
    // Connection registry (vibe 000071), shared likewise, mapping a physical
    // basis id to its chart's derivative table.
    std::shared_ptr<std::map<int, BasisConnection const*>> connection_registry_{
        std::make_shared<std::map<int, BasisConnection const*>>()};
    // Chart → physical-frame cache (vibe 000071), shared likewise.
    std::shared_ptr<std::map<int, ChartFrame>> chart_frame_registry_{
        std::make_shared<std::map<int, ChartFrame>>()};

    // Private copy constructor: shares the id factory and basis registry (basis
    // ids stay globally unique across the group), but starts with FRESH
    // connection / chart-frame registries.  Those are keyed by chart id (vibe
    // 000071), which a user may reuse across independent contexts; sharing them
    // would let one context resolve a chart to another's — since-freed — frame.
    // Fresh registries keep each context's charts self-contained.  rl_ is
    // default-initialised (ResourceList is non-copyable).
    explicit Context(Context const& other) :
      id_factory_{other.id_factory_}, basis_registry_{other.basis_registry_}
    {
    }
};

} // namespace tender
