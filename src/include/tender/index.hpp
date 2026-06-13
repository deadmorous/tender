#pragma once

#include <tender/index_space.hpp>

#include <variant>

namespace tender
{

// ---- Realm and Level ---------------------------------------------------

enum class Realm
{
    Oblique,     // covariant/contravariant distinction; upper/lower matters
    Orthonormal, // orthonormal basis; upper/lower interchangeable
    Collection,  // ordinal number within a collection; no auto-summation
    Label        // descriptive word; no summation, no index space
};

enum class Level
{
    Upper,
    Lower
};

// ---- Slot types --------------------------------------------------------

// A positional cell that holds no index and cannot have one attached.
// Its sole purpose is to fix the positions of other slots across levels.
struct VoidSlot
{
    Level level;
};

// A positional cell that must be associated with an index or concrete
// value before the expression is valid.
struct IndexSlot
{
    Level level;
    Realm realm;
    // Null for Label realm; plain pointer to the IndexSpace for all other
    // realms. Identity by pointer: same instance = same space. Lifetime of
    // the pointed-to object must exceed the lifetime of this slot.
    IndexSpace const* space;
};

using Slot = std::variant<VoidSlot, IndexSlot>;

// ---- Index association types -------------------------------------------

// A dummy index identified by an opaque numeric id. The id has no meaning
// outside the expression tree it belongs to; two expressions may reuse the
// same id independently. Whether the index is free or contracted is
// determined by the summation-detection pass (see vibe 000028).
struct CountableIndex
{
    int id;
};

// Substitution of a concrete integer value for an index slot. No summation
// is triggered for slots carrying a concrete value.
struct ConcreteIndex
{
    int value;
};

// Index for the Label realm, identified by its string name.
struct LabelIndex
{
    IndexName name;
};

using IndexAssoc = std::variant<CountableIndex, ConcreteIndex, LabelIndex>;

} // namespace tender
