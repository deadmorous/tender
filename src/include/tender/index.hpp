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

// ---- Slot type ---------------------------------------------------------

// A positional cell that carries a level, realm, and (for all realms
// except Label) a pointer to the index space it ranges over.
// Null space is valid only for the Label realm.
//
// basis_id identifies which Basis this index belongs to (vibe 000067): 0 means
// "no basis / basis-unaware" — the default for δ, ε, and every hand-written
// tensor — while a positive id is a handle into the Context's basis registry.
// It is a contraction-NEUTRAL label: Einstein summation ignores it (so a shared
// index may sum across two bases, e.g. the rotation tensor e_i ⊗ e'_i), but
// structural equality, canonical ordering, hashing, and the matcher all honour
// it, so e_i^A and e_i^B are distinct.  basis_id is the LAST member so existing
// aggregate initialisers IndexSlot{level, realm, space} keep compiling.
struct IndexSlot final
{
    Level level;
    Realm realm;
    IndexSpace const* space;
    int basis_id = 0;
};

// ---- Index association types -------------------------------------------

// A dummy index identified by an opaque numeric id. The id has no meaning
// outside the expression tree it belongs to; two expressions may reuse the
// same id independently. Whether the index is free or contracted is
// determined by the summation-detection pass (see vibe 000028).
struct CountableIndex final
{
    int id;
};

// Substitution of a concrete integer value for an index slot. No summation
// is triggered for slots carrying a concrete value.
struct ConcreteIndex final
{
    int value;
};

// Index for the Label realm, identified by its string name.
struct LabelIndex final
{
    IndexName name;
};

using IndexAssoc = std::variant<CountableIndex, ConcreteIndex, LabelIndex>;

} // namespace tender
