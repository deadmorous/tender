#pragma once

#include <cstdint>

namespace tender
{

// An algebraic constraint a symbol is *declared* to satisfy (vibe 000110 I4):
// a unit vector, or an orthogonal tensor.  It is a property of the symbol, the
// way symmetry is — but where symmetry is a linear identification of components
// that the canonical form can absorb, these are *quadratic* relations no index
// normal form expresses, so they are consumed differently:
//
//   * they mint rewrite rules (`n·n → 1`, `P·Pᵀ → I`) in the Context;
//   * they make the component decision procedure abstain, since a claim about
//     a constrained symbol is conditional and expanding the symbol as an
//     arbitrary tensor would refute true conditional claims (vibe 000110 M1);
//   * they make the symbol *literal in a rewrite pattern*: a slot-less
//     abstract tensor is otherwise a pattern variable, so the rule `P·Pᵀ → I`
//     would read "for any X, X·Xᵀ = I" and prove the orthogonality of every
//     tensor there is.  A declared symbol is a specific object, not a
//     placeholder, and that is what this bit says to the matcher.
//
// It lives both on the symbol's traits (where the matcher can see it with no
// Context in hand) and in the Context's registry (which is what can enumerate
// the declarations to mint their rules).  One factory sets both, so the two
// cannot drift.
struct SymbolConstraint final
{
    enum class Kind : std::uint8_t
    {
        Unit,       // |n| = 1, i.e. n·n = 1 (rank 1)
        Orthogonal, // P·Pᵀ = Pᵀ·P = I (rank 2)
    };

    Kind kind = Kind::Unit;
    // Orthogonal only: det = +1 (a rotation, and rotations form a group under
    // ·) or det = −1 (containing a reflection).  Nothing in the library can
    // *check* this — P·Pᵀ = I holds for both and there is no determinant — so
    // it is the user's assertion, recorded as one (vibe 000110 I5).
    bool proper = true;
};

} // namespace tender
