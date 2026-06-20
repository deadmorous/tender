#pragma once

#include <mpk/mix/enum_flags.hpp>
#include <mpk/mix/util/overloads.hpp>
#include <tender/index.hpp>
#include <tender/permutation_spec.hpp>
#include <tender/rational.hpp>

#include <concepts>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

namespace tender
{

// Forward declaration — node types hold Expr const* for child nodes.
struct Expr;

// ---- TensorTraits -------------------------------------------------------

// Well-known tensor roles, recognised by derivation rules.
enum class WellKnownKind : uint8_t
{
    Identity,   // identity tensor I
    Delta,      // Kronecker delta δ
    LeviCivita, // Levi-Civita symbol ε
    Metric,     // metric tensor g (g_ij = e_i·e_j)
};

// Renderer display flags.  Use mpk::mix::EnumFlags<RenderHint> to build sets.
enum class RenderHint : uint8_t
{
    OmitVoidIndexPlaceholders = 1, // suppress \cdot in mixed upper/lower slots
};

// Value-preserving permutation generators for a tensor object (e.g. even
// permutations of Levi-Civita, any slot swap of a symmetric tensor).
struct SymmetrySpec final
{
    PermutationSpec generators;
};

// Representative sign-flipping permutations (odd permutations, not a group).
// The full set of sign-flipping permutations is derived by composing these
// with the symmetry group.
struct AntisymmetrySpec final
{
    PermutationSpec generators;
};

struct TensorTraits
{
    std::optional<WellKnownKind> well_known = {};
    SymmetrySpec symmetry = {};
    AntisymmetrySpec antisymmetry = {};
    mpk::mix::EnumFlags<RenderHint> render_hints = {};
};

// ---- SlotBinding -------------------------------------------------------

// One positional cell of a TensorObject: the slot descriptor (level, realm,
// index space) paired with its optional index association.  When index is
// nullopt the position exists in the layout but is not yet filled.
struct SlotBinding final
{
    IndexSlot slot;
    std::optional<IndexAssoc> index;
};

// ---- Leaf nodes --------------------------------------------------------

// A named tensor object.
//
// rank:   abstract mathematical rank (nullopt = unknown / not yet declared).
// traits: non-null for well-known tensors; carries symmetry and render hints.
// slots:  positional layout with optional index fillings. The slot count
//        has no enforced relationship to rank; abstract tensors carry an
//        empty slots vector, while the same tensor in an indexed expression
//        carries the relevant SlotBindings.
struct TensorObject final
{
    TensorName name;
    std::optional<int> rank;
    std::optional<TensorTraits> traits;
    std::vector<SlotBinding> slots;
};

// A numeric scalar literal.
struct ScalarLiteral final
{
    Rational value;
};

// ---- Unary node --------------------------------------------------------

struct Negate final
{
    Expr const* operand;
};

// ---- Binary operation nodes --------------------------------------------

struct Sum final
{
    Expr const* left;
    Expr const* right;
};
struct Difference final
{
    Expr const* left;
    Expr const* right;
};
struct TensorProduct final
{
    Expr const* left;
    Expr const* right;
}; // ⊗
struct ScalarDiv final
{
    Expr const* left;
    Expr const* right;
}; // / (right rank-0)
struct Dot final
{
    Expr const* left;
    Expr const* right;
}; // ·
struct DDot final
{
    Expr const* left;
    Expr const* right;
}; // :
struct DDotAlt final
{
    Expr const* left;
    Expr const* right;
}; // ··
struct Cross final
{
    Expr const* left;
    Expr const* right;
}; // ×

// ---- Summation annotation nodes ----------------------------------------

// Force summation over index in body.
// bound == nullptr → concrete range from the index's slot space.
// bound != nullptr → symbolic upper bound (parametric cardinality, vibe
// 000028).
struct ExplicitSum final
{
    CountableIndex index;
    Expr const* body;
    Expr const* bound;
};

// Suppress automatic contraction for index in body.
struct NoSum final
{
    CountableIndex index;
    Expr const* body;
};

// ---- Expr --------------------------------------------------------------

// Expr is a struct wrapping a variant of all node types. It is
// forward-declarable as a struct, which allows node types to hold
// Expr const* pointers before Expr's full definition.
//
// Use tender::visit (below) instead of std::visit; it unwraps the .node
// member so that the visitor sees the concrete node types directly.
struct Expr final
{
    using Node = std::variant<
        TensorObject,
        ScalarLiteral,
        Negate,
        Sum,
        Difference,
        TensorProduct,
        ScalarDiv,
        Dot,
        DDot,
        DDotAlt,
        Cross,
        ExplicitSum,
        NoSum>;

    Node node;

    Expr(Expr const&) = default;
    Expr(Expr&&) noexcept = default;
    auto operator=(Expr const&) -> Expr& = default;
    auto operator=(Expr&&) noexcept -> Expr& = default;

    template <typename T>
        requires(!std::same_as<std::remove_cvref_t<T>, Expr>)
    /*implicit*/ Expr(T&& v) noexcept(
        std::is_nothrow_constructible_v<Node, T&&>) :
      node(std::forward<T>(v))
    {
    }
};

// ---- visit -------------------------------------------------------------

// Visit a single Expr. Use mpk::mix::Overloads{...} for multi-lambda visitors.
// Callable via ADL when inside the tender namespace.
template <typename Visitor>
decltype(auto) visit(Visitor&& v, Expr const& e)
{
    return std::visit(std::forward<Visitor>(v), e.node);
}

// Visit two Expr nodes simultaneously (for binary-operation visitors).
template <typename Visitor>
decltype(auto) visit(Visitor&& v, Expr const& a, Expr const& b)
{
    return std::visit(std::forward<Visitor>(v), a.node, b.node);
}

// ---- Factory function declarations -------------------------------------

// Generic tensor object.  slots defaults to empty (abstract form).
// rank is independent of slot count.
[[nodiscard]] auto make_tensor_object(
    Context&,
    TensorName,
    std::vector<SlotBinding> slots = {},
    std::optional<int> rank = std::nullopt) -> Expr const*;

// Numeric scalar literal.
[[nodiscard]] auto make_scalar(Context&, Rational) -> Expr const*;

// Unary
[[nodiscard]] auto make_negate(Context&, Expr const*) -> Expr const*;

// Binary
[[nodiscard]] auto make_sum(Context&, Expr const*, Expr const*) -> Expr const*;
[[nodiscard]] auto make_difference(Context&, Expr const*, Expr const*)
    -> Expr const*;
[[nodiscard]] auto make_tensor_product(Context&, Expr const*, Expr const*)
    -> Expr const*;
[[nodiscard]] auto make_scalar_div(Context&, Expr const*, Expr const*)
    -> Expr const*;
[[nodiscard]] auto make_dot(Context&, Expr const*, Expr const*) -> Expr const*;
[[nodiscard]] auto make_ddot(Context&, Expr const*, Expr const*) -> Expr const*;
[[nodiscard]] auto make_ddot_alt(Context&, Expr const*, Expr const*)
    -> Expr const*;
[[nodiscard]] auto make_cross(Context&, Expr const*, Expr const*) -> Expr const*;

// Summation annotations
[[nodiscard]] auto make_explicit_sum(
    Context&,
    CountableIndex,
    Expr const* body,
    Expr const* bound = nullptr) -> Expr const*;
[[nodiscard]] auto make_no_sum(Context&, CountableIndex, Expr const*)
    -> Expr const*;

// Well-known tensors
//
// Identity tensor I: invariant, rank 2, no slots.
[[nodiscard]] auto make_identity(Context&) -> Expr const*;

// Kronecker delta δ: two independently chosen levels, one realm, one space.
[[nodiscard]] auto make_delta(
    Context&,
    Realm,
    IndexSpace const*,
    Level level0,
    Level level1,
    IndexAssoc index0,
    IndexAssoc index1) -> Expr const*;

// Levi-Civita symbol ε: rank N = space->values().size().
// Precondition: levels.size() == indices.size() == space->values().size().
[[nodiscard]] auto make_levi_civita(
    Context&,
    Realm,
    IndexSpace const*,
    std::vector<Level>,
    std::vector<IndexAssoc>) -> Expr const*;

// Metric tensor g: two independently chosen levels, one realm, one space.
// g_ij = e_i·e_j (both lower), g^ij = e^i·e^j (both upper).  Symmetric in its
// two slots.  Renders as "g".
[[nodiscard]] auto make_metric(
    Context&,
    Realm,
    IndexSpace const*,
    Level level0,
    Level level1,
    IndexAssoc index0,
    IndexAssoc index1) -> Expr const*;

} // namespace tender
