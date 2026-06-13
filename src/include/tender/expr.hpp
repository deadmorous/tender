#pragma once

#include <mpk/mix/util/overloads.hpp>
#include <tender/index.hpp>
#include <tender/rational.hpp>

#include <concepts>
#include <type_traits>
#include <variant>
#include <vector>

namespace tender
{

// Forward declaration — node types hold Expr const* for child nodes.
struct Expr;

// ---- Leaf nodes --------------------------------------------------------

// A named tensor object with a complete slot layout.
//
// slots: ordered list of positional cells (VoidSlot or IndexSlot).
// indices: one entry per IndexSlot in slots, in slot order.
// Invariant: indices.size() == count of IndexSlot alternatives in slots.
// A rank-0 scalar object has empty slots and indices.
struct TensorObject
{
    TensorName name;
    std::vector<Slot> slots;
    std::vector<IndexAssoc> indices;
};

// A numeric scalar literal.
struct ScalarLiteral
{
    Rational value;
};

// ---- Unary node --------------------------------------------------------

struct Negate
{
    Expr const* operand;
};

// ---- Binary operation nodes --------------------------------------------

struct Sum
{
    Expr const* left;
    Expr const* right;
};
struct Difference
{
    Expr const* left;
    Expr const* right;
};
struct TensorProduct
{
    Expr const* left;
    Expr const* right;
}; // ⊗
struct ScalarDiv
{
    Expr const* left;
    Expr const* right;
}; // / (right must be rank-0)
struct Dot
{
    Expr const* left;
    Expr const* right;
}; // ·
struct DDot
{
    Expr const* left;
    Expr const* right;
}; // :
struct DDotAlt
{
    Expr const* left;
    Expr const* right;
}; // ··
struct Cross
{
    Expr const* left;
    Expr const* right;
}; // ×

// ---- Summation annotation nodes ----------------------------------------

// Force summation over index in body.
// bound == nullptr  → concrete range from the index's slot space.
// bound != nullptr  → symbolic upper bound (parametric cardinality, see vibe
// 000028).
struct ExplicitSum
{
    CountableIndex index;
    Expr const* body;
    Expr const* bound;
};

// Suppress automatic contraction for index in body.
struct NoSum
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
struct Expr
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

// Generic: tensor object with explicit slot layout and index associations.
// Precondition: indices.size() == count of IndexSlot in slots.
[[nodiscard]] auto make_tensor_object(
    Context&,
    TensorName,
    std::vector<Slot>,
    std::vector<IndexAssoc>) -> Expr const*;

// Zero-slot tensor object (symbolic scalar field or invariant scalar).
[[nodiscard]] auto make_scalar_object(Context&, TensorName) -> Expr const*;

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
// Kronecker delta δ: two independent levels, one realm, one index space.
[[nodiscard]] auto make_delta(
    Context&,
    Realm,
    IndexSpace const*,
    Level level0,
    Level level1,
    IndexAssoc index0,
    IndexAssoc index1) -> Expr const*;

// Identity tensor I: same slot layout as delta, different name.
[[nodiscard]] auto make_identity(
    Context&,
    Realm,
    IndexSpace const*,
    Level level0,
    Level level1,
    IndexAssoc index0,
    IndexAssoc index1) -> Expr const*;

// Levi-Civita symbol ε: rank N where N = space->values().size().
// Precondition: levels.size() == indices.size() == space->values().size().
[[nodiscard]] auto make_levi_civita(
    Context&,
    Realm,
    IndexSpace const*,
    std::vector<Level>,
    std::vector<IndexAssoc>) -> Expr const*;

} // namespace tender
