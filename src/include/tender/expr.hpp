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

// Elementary scalar functions (vibe 000069 M1).  Each kind has a derivative-
// table entry the chain rule consults (M2).  Operand and result are rank-0.
enum class ScalarFnKind : uint8_t
{
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
};

// Reference to a coordinate of a coordinate chart (vibe 000069 M1).  Stamped on
// a rank-0 TensorObject's traits to mark it as a chart coordinate the
// differentiator recognises (∂_{q^i} q^j = δ_ij).  chart_id == 0 means the
// coordinate is not yet bound to a chart (a free coordinate symbol); a chart
// stamps its id when it owns the coordinate (M4).  slot is the coordinate's
// 0-based position within the chart.
//
// `nonneg` is the one piece of the coordinate's domain the targeted scalar
// simplifier needs (vibe 000069 M3 decision 2): a coordinate known to be ≥ 0
// (e.g. a cylindrical radius), which licenses √(r²) → r and positive scale
// factors.  A chart stamps it from its known domain (M4); the richer interval
// domain is deferred until something needs it.  Identity-neutral, like the rest
// of the marker.
struct CoordinateRef final
{
    int chart_id = 0;
    int slot = 0;
    bool nonneg = false;
};

// A tensor field's coordinate dependence (vibe 000070 P7).  Stamped on a tensor
// declared as a *field* (a tensor whose value varies in space); absent on a
// plain constant tensor.  When `all` (the default — general position), the
// field depends on every coordinate, so ∂ is never silently zero; otherwise it
// depends only on the listed coordinates (each identified by its (chart_id,
// slot), allowing dependence on coordinates of different charts at once).
// Identity-neutral, like the `coordinate` marker: a field is told apart from a
// constant by inspecting this, not by comparison.
struct FieldDeps final
{
    bool all = true;
    std::vector<CoordinateRef> only = {};
};

// One direction of a field's accumulated partial derivative (vibe 000070 P7):
// the coordinate's display name (for ∂_x rendering) and its (chart_id, slot)
// identity.  A base field has none; ∂_q T appends q and keeps the list sorted
// by (chart_id, slot) so mixed partials are symmetric (T_{,xy} = T_{,yx}).
// Unlike the rest of the traits this IS part of structural identity (it
// distinguishes ∂_x T from T), so it lives on the TensorObject, not in
// TensorTraits.
struct FieldDerivDir final
{
    TensorName coord_name;
    CoordinateRef ref;
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
    // Set on a rank-0 object that is a coordinate of a chart (vibe 000069 M1).
    // Identity-neutral (structural_eq / tensor_object_cmp ignore traits), like
    // well_known: a coordinate is told apart from a plain scalar by inspecting
    // this marker, not by comparison.
    std::optional<CoordinateRef> coordinate = {};
    // Set on a tensor declared as a field (vibe 000070 P7): its coordinate
    // dependence.  Identity-neutral like the rest of the traits — field-ness is
    // consistent per tensor name.  The *derivative* directions, which DO bear
    // identity, live on TensorObject::field_derivs instead.
    std::optional<FieldDeps> field = {};
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
    // Accumulated partial-derivative directions of a field (vibe 000070 P7),
    // sorted by (chart_id, slot).  Empty on a base tensor; ∂_q of a field
    // appends q.  Part of structural identity (compared by structural_eq /
    // tensor_object_cmp), so ∂_x T and T are distinct and ∂_x∂_y T = ∂_y∂_x T.
    std::vector<FieldDerivDir> field_derivs = {};
};

// A numeric scalar literal.
struct ScalarLiteral final
{
    Rational value;
};

// ---- Unary nodes -------------------------------------------------------

struct Negate final
{
    Expr const* operand;
};

// tr(A) — the trace of a rank-2 tensor (a scalar).  For a dyad tr(a⊗b) = a·b.
struct Trace final
{
    Expr const* operand;
};

// vec(A) — the vector invariant of a rank-2 tensor.  For a dyad vec(a⊗b) = a×b.
struct VectorInvariant final
{
    Expr const* operand;
};

// A^T — the transpose of a rank-2 tensor.  For a dyad (a⊗b)^T = b⊗a.
struct Transpose final
{
    Expr const* operand;
};

// An elementary scalar function applied to a rank-0 operand: sin, cos, sqrt, …
// (vibe 000069 M1).  Linear-algebra operators stay separate; this is the scalar
// (coordinate-field) layer.
struct ScalarFn final
{
    ScalarFnKind kind;
    Expr const* operand;
};

// base raised to a power, both rank-0 (vibe 000069 M1).  Kept distinct from a
// repeated ⊗ product so the scalar simplifier can recognise patterns such as
// cos² + sin² (M3).  The exponent is an Expr (usually a ScalarLiteral, e.g.
// r²), leaving room for symbolic exponents.
struct Pow final
{
    Expr const* base;
    Expr const* exponent;
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
        Trace,
        VectorInvariant,
        Transpose,
        Sum,
        Difference,
        TensorProduct,
        ScalarDiv,
        Dot,
        DDot,
        DDotAlt,
        Cross,
        ExplicitSum,
        NoSum,
        ScalarFn,
        Pow>;

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

// Coordinate variable of a chart: a rank-0 TensorObject carrying a
// CoordinateRef trait (vibe 000069 M1).  chart_id 0 / slot 0 leave it unbound
// to a chart.  nonneg marks a coordinate known to be ≥ 0 (vibe 000069 M3),
// enabling √(x²) → x.
[[nodiscard]] auto make_coordinate(
    Context&, TensorName, int chart_id = 0, int slot = 0, bool nonneg = false)
    -> Expr const*;

// A tensor field of any rank (vibe 000070 P7): a TensorObject carrying a
// FieldDeps trait so the differentiator treats it as varying in space rather
// than constant.  `deps` empty means depend on all coordinates (general
// position — the default, so ∂ is never silently zero); a non-empty list
// restricts dependence to those coordinates (which may belong to different
// charts).  A rank-0 field is a scalar field.
//
// `symmetric` marks a rank-2 field symmetric (T_ij = T_ji, vibe 000073): its
// components inherit the slot-swap symmetry so T_θr canonicalizes to T_rθ and a
// pair such as (T_rθ + T_θr)/r folds to 2 T_rθ/r.  Only rank 2 is supported.
[[nodiscard]] auto make_field(
    Context&,
    TensorName,
    int rank,
    std::vector<CoordinateRef> deps = {},
    bool symmetric = false) -> Expr const*;

// The partial derivative ∂_q of a field (vibe 000070 P7): a fresh field of the
// same rank, name and dependence as `base`, with q appended to its (sorted)
// derivative directions.  `base` must be a field (its TensorObject carries a
// FieldDeps trait); `coord_name`/`coord` identify q.  Differentiating again
// appends further directions, kept sorted so mixed partials coincide.
[[nodiscard]] auto make_field_derivative(
    Context&,
    Expr const* base,
    TensorName coord_name,
    CoordinateRef coord) -> Expr const*;

// Elementary scalar function and power (vibe 000069 M1).
[[nodiscard]] auto make_scalar_fn(Context&, ScalarFnKind, Expr const* operand)
    -> Expr const*;
[[nodiscard]] auto make_pow(Context&, Expr const* base, Expr const* exponent)
    -> Expr const*;

// Unary
[[nodiscard]] auto make_negate(Context&, Expr const*) -> Expr const*;
[[nodiscard]] auto make_trace(Context&, Expr const*) -> Expr const*;
[[nodiscard]] auto make_vector_invariant(Context&, Expr const*) -> Expr const*;
[[nodiscard]] auto make_transpose(Context&, Expr const*) -> Expr const*;

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
