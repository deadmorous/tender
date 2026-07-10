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

// An applied differential-operator mark on a field (vibe 000077, step D): one
// derivation ∂/∂(wrt) that has been *applied* to (closed onto) this object —
// the closed form of a `Deriv` operator.  `wrt` is the coordinate
// differentiated against (its (chart_id, slot) identity), `coord_name` its
// display letter (for ∂_x rendering).  A base field carries none; applying ∂_q
// appends a mark and keeps the list sorted by (chart_id, slot) so mixed
// partials coincide (∂_x∂_y T = ∂_y∂_x T).  `link` (0 = an ordinary closed
// concrete derivative, the common case) is the abstract-direction tie of a
// *free-index* ∂ (vibe 000078): when non-zero it is the `CountableIndex` id of
// the summation direction, so this ∂_i and the frame vector `e_i` that carries
// the same id contract under implicit (Einstein) summation, and `free_slot`
// describes that index occurrence (level/realm/space) for summation detection,
// index renaming and rendering (∂_i with the index letter).  For a concrete
// mark (`link == 0`) `free_slot` is unused.  Unlike the rest of the traits
// these marks ARE part of structural identity (they distinguish ∂_x T from T),
// so they live on the TensorObject, not in TensorTraits.
//
// This replaces the old `FieldDerivDir`: a mark is now understood as an applied
// `Deriv` — the operator (vibe 000077 steps A–C) is the unapplied form,
// application (Leibniz) closes it into a mark here, so differentiation has one
// representation across its lifecycle.
struct DerivMark final
{
    TensorName coord_name;
    CoordinateRef wrt;
    int link = 0;
    IndexSlot free_slot = {};
    // Discriminates a free-index (abstract-direction) mark from a concrete one.
    // A dedicated flag rather than `link != 0` because a direction index id may
    // legitimately be 0 (index ids and the sentinel would otherwise collide).
    bool free = false;
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
    // consistent per tensor name.  The *applied derivatives*, which DO bear
    // identity, live on TensorObject::deriv_marks instead.
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
    // Applied differential-operator marks (vibe 000077 step D): the derivations
    // closed onto this field, sorted by (chart_id, slot).  Empty on a base
    // tensor; applying ∂_q appends a mark.  Part of structural identity
    // (compared by structural_eq / tensor_object_cmp), so ∂_x T and T are
    // distinct and ∂_x∂_y T = ∂_y∂_x T.
    std::vector<DerivMark> deriv_marks = {};
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

// A first-class differential operator (vibe 000077): the derivation ∂/∂(wrt),
// "differentiate with respect to the tensor object `wrt`".  `wrt` is a
// structural reference to that object — a coordinate today (a rank-0
// CoordinateRef TensorObject), any object later; the operator's *rank* is the
// rank of `wrt` (0 for a coordinate).
//
// This node is an *unapplied* operator: it acts, by convention, on everything
// to its right within its enclosing product (Leibniz greedily), and it is
// order-sensitive — canon must not commute it past its neighbours (else
// `e_i ∂_i` and `∂_i e_i` would collapse).  Application (the scope marks and
// the unapplied flag of the vibe-000077 lifecycle) arrives in the next step;
// here a Deriv is always the bare unapplied operator.
struct Deriv final
{
    Expr const* wrt;
};

// The chart-free ∇ operator (vibe 000078): a rank-1 invariant vector operator
// atom with no children.  This is the bare "∇ is a rank-1 vector operator" of
// the settled design — *not* the fused grad/div/rot node we retired.  grad, div
// and rot are ordinary product nodes with `Nabla` on the left (`∇⊗T`, `∇·T`,
// `∇×T`); `∇·∇` renders `Δ`.  Like `Deriv`, it is operator-aware: canon keeps
// it positional (it acts rightward, Leibniz-greedily), never commuted past its
// neighbours.  A chart supplies its expansion `Σ_i (1/h_i) e_i ∂_i` at
// evaluation time (`expand_nabla`); until then it stays abstract, so `inc ε =
// ∇×(∇×ε)ᵀ` can be written and inspected without a coordinate system.
struct Nabla final
{
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
        Pow,
        Deriv,
        Nabla>;

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

// Apply ∂_q to a field, closing the operator into a mark (vibe 000070 P7, marks
// per vibe 000077 step D): a fresh field of the same rank, name and dependence
// as `base`, with q appended to its (sorted) applied-derivative marks.  `base`
// must be a field (its TensorObject carries a FieldDeps trait);
// `coord_name`/`coord` identify q.  Applying again appends further marks, kept
// sorted so mixed partials coincide.
[[nodiscard]] auto make_field_derivative(
    Context&,
    Expr const* base,
    TensorName coord_name,
    CoordinateRef coord) -> Expr const*;

// Apply a *free-index* ∂_i to a field (vibe 000078): like make_field_derivative
// but the direction is the summation index `dir` (a CountableIndex tied to a
// frame vector e_i), recorded as a mark with `link = dir.id` and `free_slot`
// (the index occurrence descriptor).  `base` must be a field depending on all
// coordinates (a specific dependence set cannot license a uniform ∂_i).
[[nodiscard]] auto make_field_derivative_free(
    Context&,
    Expr const* base,
    CountableIndex dir,
    IndexSlot free_slot) -> Expr const*;

// A rank-0 coordinate-*direction* object q_i (vibe 000078): a coordinate atom
// carrying a `CountableIndex` slot, used as the `wrt` of a free-index ∂_i
// operator.  Its presence of a countable slot on a coordinate is what marks the
// derivative direction as an (Einstein-summed) frame index rather than a fixed
// coordinate.
[[nodiscard]] auto make_coordinate_direction(
    Context&, TensorName, int chart_id, CountableIndex dir, IndexSlot slot)
    -> Expr const*;

// Elementary scalar function and power (vibe 000069 M1).
[[nodiscard]] auto make_scalar_fn(Context&, ScalarFnKind, Expr const* operand)
    -> Expr const*;
[[nodiscard]] auto make_pow(Context&, Expr const* base, Expr const* exponent)
    -> Expr const*;

// The unapplied differential operator ∂/∂(wrt) (vibe 000077).  `wrt` is the
// tensor object differentiated against (a coordinate for now).
[[nodiscard]] auto make_deriv(Context&, Expr const* wrt) -> Expr const*;

// The chart-free ∇ operator (vibe 000078): a rank-1 invariant vector operator.
[[nodiscard]] auto make_nabla(Context&) -> Expr const*;

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

// The rank of an expression, or nullopt when a sub-object's rank is unknown or
// the tree is ill-formed (a contraction removing more legs than the operands
// have).  A pure structural query over the node kinds — the scalar/contraction
// factories consult it to keep a scalar out of a contraction slot (`s·T →
// s⊗T`).
[[nodiscard]] auto infer_rank(Expr const* e) -> std::optional<int>;

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

// Dimensioned identity: rank 2 with two unbound slots carrying `space`, so its
// dimension is known (tr(I) → n).  Matched by well-known kind like the bare I.
[[nodiscard]] auto make_identity(Context&, IndexSpace const* space)
    -> Expr const*;

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
