# AST design: plain data with std::variant

_First discussed: 2026-06-13_

---

## Why not a virtual hierarchy

Attempt 01 built the expression tree as an abstract class hierarchy
(`Expr` base, concrete subclasses, virtual `latex()`, `rank()`, etc.).
The interface was not useful enough to justify the cost:

- Adding a new operation over the tree (e.g. a new simplification pass)
  requires a visitor pattern bolted on top — the virtual interface does
  not help.
- `std::visit` over `std::variant` gives exhaustiveness checking for
  free: if a new node type is added to the variant, every visitor that
  does not handle it is a **compile error**.  Forgetting an override in
  a virtual hierarchy is a silent bug.
- Nodes are plain data — they have no behaviour that needs to vary
  polymorphically.  The `variant` expresses this intent directly.
- No vtable pointers, no heap-allocated subclass instances hidden behind
  base pointers, no `dynamic_cast`.

## The `Expr` type

`Expr` is a type alias for a `std::variant` of all concrete node types.
Nodes that reference child expressions hold `Expr const*` — a plain
pointer into Context-owned memory.  The variant is never nested by
value; only by pointer, so the definition is non-recursive and complete.

```cpp
using Expr = std::variant<
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
    NoSum
>;
```

All `Expr` objects are allocated via `Context::make<Expr>(...)` and are
**immutable** after construction.  The `const` in `Expr const*` enforces
this.  Rewriting an expression produces new nodes in the same (or a
child) Context; the original nodes are never modified.

Structural sharing is safe and free: multiple pointers may alias the
same `Expr` node because it is const and Context-owned.

---

## Node definitions

### Leaf nodes

```cpp
// A named tensor with its complete slot layout and index fillings.
//
// `slots` is the ordered list of all positional cells: VoidSlot cells
// have no index; IndexSlot cells each consume one entry from `indices`
// (in slot order, skipping VoidSlots).  The invariant that must hold:
//   indices.size() == count of IndexSlot alternatives in slots.
//
// A rank-0 scalar object has an empty slots vector.
struct TensorObject {
    TensorName              name;
    std::vector<Slot>       slots;    // slot layout (VoidSlot or IndexSlot)
    std::vector<IndexAssoc> indices;  // one per IndexSlot, in slot order
};

// A numeric scalar literal.
struct ScalarLiteral {
    Rational value;
};
```

### Sources of slots on TensorObject

Slots arise from four distinct contexts:

**1. Basis vectors and cobasis vectors.**
When a coordinate system is introduced and a tensor is expanded in its
basis, the individual basis vectors `gᵢ` / `gⁱ` become `TensorObject`
nodes.  Their slots encode the level and index space of the basis:

- covariant basis vector `gᵢ`: one `IndexSlot{Lower, Oblique, space}`
- contravariant basis vector `gⁱ`: one `IndexSlot{Upper, Oblique, space}`
- Orthonormal frame `eᵢ`: one `IndexSlot{Lower, Orthonormal, space}`

**2. Coordinates in a basis expansion.**
When an abstract tensor **A** is expanded as `Aⁱʲ gᵢ ⊗ gⱼ`, the
scalar array `Aⁱʲ` appears as a `TensorObject` with name `A` and two
`IndexSlot`s whose level and space match the dual of each basis slot:

- coordinate `Aⁱʲ`: `[IndexSlot{Upper, Oblique, space}, IndexSlot{Upper, Oblique, space}]`

The component `TensorObject` is rank-0 from the expression tree's
point of view — the slots carry the free indices, but the node itself
is a scalar factor.

**3. Well-known tensors with hardcoded slot layouts.**
Some tensor objects have a canonical, basis-independent slot layout.
They are modelled as abstract tensors — not as coordinates in a
specific WCS — because their components are identical in every basis
(they are isotropic).  Factory functions generate the slot layout from
their parameters; the caller supplies only the index associations.

**Kronecker delta δ.**
Takes one realm, one index space, and **two independent levels** — one
per slot.  Oblique realm imposes no constraint on levels; all four
combinations (upper+upper, upper+lower, lower+upper, lower+lower) are
valid and arise in practice.

```
make_delta(ctx, realm, space, level0, level1, index0, index1)
  → TensorObject{ name="delta",
                  slots=[IndexSlot{level0, realm, space},
                         IndexSlot{level1, realm, space}],
                  indices=[index0, index1] }
```

**Levi-Civita symbol ε.**
Rank N where N = `space->values().size()`.  Takes one realm, one index
space, and **N independent levels** — one per slot.  Practically
important cases are N = 2 (surface/shell, complex-plane identities) and
N = 3 (3D mechanics).

```
make_levi_civita(ctx, realm, space, levels, indices)
    // levels.size() == indices.size() == space->values().size()
  → TensorObject{ name="epsilon",
                  slots=[IndexSlot{levels[k], realm, space}] for k in 0..N-1,
                  indices=indices }
```

**Metric tensor — not a built-in.**
What is commonly called "the metric tensor g" is not an independent
tensor; it is the matrix of covariant (or contravariant) coordinates of
the identity tensor **I** in a given basis.  There is therefore no
`make_metric` factory.  The components are obtained directly from the
definition:

```
g_{ij} = eᵢ @ I @ eⱼ
```

where `eᵢ` and `eⱼ` are basis vectors and `I` is the identity tensor
(a `TensorObject` with its own hardcoded slot layout).  This formula is
sufficient and avoids introducing a redundant special case.

**4. User-attached Collection and Label slots.**
A user may voluntarily attach Collection or Label index slots to any
tensor object to distinguish it within a family.  Examples:

- eigenmode index: `v[Collection index n]`
- region label: `A[Label "vol"]`

These slots are added explicitly when the tensor object is constructed
and carry no summation semantics beyond what the user declares (always
Collection = no auto-sum; Label = no summation at all).

### Unary node

```cpp
struct Negate {
    Expr const* operand;
};
```

### Binary operation nodes

All share the same shape; the type tag encodes the operation.

```cpp
struct Sum          { Expr const* left; Expr const* right; };
struct Difference   { Expr const* left; Expr const* right; };
struct TensorProduct{ Expr const* left; Expr const* right; };  // *  (⊗)
struct ScalarDiv    { Expr const* left; Expr const* right; };  // /  (right must be rank-0)
struct Dot          { Expr const* left; Expr const* right; };  // @  (·)
struct DDot         { Expr const* left; Expr const* right; };  // :
struct DDotAlt      { Expr const* left; Expr const* right; };  // // (··)
struct Cross        { Expr const* left; Expr const* right; };  // %  (×)
```

### Summation annotation nodes

```cpp
// Force summation over `index` in `body`.
// When `bound` is nullptr the range is taken from the index's slot space
// (which must have a non-empty concrete value set).
// When `bound` is non-null it is a symbolic scalar expression giving the
// upper bound, and the index's space may have an empty value set
// (parametric cardinality — see vibe 000028).
struct ExplicitSum {
    CountableIndex index;
    Expr const*    body;
    Expr const*    bound;  // nullptr for concrete-range spaces
};

// Suppress automatic summation for `index` in `body`, making it free
// even when the implicit Einstein rule would contract it.
struct NoSum {
    CountableIndex index;
    Expr const*    body;
};
```

---

## First-class basis

A basis is a first-class object, not a derived concept.  Two routes
produce one:

1. **From a coordinate system** — the covariant basis vectors
   `gᵢ = ∂x/∂qⁱ` are derived automatically from the embedding map.
   (This is the path described in vibe 000004.)

2. **From a user-supplied tuple of vectors** — the user provides N
   named vectors and declares:
   - **Level**: whether the vectors carry lower or upper indices
     (i.e. are they covariant `eᵢ` or contravariant `eⁱ`?)
   - **Realm**: Orthonormal or Oblique

   This makes, for example, a Cartesian frame `{e₁, e₂, e₃}` with
   lower Orthonormal indices a fully valid basis without any coordinate
   system definition.

In both cases the result is the same structure: a named collection of
`TensorObject` nodes, each carrying one `IndexSlot` whose level, realm,
and space are determined by the basis declaration.  The index space is
shared across all vectors of the same basis (it defines the summation
range when expanding in that basis).

The distinction between the two construction routes is an implementation
detail — once built, a basis from a coordinate system and a
user-supplied basis are indistinguishable to the expression tree and to
all simplification passes.

This has a direct consequence for the AST: the "source of slots" for a
basis vector `TensorObject` is always the basis it belongs to, regardless
of how that basis was constructed.  The basis object is the authority
on level, realm, and space for all its member vectors.

## Memory ownership

Every `Expr` node lives in a `Context` (via `Context::make<Expr>(...)`).
The Context's `ResourceList` is the sole owner; raw pointers are used
everywhere else.  The lifetime contract is:

- An `Expr const*` is valid for as long as the `Context` that allocated
  it (or any ancestor Context sharing the same resource list root) is
  alive.
- Child pointers within a node may point into a parent Context — this is
  safe because parent Contexts outlive their children.

---

## Factory functions

Each node type has a corresponding free function that allocates into a
given Context and returns `Expr const*`:

```cpp
auto make_tensor_object(Context&, TensorName,
                        std::vector<Slot>,
                        std::vector<IndexAssoc>) -> Expr const*;
// Convenience: zero-slot scalar tensor object
auto make_scalar_object(Context&, TensorName)   -> Expr const*;
auto make_scalar(Context&, Rational)             -> Expr const*;
auto make_negate(Context&, Expr const*)                                       -> Expr const*;
auto make_sum(Context&, Expr const*, Expr const*)                             -> Expr const*;
auto make_difference(Context&, Expr const*, Expr const*)                      -> Expr const*;
auto make_tensor_product(Context&, Expr const*, Expr const*)                  -> Expr const*;
auto make_scalar_div(Context&, Expr const*, Expr const*)                      -> Expr const*;
auto make_dot(Context&, Expr const*, Expr const*)                             -> Expr const*;
auto make_ddot(Context&, Expr const*, Expr const*)                            -> Expr const*;
auto make_ddot_alt(Context&, Expr const*, Expr const*)                        -> Expr const*;
auto make_cross(Context&, Expr const*, Expr const*)                           -> Expr const*;
auto make_explicit_sum(Context&, CountableIndex, Expr const*, Expr const* bound = nullptr) -> Expr const*;
auto make_no_sum(Context&, CountableIndex, Expr const*)                       -> Expr const*;
```

---

## Visiting with Overloads

`mpk::mix::Overloads` (from `<mpk/mix/util/overloads.hpp>`) is the
standard tool for multi-lambda visitors:

```cpp
std::visit(mpk::mix::Overloads{
    [](TensorObject const& t)  { /* use t.name, t.indices */ },
    [](ScalarLiteral const& s) { /* use s.value */           },
    [](Negate const& n)        { /* recurse into n.operand */},
    [](Sum const& s)           { /* recurse left and right */},
    [](Difference const& d)    { /* ... */                   },
    [](TensorProduct const& p) { /* ... */                   },
    [](ScalarDiv const& d)     { /* ... */                   },
    [](Dot const& d)           { /* ... */                   },
    [](DDot const& d)          { /* ... */                   },
    [](DDotAlt const& d)       { /* ... */                   },
    [](Cross const& c)         { /* ... */                   },
    [](ExplicitSum const& s)   { /* ... */                   },
    [](NoSum const& s)         { /* ... */                   },
}, expr);
```

Omitting any alternative is a **compile error** — exhaustiveness is
enforced structurally, not by convention.

When recursion is needed (e.g. a tree walk), define the visitor as a
named struct so it can call itself:

```cpp
struct DepthCounter {
    auto operator()(TensorObject const&) const -> int { return 0; }
    auto operator()(ScalarLiteral const&) const -> int { return 0; }
    auto operator()(Negate const& n) const -> int {
        return 1 + std::visit(*this, *n.operand);
    }
    auto operator()(Sum const& s) const -> int {
        return 1 + std::max(std::visit(*this, *s.left),
                            std::visit(*this, *s.right));
    }
    // ... remaining alternatives
};
int depth = std::visit(DepthCounter{}, expr);
```

---

## Relation to earlier vibes

| Concept              | Defined in   | Used in AST as           |
|----------------------|--------------|--------------------------|
| `TensorName`         | vibe 000027  | `TensorObject::name`     |
| `IndexAssoc`         | vibe 000028  | `TensorObject::indices`  |
| `CountableIndex`     | vibe 000028  | `ExplicitSum::index`, `NoSum::index` |
| `Rational`           | (existing)   | `ScalarLiteral::value`   |
| `Context`            | vibe 000020  | allocator for all nodes  |
| `Overloads`          | mpk_mix      | visitor construction     |

AST node kinds correspond one-to-one with the semantic tags in vibe 000026.
