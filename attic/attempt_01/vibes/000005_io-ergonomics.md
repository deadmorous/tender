# I/O Ergonomics: Input Syntax and Output Rendering

## Status: Q_python_operators resolved; Q_named_tensor_syntax, Q_python_output, Q_derivation_display open

---

## Input: Python API

Python is the sole user-facing interface (see vibe 000004, Q_ui_cppapi).
The challenge: map tensor algebra notation onto Python syntax cleanly.

### Q_python_operators — RESOLVED

**Decision**: support both operators and free functions; user chooses what reads better.
Operators use only symbols with precedence ≥ multiplication (discarding `^`, `|`, `&`, etc.).
No unary/binary operators as methods — all algebraic operations are free functions.
Methods are reserved for other purposes (to be determined).

#### Operator table

| Python op | Precedence | Tensor meaning | Notes |
|---|---|---|---|
| `+` | add | addition | standard |
| `-` | add | subtraction / negation | standard |
| `*` | mul | tensor product ⊗; scalar scale is the rank-0 case | no type-dispatch ambiguity: scaling IS multiplication by a rank-0 tensor |
| `/` | mul | division by scalar | |
| `%` | mul | cross product × | `^` excluded — its precedence is lower than `+` |
| `@` | mul | single contraction · | matches matrix-multiply intuition |
| `**` | power | double contraction reversed ·· | higher precedence than `*`, which is fine |
| `//` | mul | double contraction : (Frobenius) | |

Note: `**` having higher precedence than `*` means `A ** B * C` parses as `(A··B) * C`,
which is correct (scalar result scaled). Check all mixed-precedence cases during implementation.

**Non-associativity of `%` (cross product)**: `*`, `@`, `//`, `**` are all associative,
so Python's default left-to-right chaining is harmless. Cross product is not associative:
`(a×b)×c ≠ a×(b×c)` in general. Therefore `a % b % c` must raise an error at construction
time rather than silently evaluating left-to-right. Users must parenthesise explicitly:
`(a%b)%c` or `a%(b%c)`. The free function `cross(a, b)` does not have this ambiguity
(it is binary by definition) and should be preferred when chaining is needed.

#### Free function API (always available, safe default)

```python
# binary operations
tp(a, b)           # tensor product: a⊗b  (also: a*b)
dot(a, b)          # single contraction: a·b  (also: a@b)
ddot(A, B)         # double contraction: A:B  (also: A//B)
ddot2(A, B)        # reversed double contraction: A··B  (also: A**B)
cross(a, b)        # cross product: a×b  (also: a%b)

# differential operators (nabla is a first-class symbolic object)
grad(f)            # ∇f — gradient of scalar or tensor field
div(v)             # ∇·v — divergence
rot(v)             # ∇×v — curl / rotation
lap(f)             # ∇²f = ∇·(∇f) — Laplacian; also available as object `lap`

# partial derivative
deriv(x, expr)     # ∂expr/∂x

# integral
integral(domain, expr)       # ∫_domain expr dV (or dA, dL — inferred from domain)
boundary(domain)             # ∂domain — boundary of a domain, for use in integrals

# rank-2 tensor invariants and operations
trace(A)           # tr A
vec_inv(A)         # vector invariant (second invariant family)
det(A)             # det A
transpose(A)       # Aᵀ
permute(T, ...)    # general index permutation for rank ≥ 2
```

#### The nabla object

`nabla` is a symbolic differential operator object. Combined via operators:

```python
nabla @ v          # divergence: ∇·v
nabla * v          # gradient: ∇⊗v (tensor product gives gradient tensor)
nabla % v          # curl: ∇×v
nabla @ (nabla * f)  # Laplacian: ∇·(∇f); shorthand: lap(f)
```

`lap` is also available as a standalone object equivalent to `nabla @ nabla`.

---

## Input: named tensors and index control

### Q_named_tensor_syntax — RESOLVED

#### World Cartesian System (WCS)

The WCS is the root coordinate system — the fixed reference frame everything
else is anchored to. It is importable and provides its basis directly:

```python
from tender import wcs
i, j, k = wcs.basis    # or wcs.i, wcs.j, wcs.k
```

All other coordinate systems are eventually linked to the WCS via a chain of
transformations (see below).

#### Constructing tensors from basis vectors

Any tensor can be built up from basis vectors via tensor product and addition.
For example, the identity tensor:

```python
I_expr = i*i + j*j + k*k
```

This does not mean `I` must be constructed this way — well-known tensors are
importable directly. The construction above is the underlying definition.

#### Named tensors: attaching a display name to an expression

Every expression node carries an optional `name` attribute (empty by default).
Setting it changes how the node is rendered and serialised — it does not create
a new wrapper object or a new type.

```python
from tender import I, eps, nabla   # well-known importable singletons; name pre-set

# User-defined: attach a display name to any expression
sigma = named('sigma', some_expression)
```

`named(name, expr)` sets `expr.name = name` and returns `expr`. It exists purely
as a one-liner convenience — the two-line form `expr.name = 'sigma'` is equivalent
but requires the expression to already be bound to a variable.

Setting a name is idempotent (setting the same name twice is fine) but setting a
different name on an already-named expression is an error.

The name is used by the LaTeX renderer (`\boldsymbol{\sigma}`) and by `python()`
output. Well-known tensors like `I` and `eps` have their names pre-set.

#### Indices with domains

An index is an object that knows the values it ranges over. Einstein summation
applies when an index appears exactly twice (upper + lower position; position
relaxed for orthonormal bases):

- Appears once → no summation (free index)
- Appears twice → automatic summation over its domain
- Appears three or more times → error at construction time

```python
i = Index('i', domain=range(3))          # spatial 3D, auto-sum
n = Index('n', domain=range(N))          # FEM shape functions, auto-sum
```

Shortcut constructors for common cases:

```python
i = AutoSumIndex3d('i')    # domain=range(3), auto_sum=True
i = AutoSumIndex2d('i')    # domain=range(2), auto_sum=True
```

The `ExplicitSum` and `NoSum` annotation wrappers from vibe 000004 (Q_index_slots)
remain available for overriding auto-sum behaviour in edge cases.

#### Coordinate systems and transformations

Every coordinate system (CS) is created relative to a reference CS, with a
transformation that maps between them. All CS are thereby linked back to the WCS
through a chain of transformations.

A transformation has two components, both optional (default: identity/zero):
- A rotation tensor (rank-2, orthogonal); identity rotation is zero angle about any axis
- A translation vector

```python
local = CS(reference=wcs, rotation=R, translation=t)
local = CS(reference=wcs)          # identity rotation, zero translation
```

The five CS construction paths (built-in, embedding map, metric, curve/Frenet-Serret,
surface — see vibe 000001) describe how the basis vectors of a CS are derived, which
is a separate concern from the transformation used to position it relative to its
reference CS.

The basis vectors of any CS are accessible as `cs.basis`, cobasis as `cs.cobasis`,
and the metric tensor as `cs.metric`.

---

## Output: rendering targets

### Target 1 — LaTeX string

```python
expr.latex()    # returns str, e.g. r"\mathbf{A} : \mathbf{B}"
```

Always available; useful for copy-paste into papers and for programmatic use.

### Target 2 — Rendered LaTeX in Jupyter

Implement `_repr_latex_()` on expression objects so that Jupyter/IPython
automatically renders them as math when a cell evaluates to an expression.
This is the standard Jupyter protocol (used by SymPy, pandas, etc.) — no extra
user action required.

```python
A + B   # Jupyter displays rendered equation automatically
```

Step-by-step derivations would be a sequence of rendered cells, or a single
display call that renders each step with annotations:

```python
derivation.display()   # renders each step as LaTeX with step names
```

### Target 3 — Python code as output

### Q_python_output — RESOLVED

**Use case**: compositional derivations. The output of one derivation becomes
the input of the next. `expr.python()` emits a valid Python string that
reconstructs the expression, so it can be copy-pasted or `eval`'d into a new
session or notebook cell.

**Variable name tracking**: Python does not expose assignment-time variable names
reliably for ordinary variables (frame inspection via `sys._getframe` is fragile).

**Decision**: named tensors carry their name explicitly at construction time:

```python
sigma = tensor('sigma')    # object knows its name is 'sigma'
u     = tensor('u')
```

`expr.python()` uses these stored names when emitting code. Anonymous intermediate
nodes (results of operations not assigned to a named tensor) get generated names
`_t0`, `_t1`, … in the output. The emitted code is valid Python that `eval`'s back
to the same expression tree.

This also means the LaTeX renderer can use the stored name as the symbol, making
output like `\boldsymbol{\sigma}` automatic.

---

## Output: derivation steps

### Q_derivation_display — RESOLVED

**Full history is kept.** Memory is not a concern (gigabytes available; each step
is kilobytes). Display is user's choice: show only the final result, or show all steps.

#### Core architectural concept: State vs. Derivation

These are two distinct objects:

- **State**: a pure immutable snapshot — just an expression. No history inside it.
  History is not a property of a state; it is a property of a sequence of states.

- **Derivation**: a recipe — an ordered, named list of steps. Takes a State as
  input, produces a sequence of States as output (one per step, including the
  initial). History lives in that sequence, not in any individual state.
  A derivation sees only the state it is given; it cannot observe how that state
  was reached. This keeps derivations clean and reusable.

```python
# Sketch of the model
s0 = State(initial_expr)           # immutable snapshot

deriv = Derivation([               # reusable recipe
    expand_grad,
    contract,
    apply_divergence_theorem,
    localize,
])
history = deriv.apply(s0)          # returns list of States, one per step

# Feed output into another derivation
history2 = another_deriv.apply(history[-1])

# Display options
show(history)                      # renders all steps
show(history, final_only=True)     # renders only the last state
```

Derivations are deterministic: the same recipe applied to the same State always
produces the same sequence. Derivations are composable: `deriv_a + deriv_b`
produces a combined recipe.

#### Display format (Jupyter)

Each step renders as:

```
[step name]
  ← <LaTeX: expression before>
  → <LaTeX: expression after>
```

`state.show()` renders the full sequence. `state.result` (or evaluating `state`
in a Jupyter cell) renders only the current expression via `_repr_latex_()`.

---

## Longer-term: DSL

A domain-specific language (not Python) could allow notation closer to
mathematical writing, e.g.:

```
∇ · (σ · δu) = (∇ · σ) · δu + σ : ∇δu
```

This is out of scope for Tier 1 but worth keeping in mind so the Python API
does not box us in. In particular, the expression tree should be
language-agnostic — the Python API is one front-end, a future DSL parser
would be another.
