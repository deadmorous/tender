# Expression grammar for tensor algebra

## Motivation

Attempt 01 grew an ad-hoc expression tree without a written grammar.
Attempt 02 starts from the grammar and derives the AST from it, so that
the four surfaces — Python API, AST, formatted output, and a possible
text DSL — are clearly related to one another.

## Core constraint: only named objects carry indices

In coordinate-free (direct) notation the index is a *label attached to a
named tensor*, not a coordinate selector applied to a computed value.
This means `(A + B)[i,j]` is not a valid tensor expression — the sum has
no name to attach indices to.  The grammar enforces this by allowing
`IndexList` only on `TensorObject` (a bare identifier).

## Grammar

```
Expression   ::= ("-")? Term (add Term)*

Term         ::= Factor (mul Factor)*

Factor       ::= Atom
               | BracketExpr

BracketExpr  ::= "(" Expression ")"

Atom         ::= TensorObject
               | UnsignedNumber

TensorObject ::= Identifier IndexList?

IndexList    ::= "[" Index ("," Index)* "]"

Index        ::= Identifier        # symbolic index
               | UnsignedInteger   # concrete slot number (if needed)

add  ::= "+" | "-"
mul  ::= "*" | "@" | "%" | "//" | ":" | "/"
```

### Operator meanings

| Symbol | Tensor operation             | Notes                                                     |
|--------|------------------------------|-----------------------------------------------------------|
| `*`    | tensor product ⊗             | rank-0 case is scalar scaling                             |
| `/`    | scalar division              | denominator must be rank-0                                |
| `@`    | single contraction ·         | Python matmul; `A @ B` contracts last slot of A with first slot of B |
| `:`    | double contraction (ddot)    | `A : B = Aᵢⱼ Bᵢⱼ = tr(AᵀB)`; contracts first–first and second–second slots |
| `//`   | double contraction, alt. order | contracts *closest* free slots to the operator first, then next: `A // B = Aᵢⱼ Bⱼᵢ = tr(AB)` |
| `%`    | cross product ×              | raises error if chained without parentheses (not associative) |

The distinction between `:` and `//` matters for tensors of rank ≥ 2
that are not symmetric.  For a symmetric rank-2 tensor `A : B = A // B`.

The clearest way to see the difference is via dyads (rank-2 tensors
written as tensor products of vectors):

```
(a*b) :  (c*d)  =  (a@c) * (b@d)   # first-of-left ↔ first-of-right,
                                    # second-of-left ↔ second-of-right
(a*b) // (c*d)  =  (b@c) * (a@d)   # last-of-left  ↔ first-of-right (closest),
                                    # first-of-left ↔ last-of-right  (next)
```

The same principle extends to higher-rank tensors: `:` contracts the
outermost free-slot pairs, `//` contracts the innermost ones first.
See vibe 000005 for the original motivation.

Note: vibe 000005 assigned `**` to the reversed double contraction.
`**` has power precedence in Python (higher than mul), so it does not
fit alongside `*`, `@`, `:`, `//` at the same level.  The grammar here
uses `//` for the alternate ordering at mul precedence, and `**` is
dropped from the operator set.

### Cross product non-associativity

`%` is not associative: `(a × b) × c ≠ a × (b × c)` in general.
Therefore `a % b % c` must be a construction-time error; users must
write `(a%b)%c` or `a%(b%c)` explicitly.  The free function `cross(a,b)`
has no such ambiguity and should be preferred when chaining.

## Precedence and associativity

All `mul` operators share a single precedence level and are
left-associative.  This means `A @ B @ C = (A @ B) @ C`, which is
correct for non-commutative operations.  Explicit parentheses must be
used whenever a different grouping is intended.

In hand-written mathematics `:` (double contraction) is sometimes
written as if it binds more loosely than `·` (single contraction), but
encoding separate precedence levels here would complicate the grammar
without clear benefit; parentheses are the explicit, portable solution.

Unary minus attaches to the whole `Expression`, so `-A * B = -(A * B)`.
Inside a `BracketExpr` a new `Expression` starts, so `(-A) * B` is also
grammatically valid and means what it says.

## AST uniqueness

The grammar is unambiguous: the parse tree is fully determined by
precedence and left-associativity, so the AST is unique for any given
input.

Two representation choices for the AST nodes:

1. **Binary nodes** — faithful to the grammar; `A + B + C` becomes a
   left-leaning binary tree.  Requires explicit associativity rules for
   pattern matching over sums.

2. **Flat n-ary nodes** — additions and same-operator products are
   collapsed into a child list.  Pattern matching is simpler, but the
   representation no longer encodes parse order; safe only for operators
   that are provably associative in this algebra.

**Decision: binary nodes** are the canonical AST.  N-ary normalisation
is an explicit optional pass, applied only to provably-associative
operators.

### Semantic tags in AST nodes

AST nodes store semantic tags, not operator strings.  The operator
symbols (`*`, `@`, `:`, …) are surface syntax; the AST uses typed node
kinds independent of any surface:

| AST node kind | Operator(s)   |
|---------------|---------------|
| `TensorProduct` | `*`         |
| `ScalarDiv`     | `/`         |
| `Dot`           | `@`         |
| `DDot`          | `:`         |
| `DDotAlt`       | `//`        |
| `Cross`         | `%`         |
| `Sum`           | `+`         |
| `Difference`    | `-` (binary)|
| `Negate`        | `-` (unary) |

This makes the AST portable across surfaces: the Python API, a future
DSL parser, and any programmatic builder all produce the same node types.

## Index syntax: `[]` canonical

The Python surface uses `A[i, j]` (subscript via `__getitem__`) to
annotate a named tensor with symbolic indices.  Reasons:

- Avoids confusion with method calls (`A(i,j)` looks like calling A).
- Matches NumPy/SymPy conventions.
- `__getitem__` with a tuple argument is straightforward to implement.

`A[i]` returns a new `TensorObject` node carrying the index list as
metadata; it does not perform any numerical indexing.

## Differences across the four surfaces

### Python API

All six `mul` operators and both `add` operators map to Python dunder
methods on expression objects.  No juxtaposition is possible in Python;
every binary operation requires an explicit symbol.

### AST

The grammar above *is* the AST grammar.  `IndexList` is metadata on the
`TensorObject` node, not a child node.  The tree shape is therefore
independent of whether a tensor happens to carry indices.

### Formatted output

The formatter may suppress operator symbols for readability:
- `@` (single contraction) → `·` in LaTeX
- `:` → `:`  or `∶` glyph
- `//` → `··`
- `%` → `×`
- `*` → juxtaposition (default); `⊗` only when explicitly requested

Indices render as subscripts/superscripts, not `[…]`.

A display-only convention: if a sum `A + B` is subsequently bound to a
named object `C`, the formatter may render `Cᵢⱼ` for the named result.
The AST always carries the index on the named node; the bracket form is
purely cosmetic.

### Text DSL

A future text DSL can accept juxtaposition and standard mathematical
index notation (`Aᵢⱼ`, `A_{ij}`).  The DSL parser desugars these into
the same AST nodes produced by the Python API.  The grammar above is a
strict syntactic subset of the DSL.

## Atom taxonomy

There are two distinct kinds of `Atom` and they must not be unified:

| Kind            | Can carry `IndexList`? | Examples              |
|-----------------|------------------------|-----------------------|
| `TensorObject`  | yes                    | `A`, `sigma[i,j]`     |
| `UnsignedNumber`| no                     | `2`, `3`              |

A named rank-0 scalar (e.g. `alpha`) is a `TensorObject` — it has a
name and could in principle be given a zero-length index list, even
though it has no free slots.  A numeric literal is never a tensor object;
it carries no name and cannot be indexed.

## `nabla` is a `TensorObject`

`nabla` (∇) is a rank-1 named object in the grammar — a `TensorObject`
with a pre-assigned name.  Its differential semantics (it differentiates
whatever it is combined with) are handled by the evaluator and
simplification rules, not by the grammar or the AST node type.  At the
expression-building level it participates in all the same operations as
any other rank-1 tensor object.
