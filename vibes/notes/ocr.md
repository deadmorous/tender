# OCR conventions for handwritten tensor notation

This note records the notation conventions used in handwritten derivations in
this project so that future OCR reading (by Claude or a dedicated skill) can
interpret symbols correctly.

## Rank indication by underlining / decoration

| Notation | Rank | Example |
|---|---|---|
| plain letter | 0 (scalar) | $\alpha$, $\lambda$ |
| single underline | 1 (vector) | $\underline{a}$, $\underline{e}_i$ |
| double underline | 2 (second-order tensor) | $\underline{\underline{A}}$, $\underline{\underline{\varepsilon}}$ |
| left superscript | ≥ 3 (higher-order tensor) | ${}^3\!T$, ${}^4\!C$ |

The left superscript convention for rank ≥ 3 is uncommon in printed literature
but appears in hand-written mechanics derivations.  When reading an image,
treat a small numeral to the upper-left of a letter as a rank indicator, not an
exponent.

## Index notation

- Repeated indices (one upper, one lower) imply Einstein summation unless
  explicitly bracketed or otherwise noted.
- Basis vectors are written $\underline{e}_i$ (underlined, subscript index).
- Components are plain: $a_i$, $b^j$.
- The Levi-Civita symbol is $\varepsilon_{ijk}$ (no underline — it is a scalar
  component, not a tensor in the coordinate-free sense when written this way).
- The Kronecker delta is $\delta_{ij}$ or $\delta^i_j$.

## Determinant notation

A 3×3 determinant is written with vertical bars:
$$\begin{vmatrix} a & b & c \\ d & e & f \\ g & h & i \end{vmatrix}$$

In derivations involving ε, the entries are typically Kronecker deltas
$\delta_{i\alpha}$ where $i$ is a free index and $\alpha \in \{1,2,3\}$ is the
column (or row) selector.

## Common ambiguities to watch for

- `i` vs `j` vs `l` (lowercase L) — context (index already used) usually
  disambiguates; flag if truly unclear.
- A numeral `1` as an index vs the identity scalar — check surrounding context.
- Cyclic permutation steps like $\varepsilon_{ijk} = \varepsilon_{kij}$ may be
  written without comment; the sign is +1 for even permutations.
- A small superscript to the upper-left should be read as a rank indicator
  (see above), not a power.

## Future work

A dedicated `ocr` Claude skill could:
1. Accept a path to a scanned image.
2. Apply these conventions to produce a LaTeX transcription.
3. Optionally map the derivation onto tender proof steps.
