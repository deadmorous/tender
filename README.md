# tender

**Tender** is a tensor-algebra library for computational mechanics that works in
**direct (coordinate-free) notation** — the same `∇·T`, `a×(b×c)`, `ε = ∇×(∇×ε)ᵀ`
you write on paper — and reduces, differentiates, and renders those expressions
symbolically. It is a C++20 core with a Python front end.

Where a traditional CAS makes you drop to indices, tender keeps tensors whole:
it carries invariants (`tr`, `vec`, transpose, dot, cross, double contraction),
derives the full orthogonal-curvilinear geometry from a coordinate mapping, and
lowers a coordinate-free `∇` expression onto any chart — so a derivation reads
the way a mechanician would perform it, and every step renders to LaTeX.

```python
import tender as t
from tender.operators import nabla

ws = t.Workspace()

# A cylindrical chart, derived entirely from its embedding x = r cosθ, …
r, th, z = ws.coords("r", "\\theta", "z", nonneg=("r",))
cyl = ws.chart(ws.wcs(), [r, th, z], [r * t.cos(th), r * t.sin(th), z])

u = cyl.field("u", 1)                 # a vector field u(r, θ, z)
div_u = (nabla @ u).evaluate(cyl)     # ∇·u, lowered onto the chart

print(div_u.latex())
# \frac{u_{r} + \partial_{\theta} u_{\theta} + r \, (\partial_{r} u_{r}) + r \, (\partial_{z} u_{z})}{r}
```

## What it does

- **Direct-notation algebra** — build expressions with overloaded operators
  (`*` ⊗, `@` ·, `%` ×, `//` ··, `**`, `+`, `-`, `/`) plus `tr` / `vec` /
  `transpose` / `ddot`. Rank is inferred through every contraction.
- **Algebraic normal form & rewriting** — `canonicalize` / `simplify`, an
  identity-rule engine with equality saturation over an **e-graph**, the ε–δ
  machinery, and the invariant⇄index bridge.
- **Curvilinear geometry, derived** — give a chart its coordinate embedding and
  tender derives the tangent basis `gᵢ`, metric `g_ij`, scale factors `hᵢ`, the
  physical frame `eᵢ`, and the connection `∂ⱼeᵢ` (Christoffel coefficients).
- **Differential operators** — `grad` / `div` / `rot` / `laplacian` on any
  chart, a first-class composable `∇`, and `evaluate` to lower a coordinate-free
  `∇` expression (including `Δ = ∇·∇`) onto a chart — moving-frame terms and all.
- **LaTeX rendering** — every expression renders to LaTeX; notebooks display it
  live via MathJax.

The worked examples take this end-to-end: cylindrical equilibrium to the hoop-
stress formula, the strain-compatibility identity `ε = ∇×(∇×ε)ᵀ`, and the
Navier–Lamé reduction `∇·T = μ∇·∇u + ∇((λ+μ)∇·u)` — each derived *as performed*,
then verified componentwise.

## Documentation

| Where | What |
|---|---|
| [`doc/cheatsheet.md`](doc/cheatsheet.md) | **API cheatsheet** — the layered classes and a do/don't table of every method. Start here. |
| [`examples/README.md`](examples/README.md) | The worked examples (script + notebook forms) and how to run them. |
| [`CLAUDE.md`](CLAUDE.md) | Coding principles and repository conventions. |
| [`vibes/`](vibes/) | The design log — one Markdown file per design topic (`NNNNNN_subject.md`), recording the rationale behind every feature. |

## Building

Tender needs a C++20 compiler (GCC 14+), CMake ≥ 3.25, and Ninja. Third-party
dependencies (the `mpk_mix` utility library, GoogleTest, nanobind) are fetched
automatically by CMake.

**C++ core and tests:**

```bash
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=g++-14
cmake --build build
ctest --test-dir build
```

**Python bindings:**

```bash
cmake -B build-py -G Ninja \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DTENDER_BUILD_PYTHON=ON \
  -DTENDER_BUILD_TESTS=OFF
cmake --build build-py
```

The compiled extension is written straight into `python/tender/`, so no install
step is needed — just put the package on the path:

```bash
source examples/env.sh          # sets PYTHONPATH
python examples/navier_lame.py
pytest python/tests             # the Python test suite
```

See [`examples/README.md`](examples/README.md) for running the notebooks and the
LaTeX/PDF Makefile.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | The C++20 core — expression tree, normal form, e-graph, bases, charts, rendering (headers in `src/include/tender/`). |
| `python/` | nanobind bindings (`_core.cpp`) and the `tender` Python package (`Workspace`, `derivation`, `basis`, `chart`, `operators`, `render`). |
| `tests/` | C++ unit and integration tests (GoogleTest). |
| `python/tests/` | Python test suite (pytest). |
| `benchmarks/` | Micro-benchmarks for performance-sensitive operations. |
| `examples/` | End-to-end derivations as `.py` scripts and `.ipynb` notebooks. |
| `doc/` | Reference documentation. |
| `vibes/` | Design-discussion log. |

## License

GNU General Public License v3.0 — see [`LICENSE`](LICENSE).
