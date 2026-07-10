# tender — examples

Each example comes in two forms: a plain Python script (`.py`) and a Jupyter
notebook (`.ipynb`).  The scripts are good for scripted runs and CI; the
notebooks render the result as formatted LaTeX in the browser.

## Prerequisites

Build the Python bindings once:

```bash
cmake -B build-py -G Ninja \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DTENDER_BUILD_PYTHON=ON \
  -DTENDER_BUILD_TESTS=OFF
cmake --build build-py
```

The compiled `.so` is written directly into `python/tender/` in the repository,
so no additional install step is needed.

Then source `env.sh` to put the package on `PYTHONPATH`:

```bash
source examples/env.sh
```

## Running the scripts

```bash
python examples/kronecker_delta.py
```

Or use the Makefile, which also compiles the generated `.tex` to PDF:

```bash
make -C examples                    # all examples
make -C examples kronecker_delta    # one example
make -C examples clean              # remove out/
```

Output files (`.tex`, `.pdf`, `.aux`, `.log`) are written to `examples/out/`,
which is git-ignored.

## Running the notebooks

```bash
cd examples
source env.sh
jupyter lab
```

Open any `.ipynb` file and run all cells (`Kernel › Restart Kernel and Run All
Cells`).  Expressions with `_repr_latex_()` render as displayed LaTeX via
MathJax.

## Available examples

| File | Description |
|---|---|
| `kronecker_delta.py` / `.ipynb` | Kronecker delta δ^i_j in 3D oblique space — renders to LaTeX |
| `delta_trace.py` / `.ipynb` | Trace of the Kronecker delta δ^i_i = 3 by implicit summation |
| `eps_delta.py` / `.ipynb` | The ε–δ identity ε_ijk ε_ilm = δ_jl δ_km − δ_jm δ_kl |
| `basis_dot_product.py` / `.ipynb` | Dot product through a basis: deriving a·b = b·a from first principles |
| `cross_identity.py` / `.ipynb` | Cross with the identity tensor: deriving the theorem a × I = I × a |
| `basis_aware_coordinates.py` / `.ipynb` | Basis-aware indices & coordinate notation: familiar letters (i,j,k / e_r), basis identity, basis-aware steps, custom naming & labels |
| `curvilinear_operators.py` / `.ipynb` | Curvilinear ∇/div/rot/Δ derived from a coordinate mapping: tangent basis → metric → scale factors → physical frame → ∂e connection → operators (cylindrical & spherical) |
| `cyl_equilibrium.py` / `.ipynb` | Cylindrical equilibrium ∇·T + f = 0 worked through to the thin-pipe hoop-stress formula |
| `strain_compatibility.py` / `.ipynb` | Strain compatibility inc ε = ∇×(∇×ε)ᵀ derived *as performed* — expand ∇, derive & apply the a×B×c cross-removal identity, reassemble into ∇ operators — then verified componentwise (Cartesian & cylindrical) |
| `navier_lame.py` / `.ipynb` | Navier–Lamé reduction ∇·T = μ∇·∇u + ∇((λ+μ)∇·u) of the isotropic Hooke stress derived *as performed* — expand ∇, apply the Leibniz ∂'s, fold e_i·I, reassemble into ∇ operators, reverse the distribution — then verified componentwise (Cartesian & cylindrical) |

## Notebook hygiene — stripping outputs before committing

Notebooks store cell outputs and execution counts in JSON.  Committing those
clutters diffs.  The rule: **notebooks must be clean before every commit.**

Strip manually with the provided script:

```bash
examples/strip_notebooks.sh        # asks for confirmation
examples/strip_notebooks.sh -y     # strips without asking
```

Or strip a single notebook:

```bash
nbstripout examples/kronecker_delta.ipynb
```

To automate stripping on `git add`:

```bash
pip install nbstripout
nbstripout --install
```
