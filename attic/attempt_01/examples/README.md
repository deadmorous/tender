# tender — examples

Each example comes in two forms: a plain Python script (`.py`) and a Jupyter
notebook (`.ipynb`).  The scripts are good for scripted runs and CI; the
notebooks render each derivation step as formatted LaTeX in the browser.

## Prerequisites

Build the Python bindings once:

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DTENDER_BUILD_PYTHON=ON \
  -DTENDER_BUILD_TESTS=OFF
cmake --build build
```

Then source `env.sh` to put the package on `PYTHONPATH`:

```bash
source examples/env.sh           # uses build/
source examples/env.sh build-py  # alternative build directory
```

## Running the scripts

```bash
python examples/pvw_continuum.py
python examples/bac_cab.py
python examples/bac_cab_search.py
```

Or use the Makefile, which also compiles the generated `.tex` to PDF and opens
the result:

```bash
make -C examples                   # all examples
make -C examples bac_cab_search    # one example
make -C examples VIEWER=evince     # custom PDF viewer
```

Output files (`.tex`, `.pdf`, `.aux`, `.log`) are written to `examples/out/`
which is git-ignored.

## Running the notebooks

```bash
cd examples
source env.sh
jupyter lab
```

Open any `.ipynb` file and run all cells (`Kernel › Restart Kernel and Run All
Cells`).  Each step of the derivation is rendered as a displayed LaTeX equation.

## Available examples

| File | Description |
|---|---|
| `pvw_continuum.py` / `.ipynb` | Principle of virtual work — continuum mechanics derivation |
| `bac_cab.py` / `.ipynb` | BAC-CAB identity — manual and automatic application |
| `bac_cab_search.py` / `.ipynb` | BAC-CAB via `search_apply` — automatic rewrite search |

## Notebook hygiene — stripping outputs before committing

Notebooks store cell outputs (rendered LaTeX, printed text) and execution
counts in JSON.  Committing those clutters diffs and can embed binary image
data.  The rule is: **notebooks must be clean (no outputs, no execution counts)
before every commit.**

CI enforces this in the `notebooks-clean` job, which rejects any notebook
whose code cells contain stored outputs or a non-null `execution_count`.

### Stripping manually

Use the provided script, which runs `nbstripout` on every `.ipynb` in this
directory:

```bash
examples/strip_notebooks.sh        # asks for confirmation
examples/strip_notebooks.sh -y     # strips without asking
```

Or strip a single notebook:

```bash
nbstripout examples/bac_cab_search.ipynb
```

### Automating with git hooks (recommended)

Install `nbstripout` as a git filter so stripping happens automatically on
`git add`:

```bash
pip install nbstripout
nbstripout --install          # installs into .git/config and .gitattributes
```

After that, running `git add` on a notebook automatically strips it before
staging.  You never have to think about it again.
