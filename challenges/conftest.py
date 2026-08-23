"""Make `import tender` and `from challenges import harness` work with no
external PYTHONPATH: the bindings are compiled in-place into python/tender/,
so pointing sys.path at <repo>/python (and at <repo> for this package) is all
the setup a human or CI needs to run `pytest challenges`."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
for p in (str(_root / "python"), str(_root)):
    if p not in sys.path:
        sys.path.insert(0, p)
