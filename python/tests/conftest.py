"""Put <repo>/python on sys.path so `import tender` works without an external
PYTHONPATH (the bindings are compiled in-place into python/tender/).  Same
self-sufficiency as challenges/conftest.py: a bare `pytest` from the repo root
runs both suites with no environment setup."""

import sys
from pathlib import Path

_pydir = str(Path(__file__).resolve().parent.parent)
if _pydir not in sys.path:
    sys.path.insert(0, _pydir)
