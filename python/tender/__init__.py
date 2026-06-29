"""tender — tensor algebra library for computational mechanics."""

from ._core import (
    Rational,
    Realm,
    Level,
    Context,
    CountableIndex,
    IndexSpace,
    IndexNameMap,
    tensor,
    scalar,
    identity,
    coordinate,
    sin,
    cos,
    tan,
    exp,
    log,
    sqrt,
    tr,
    vec,
    transpose,
    delta,
    levi_civita,
    explicit_sum,
    no_sum,
    alloc_index,
    render_latex,
    _space_2d,
    _space_3d,
    _space_4d,
)

# Module-level index-space singletons (call once; pointers are stable).
space_2d: IndexSpace = _space_2d()
space_3d: IndexSpace = _space_3d()
space_4d: IndexSpace = _space_4d()

__all__ = [
    # Numeric type
    "Rational",
    # Enumerations
    "Realm",
    "Level",
    # C++ context and types
    "Context",
    "CountableIndex",
    "IndexSpace",
    "IndexNameMap",
    # Expression factories
    "tensor",
    "scalar",
    "identity",
    "coordinate",
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
    "tr",
    "vec",
    "transpose",
    "delta",
    "levi_civita",
    "explicit_sum",
    "no_sum",
    "alloc_index",
    # Rendering
    "render_latex",
    # Predefined index spaces
    "space_2d",
    "space_3d",
    "space_4d",
]
