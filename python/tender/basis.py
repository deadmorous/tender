"""tender.basis — vector bases and coordinate systems.

The bridge between the invariant (direct-notation) layer and the coordinate
layer (vibe 000049).  A :class:`Basis` is built from a tuple of rank-1 vectors
or obtained from a well-known coordinate system; the basis-parameterized steps
expand an invariant into coordinates and fold it back::

    import tender
    import tender.basis as tb
    import tender.derivation as td

    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    frame = tb.wcs(ctx)

    # a · b, reduced through the basis to the scalar contraction Σ_i a_i b_i.
    e = a @ b
    e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
    e = tb.simplify_basis_dot(e, frame)
    e = td.canonicalize(e)
    e = td.unroll_sums(e)
    e = td.eval_delta_concrete(e)
    e = td.fold_arithmetic(e)
    e = td.fold_sums(td.canonicalize(e))
    print(e.latex())   # "\\sum_{i} a_{i} \\, b_{i}"

The steps take ``(expr, basis, ...)``; wrap them in a ``lambda`` to drive a
:class:`tender.derivation.Derivation`.
"""

from tender._core import basis as _b

Basis = _b.Basis
Variance = _b.Variance
make_orthonormal_basis = _b.make_orthonormal_basis
make_oblique_basis = _b.make_oblique_basis
wcs = _b.wcs
cylindrical = _b.cylindrical
spherical = _b.spherical
polar_2d = _b.polar_2d
expand_in_basis = _b.expand_in_basis
simplify_basis_dot = _b.simplify_basis_dot
simplify_basis_cross = _b.simplify_basis_cross
reassemble = _b.reassemble

__all__ = [
    "Basis",
    "Variance",
    "make_orthonormal_basis",
    "make_oblique_basis",
    "wcs",
    "cylindrical",
    "spherical",
    "polar_2d",
    "expand_in_basis",
    "simplify_basis_dot",
    "simplify_basis_cross",
    "reassemble",
]
