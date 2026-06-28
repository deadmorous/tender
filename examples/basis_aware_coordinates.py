"""Basis-aware indices and coordinate-system notation (vibe 000067).

Four short sections, one per feature the basis-awareness work made available:

  1. Familiar coordinate-system notation — an expansion in a well-known frame
     prints in that frame's own letters: WCS i, j, k; cylindrical e_r, e_θ, e_z.
  2. Coordinates know their basis — the same invariant expanded in two frames is
     algebraically distinct, yet a shared index still sums *across* frames
     (Einstein summation is basis-blind), so the rotation/two-point tensor
     e_i ⊗ e'_i is well formed.
  3. Basis-aware steps — a step keyed to one frame only acts on that frame:
     e_i^A · e_j^A → δ, but the cross-frame overlap e_i^A · e_j^B is left alone.
  4. Custom naming & display label — build your own frame with coordinate
     letters, standalone vector symbols, and a frame marker (the primed index
     e_{i'}) that distinguishes two frames in one term.

Writes a standalone LaTeX summary to ``out/``.
"""

import pathlib

import tender
import tender.basis as tb
import tender.derivation as td

ctx = tender.Context()

# Each (title, [(label, expr_or_text)]) tuple becomes a section of the report.
report: list[tuple[str, list[tuple[str, str]]]] = []


def show(title, rows):
    """Print a section to stdout and record it for the LaTeX report."""
    print(f"\n{title}")
    print("-" * len(title))
    latex_rows = []
    for label, value in rows:
        text = value.latex() if hasattr(value, "latex") else str(value)
        print(f"  {label:34s} {text}")
        latex_rows.append((label, text))
    report.append((title, latex_rows))


# ---------------------------------------------------------------------------
# 1. Familiar coordinate-system notation (value_names + vector symbols)
# ---------------------------------------------------------------------------

a = tender.tensor("a", rank=1, ctx=ctx)


def expand_unroll(frame):
    """a = a^i e_i, expanded in `frame` and unrolled to concrete directions."""
    e = tb.expand_in_basis(a, frame, tb.Variance.Covariant)
    return td.unroll_sums(td.canonicalize(e))


show(
    "1. Familiar coordinate-system notation",
    [
        ("WCS frame", expand_unroll(tb.wcs(ctx))),
        ("cylindrical frame", expand_unroll(tb.cylindrical(ctx))),
        ("spherical frame", expand_unroll(tb.spherical(ctx))),
    ],
)

# ---------------------------------------------------------------------------
# 2. Coordinates know their basis
# ---------------------------------------------------------------------------

wcs = tb.wcs(ctx)
# A second frame, marked with a prime so it is visibly distinct in one term.
p = tender.tensor("p", rank=1, ctx=ctx)
q = tender.tensor("q", rank=1, ctx=ctx)
s = tender.tensor("s", rank=1, ctx=ctx)
primed = tb.make_orthonormal_basis([p, q, s], tender.space_3d, label="'")

a_in_wcs = td.canonicalize(tb.expand_in_basis(a, wcs, tb.Variance.Covariant))
a_in_primed = td.canonicalize(
    tb.expand_in_basis(a, primed, tb.Variance.Covariant)
)

# The rotation / two-point tensor R = e_i ⊗ e'_i: one index, two frames.
i = ctx.alloc_index()
rotation = td.canonicalize(wcs.covariant_vector(i) * primed.covariant_vector(i))

same_frame = td.algebraic_eq(a_in_wcs, a_in_wcs)
cross_frame = td.algebraic_eq(a_in_wcs, a_in_primed)

show(
    "2. Coordinates know their basis",
    [
        ("a expanded in WCS", a_in_wcs),
        ("a expanded in the primed frame", a_in_primed),
        ("algebraic eq, WCS vs WCS", same_frame),
        ("algebraic eq, WCS vs primed", cross_frame),
        ("rotation tensor (one index, two frames)", rotation),
    ],
)
assert same_frame and not cross_frame

# ---------------------------------------------------------------------------
# 3. Basis-aware steps act only on their own frame
# ---------------------------------------------------------------------------

j = ctx.alloc_index()
same_dot = td.canonicalize(
    tb.simplify_basis_dot(
        wcs.covariant_vector(i) @ wcs.covariant_vector(j), wcs
    )
)
overlap = tb.simplify_basis_dot(
    wcs.covariant_vector(i) @ primed.covariant_vector(j), wcs
)

show(
    "3. Basis-aware steps act only on their own frame",
    [
        ("same-frame dot (WCS, WCS)", same_dot),
        ("cross-frame overlap (WCS, primed)", overlap),
    ],
)
# The same-frame dot collapses to a Kronecker delta; the cross-frame overlap
# does not (it is the change-of-basis matrix, not δ).
assert "delta" in same_dot.latex()
assert "cdot" in overlap.latex()

# ---------------------------------------------------------------------------
# 4. Custom naming & display label
# ---------------------------------------------------------------------------

# A user-defined orthonormal frame with its own coordinate letters and
# standalone vector symbols — exactly how the built-in systems are configured.
u = tender.tensor("u", rank=1, ctx=ctx)
v = tender.tensor("v", rank=1, ctx=ctx)
w = tender.tensor("w", rank=1, ctx=ctx)
custom = tb.make_orthonormal_basis(
    [u, v, w],
    tender.space_3d,
    value_names=["\\xi", "\\eta", "\\zeta"],
    vector_symbols=["U", "V", "W"],
)
custom_expansion = td.unroll_sums(
    td.canonicalize(tb.expand_in_basis(a, custom, tb.Variance.Covariant))
)

# The primed frame's label marks every index, telling the two frames apart even
# though both use the generic symbol e.
k = ctx.alloc_index()
two_frames = wcs.covariant_vector(k) * primed.covariant_vector(k)

show(
    "4. Custom naming and display label",
    [
        ("custom frame expansion", custom_expansion),
        ("two frames in one term", td.canonicalize(two_frames)),
    ],
)

# ---------------------------------------------------------------------------
# Write a standalone LaTeX summary
# ---------------------------------------------------------------------------

sections = []
for title, rows in report:
    items = "\n".join(
        rf"  \item {label} \quad $\displaystyle {text}$"
        if any(c in text for c in "\\^_{}")
        else rf"  \item {label} \quad \texttt{{{text}}}"
        for label, text in rows
    )
    # LaTeX section titles drop the leading "N. ".
    heading = title.split(". ", 1)[-1]
    sections.append(
        rf"\subsection*{{{heading}}}" + "\n"
        r"\begin{itemize}" + "\n" + items + "\n" + r"\end{itemize}"
    )

doc = (
    r"\documentclass{article}"
    "\n"
    r"\usepackage[utf8]{inputenc}"
    "\n"
    r"\usepackage{amsmath,amssymb}"
    "\n"
    r"\begin{document}"
    "\n\n"
    r"\section*{Basis-aware indices and coordinate-system notation}"
    "\n\n"
    "Generated by \\texttt{basis\\_aware\\_coordinates.py} (vibe 000067).\n\n"
    + "\n\n".join(sections)
    + "\n\n"
    r"\end{document}"
    "\n"
)

out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "basis_aware_coordinates.tex"
out_path.write_text(doc)

print(f"\nWritten : {out_path}")
print(f"Compile : pdflatex -output-directory out {out_path}")
