"""Strain compatibility, the operator way (vibe 000077–000078, gap D).

The strain incompatibility of a symmetric strain field ε is

    inc ε = ∇×(∇×ε)ᵀ ,

and compatibility is inc ε = 0.  The point of this example is *how* tender
derives the closed form of inc ε "as performed" — expand ∇ only, keep ε
abstract, reduce the cross, reassemble the operators — never touching a single
ε_xy component until the very end, where the result is verified:

  1. With the first-class chart-free ∇ operator, inc ε is written coordinate-free
     — `∇ × (∇ × ε)ᵀ` — with **ε abstract**.
  2. `expand_nabla` lowers each ∇ to the *free-index* frame form e_i ∂_i, so
     inc ε becomes the interior  Σ_{i,j} e_i × (e_j × ∂_i∂_j ε)ᵀ  — still abstract
     in ε (only second derivatives ∂_i∂_j ε appear, never ε_xy components).
  3. The a×B×c cross-removal identity is *derived in-codebase* (not asserted) and
     applied to that interior, folding the nested cross into a cross-free sum of
     dyad / divergence / trace terms.
  4. `reassemble_nabla` reads each frame-indexed ∂'s role and folds it back into
     the invariant operators, giving the closed identity

         inc ε = −∇∇θ + Δθ·I − (∇∇··ε)·I − Δε + 2(∇∇·ε)ˢ ,   θ = tr ε.

  5. That closed identity is *verified* against inc ε component-by-component — in
     a Cartesian frame and, the curvilinear endpoint, in a cylindrical frame.
     Both sides are coordinate-free tensors, so the identity holds in every
     frame; tender confirms it in both.

Writes a standalone LaTeX summary to ``out/``.

Run:  python examples/strain_compatibility.py
"""

import pathlib

import tender as t
import tender.basis as tb
import tender.derivation as td

report: list[tuple[str, list[tuple[str, str]]]] = []


def show(title, rows):
    """Print a section and accumulate its rows for the LaTeX summary."""
    print(f"\n{title}")
    print("-" * len(title))
    latex_rows = []
    for label, value in rows:
        text = value.latex() if hasattr(value, "latex") else str(value)
        print(f"  {label:32s} {text}")
        latex_rows.append((label, text))
    report.append((title, latex_rows))


def cross_removal_identity(ctx):
    """Derive, in-codebase, the strain-interior identity

        id_inc :  a×(c×E)ᵀ = <cross-free δ / dyad RHS>     (E symmetric)

    by proving the a×B×c cross-removal identity from first principles (expand in
    the basis, apply c⊗a = a×I×c + (a·c)I, reduce the ε-ε contraction, reassemble)
    and composing it with the transpose-cross helper a×(c×E)ᵀ = −a×E×c.  Both
    pieces are proven by construction, not hand-asserted (vibe 000078 Q3)."""
    basis = tb.wcs(ctx)
    co = tb.Variance.Covariant
    a = t.tensor("a", 1, ctx=ctx)
    c = t.tensor("c", 1, ctx=ctx)
    B = t.tensor("B", 2, ctx=ctx)
    E = t.field("E", 2, ctx=ctx, symmetric=True)
    I = t.identity(ctx)

    def derive(initial, steps):
        d = td.Derivation(initial)
        for s in steps:
            d.step(s)
        return d.current

    axIxb = derive(
        a % I % c,
        (
            lambda x: tb.expand_in_basis(x, basis, co),
            lambda x: tb.simplify_basis_cross(x, basis),
            td.contract_eps_pair,
            td.contract_delta,
            lambda x: tb.reassemble(x, basis),
        ),
    )
    id_alt = td.Identity(
        "axIxb_alt", td.fold_equal_addends(axIxb + a @ c * I), a % I % c + a @ c * I
    )
    axBxc = derive(
        a % B % c,
        (
            lambda x: tb.expand_in_basis(x, basis, co),
            td.apply_identity(id_alt),
            lambda x: tb.expand_in_basis(x, basis, co),
            lambda x: tb.simplify_basis_cross(x, basis),
            lambda x: tb.simplify_basis_dot(x, basis),
            td.contract_delta,
            td.contract_eps_pair,
            td.contract_delta,
            td.contract_eps_pair,
            td.contract_delta,
            lambda x: tb.reassemble(x, basis),
        ),
    )
    id_axBxc = td.Identity("axBxc", a % B % c, axBxc)
    id_inc = td.Identity(
        "inc",
        a % (c % E).transpose(),
        td.canonicalize(-td.apply_identity(id_axBxc)(a % E % c)),
    )
    return id_axBxc, id_inc


def is_zero(chart, e):
    """True if the scalar field `e` reduces to 0 on the chart."""
    return td.simplify_scalars(td.canonicalize(chart.expand(e))).latex() == "0"


def closed_identity_matrix(chart, eps):
    """The RHS −∇∇θ + Δθ·I − (∇∇··ε)·I − Δε + 2(∇∇·ε)ˢ as a 3×3 component matrix
    on the chart's physical frame, assembled from the invariant operators.  The
    symmetric part 2(∇∇·ε)ˢ = ∇∇·ε + (∇∇·ε)ᵀ is taken at the matrix level."""
    theta = t.tr(eps)
    gg = chart.components(chart.grad(chart.grad(theta)))  # ∇∇θ
    de = chart.components(chart.div(chart.grad(eps)))  # Δε = ∇·∇ε
    gd = chart.components(chart.grad(chart.div(eps)))  # ∇∇·ε = ∇(∇·ε)
    lap = chart.laplacian(theta)  # Δθ
    dd = chart.div(chart.div(eps))  # ∇∇··ε
    out = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            r = (gg[i][j] * (-1)) + (de[i][j] * (-1)) + gd[i][j] + gd[j][i]
            if i == j:
                r = r + lap - dd
            out[i][j] = r
    return out


def verify(chart, eps, label):
    inc = chart.components(chart.rot(chart.rot(eps).transpose()))  # inc ε
    rhs = closed_identity_matrix(chart, eps)
    ok = all(
        is_zero(chart, chart.expand(inc[i][j]) - chart.expand(rhs[i][j]))
        for i in range(3)
        for j in range(3)
    )
    status = "✓ all 9 components equal" if ok else "✗ MISMATCH"
    print(f"  {label:32s} {status}")
    assert ok, f"closed identity failed in {label}"
    return ok


def main():
    ws = t.Workspace()
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    nabla = t.nabla(ctx=ws.ctx)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])

    # ---- 1. chart-free, ε abstract ------------------------------------------
    inc = nabla % (nabla % eps).transpose()
    show(
        "1. Coordinate-free  (ε abstract)",
        [("inc ε", inc), ("ε components?", "no" if "varepsilon_{" not in inc.latex() else "yes")],
    )

    # ---- 2. expand ∇ only, keep ε abstract ----------------------------------
    interior = cart.expand_nabla(inc)
    show(
        "2. expand_nabla → free-index interior",
        [
            ("Σ e_i × (e_j × ∂_i∂_j ε)ᵀ", interior),
            ("ε still abstract?", "yes" if "varepsilon_{" not in interior.latex() else "no"),
            ("second derivatives ∂∂?", "yes" if "partial" in interior.latex() else "no"),
        ],
    )

    # ---- 3. derive the a×B×c cross-removal identity, in-codebase -------------
    id_axBxc, id_inc = cross_removal_identity(ws.ctx)
    show(
        "3. a×B×c cross removal (derived, not asserted)",
        [("a×B×c", id_axBxc.rhs)],
    )

    # ---- 4. Phase-1 reduction + Phase-2 reassembly, natively ----------------
    phase1 = td.canonicalize(td.apply_identity(id_inc)(interior))
    reass = cart.reassemble_nabla(phase1)
    show(
        "4. reduce (cross-free) then reassemble into ∇ operators",
        [
            ("Phase-1  (cross-free?)", "yes" if "times" not in phase1.latex() else "no"),
            ("inc ε reassembled", reass),
        ],
    )

    # Confirm the reassembled operators are the closed compatibility identity.
    # (reassemble_nabla dimensions its identity to the chart's space for tr(I)=n,
    # but dimension-awareness is identity-neutral — vibe 000081 — so a plain I
    # here compares equal.)
    theta = t.tr(eps)
    I = t.identity(ws.ctx)
    closed = (
        -(nabla @ (nabla @ eps)) * I  # −(∇∇··ε) I
        + t.laplacian(theta) * I  # +Δθ I  (invariant Laplacian, vibe 000083)
        - t.laplacian(eps)  # −Δε
        - (nabla * (nabla * theta))  # −∇∇θ  (scalar Hessian: symmetric, no ᵀ)
        + (nabla * (nabla @ eps))  # +∇∇·ε
        + (nabla * (nabla @ eps)).transpose()  # +(∇∇·ε)ᵀ
    )
    assert td.algebraic_eq(reass, closed), "reassembly ≠ closed identity"

    # ---- 5. verify the closed identity component-by-component ---------------
    print("\n5. Verification  (component-by-component, via the chart operators)")
    verify(cart, eps, "Cartesian")
    ws2 = t.Workspace()
    eps2 = ws2.field(r"\varepsilon", 2, symmetric=True)
    r, th, zc = ws2.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws2.chart(ws2.wcs(), [r, th, zc], [r * t.cos(th), r * t.sin(th), zc])
    verify(cyl, eps2, "Cylindrical")
    report.append(
        (
            "5. Verification (component-by-component)",
            [
                ("Cartesian frame", r"\checkmark\ \text{all 9 components equal}"),
                ("Cylindrical frame", r"\checkmark\ \text{all 9 components equal}"),
            ],
        )
    )

    print("\nCompatibility inc ε = 0 is thus the vanishing of the closed-form")
    print("right-hand side — the same tensor equation in any frame.")

    write_latex()


def write_latex():
    sections = []
    for title, rows in report:
        items = "\n".join(
            rf"  \item {label} \quad $\displaystyle {text}$"
            if any(ch in text for ch in "\\^_{}")
            else rf"  \item {label} \quad \texttt{{{text}}}"
            for label, text in rows
        )
        heading = title.split(". ", 1)[-1]
        sections.append(
            rf"\subsection*{{{heading}}}" + "\n"
            r"\begin{itemize}" + "\n" + items + "\n" + r"\end{itemize}"
        )
    doc = (
        r"\documentclass{article}" "\n"
        r"\usepackage[utf8]{inputenc}" "\n"
        r"\usepackage{amsmath,amssymb}" "\n"
        r"\allowdisplaybreaks" "\n"
        r"\begin{document}" "\n\n"
        r"\section*{Strain compatibility $\operatorname{inc}\varepsilon"
        r" = \nabla\times(\nabla\times\varepsilon)^{\mathsf{T}}$}" "\n\n"
        "Generated by \\texttt{strain\\_compatibility.py} (vibe 000078): "
        "inc $\\varepsilon$ derived as performed --- expand $\\nabla$, reduce the "
        "cross via the in-codebase a$\\times$B$\\times$c identity, reassemble into "
        "$\\nabla$ operators --- then verified.\n\n"
        + "\n\n".join(sections)
        + "\n\n" r"\end{document}" "\n"
    )
    out_dir = pathlib.Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "strain_compatibility.tex").write_text(doc)


if __name__ == "__main__":
    main()
