"""Navier–Lamé reduction, the operator way (vibe 000080, Increment 8).

The isotropic Hooke stress for a displacement field u is

    T = λ (∇·u) I + μ (∇u + (∇u)ᵀ) ,

and the equilibrium divergence ∇·T is the elastic operator acting on u.  The
point of this example is *how* tender reduces ∇·T "as performed" — expand ∇
only, keep u abstract, apply the Leibniz ∂'s, fold e_i·I, reassemble the
operators — never touching a single u_x component until the very end, where the
Navier–Lamé endpoint is verified:

  1. With the first-class chart-free ∇ operator, ∇·T is written coordinate-free
     with **u abstract** (only ∇u, ∇·u appear — never u_x components).
  2. `expand_nabla` lowers each ∇ to the *free-index* frame form e_i ∂_i, so
     ∇·T becomes an interior Σ ∂'s acting on the abstract T — still abstract
     in u.  The inner ∇·u hidden inside (∇·u)I is resolved by Leibniz before the
     outer ∂ (the apply_operators nested-operator fix).
  3. `contract_identity` folds every e_i·I → e_i, and `reassemble_nabla` reads
     each frame-indexed ∂'s role and folds it back into the invariant operators,
     carrying the Lamé coefficients λ, μ through, giving

         ∇·T = λ ∇(∇·u) + μ ∇(∇·u) + μ ∇·∇u .

  4. `collect_terms` + `factor_common` reverse the distribution, pulling the
     shared ∇·u out of the λ- and μ-gradients, for the clean Navier–Lamé form

         ∇·T = μ ∇·∇u + ∇((λ+μ) ∇·u)   ( = μ Δu + (λ+μ) ∇(∇·u) ).

  5. That endpoint is *verified* against ∇·T component-by-component — in a
     Cartesian frame and, the curvilinear endpoint, in a cylindrical frame.
     Both sides are coordinate-free vectors, so the identity holds in every
     frame; tender confirms it in both.

Writes a standalone LaTeX summary to ``out/``.

Run:  python examples/navier_lame.py
"""

import pathlib

import tender as t
import tender.derivation as td

report: list[tuple[str, list[tuple[str, str]]]] = []


def show(title, rows):
    """Print a section and accumulate its rows for the LaTeX summary."""
    print(f"\n{title}")
    print("-" * len(title))
    latex_rows = []
    for label, value in rows:
        text = value.latex() if hasattr(value, "latex") else str(value)
        print(f"  {label:34s} {text}")
        latex_rows.append((label, text))
    report.append((title, latex_rows))


def is_zero(chart, e):
    """True if the scalar field `e` reduces to 0 on the chart."""
    return td.simplify_scalars(td.canonicalize(chart.expand(e))).latex() == "0"


def hooke_stress(chart, u, lam, mu, I):
    """The isotropic Hooke stress T = λ(∇·u)I + μ(∇u + (∇u)ᵀ) on the chart."""
    gradu = chart.grad(u)
    return lam * chart.div(u) * I + mu * (gradu + gradu.transpose())


def navier_lame_vector(chart, u, lam, mu):
    """The Navier–Lamé operator μ∇·∇u + (λ+μ)∇(∇·u) on the chart."""
    return mu * chart.div(chart.grad(u)) + (lam + mu) * chart.grad(chart.div(u))


def verify(chart, u, lam, mu, I, label):
    """∇·T equals the Navier–Lamé vector, component by component, on the chart."""
    lhs = chart.components(chart.div(hooke_stress(chart, u, lam, mu, I)))
    rhs = chart.components(navier_lame_vector(chart, u, lam, mu))
    ok = all(
        is_zero(chart, chart.expand(lhs[i]) - chart.expand(rhs[i])) for i in range(3)
    )
    status = "✓ all 3 components equal" if ok else "✗ MISMATCH"
    print(f"  {label:34s} {status}")
    assert ok, f"Navier–Lamé endpoint failed in {label}"
    return ok


def verify_evaluate(chart, nabla, u, lam, mu, I, label):
    """The coordinate-free ∇·T evaluated *directly* in the chart (vibe 000084).

    No hand-rewrite via chart.grad/div/rot — write T with the invariant ∇ and
    call chart.evaluate, which lowers each ∇-combination to the curvilinear
    operators.  It must match the same Navier–Lamé endpoint.
    """
    T = lam * (nabla @ u) * I + mu * (nabla * u + (nabla * u).transpose())
    lhs = chart.components(chart.evaluate(nabla @ T))  # ∇·T, evaluated in-chart
    rhs = chart.components(navier_lame_vector(chart, u, lam, mu))
    ok = all(
        is_zero(chart, chart.expand(lhs[i]) - chart.expand(rhs[i])) for i in range(3)
    )
    status = "✓ all 3 components equal" if ok else "✗ MISMATCH"
    print(f"  {label:34s} {status}")
    assert ok, f"chart.evaluate(∇·T) failed in {label}"
    return ok


def main():
    ws = t.Workspace()
    u = ws.field("u", 1)
    nabla = t.nabla(ctx=ws.ctx)
    I = t.identity(ws.ctx)
    lam = t.tensor(r"\lambda", 0, ctx=ws.ctx)
    mu = t.tensor(r"\mu", 0, ctx=ws.ctx)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])

    # ---- 1. chart-free, u abstract ------------------------------------------
    T = lam * (nabla @ u) * I + mu * (nabla * u + (nabla * u).transpose())
    divT = nabla @ T
    show(
        "1. Coordinate-free  (u abstract)",
        [
            ("T  (Hooke stress)", T),
            ("∇·T", divT),
            ("u components?", "no" if "u_{" not in divT.latex() else "yes"),
        ],
    )

    # ---- 2. expand ∇ only, keep u abstract ----------------------------------
    interior = td.contract_identity(td.canonicalize(cart.expand_nabla(divT)))
    show(
        "2. expand_nabla → free-index interior (e_i·I folded)",
        [
            ("interior", interior),
            ("u still abstract?", "yes" if "u_{" not in interior.latex() else "no"),
            ("first derivatives ∂?", "yes" if "partial" in interior.latex() else "no"),
        ],
    )

    # ---- 3. reassemble into ∇ operators -------------------------------------
    reass = cart.reassemble_nabla(td.canonicalize(interior))
    expanded_form = (
        lam * (nabla * (nabla @ u))  # λ∇(∇·u)
        + mu * (nabla * (nabla @ u))  # μ∇(∇·u)  (from ∇·((∇u)ᵀ))
        + mu * (nabla @ (nabla * u))  # μ∇·∇u    (from ∇·(∇u))
    )
    assert td.algebraic_eq(reass, expanded_form), "reassembly ≠ operator form"
    show(
        "3. reassemble ∂'s back into ∇ operators (Lamé coeffs carried)",
        [
            ("∇·T reassembled", td.collect_terms(reass)),
            ("λ, μ survived?", "yes" if r"\lambda" in reass.latex() else "no"),
        ],
    )

    # ---- 4. reverse the distribution: the Navier–Lamé endpoint --------------
    # factor_common pulls the shared ∇·u out of the λ- and μ-gradients and then
    # hoists the constant (λ+μ) fully outside the gradient (∇(cX)=c∇X, ∇c=0).
    # That last step is an operator-linearity rewrite expand_products can't
    # model, so the endpoint's correctness is proven componentwise in §5, not by
    # a distribution round-trip (see the vibe-000080 "don't re-canonicalize"
    # caveat).
    nl = td.factor_common(td.collect_terms(reass))
    show(
        "4. factor_common → clean Navier–Lamé endpoint",
        [
            ("∇·T  =  μΔu + (λ+μ)∇(∇·u)", nl),
        ],
    )

    # ---- 4b. the textbook form T = λ tr(ε)I + 2με reduces identically -------
    # ε = sym(∇u) = (∇u + (∇u)ᵀ)/2; the scalar /2 rides out through the whole
    # reduction and the endpoint is the same clean Navier–Lamé vector.
    T_std = lam * (nabla @ u) * I + t.scalar(2, ctx=ws.ctx) * mu * td.sym(nabla * u)
    reass_std = cart.reassemble_nabla(
        td.canonicalize(td.contract_identity(td.canonicalize(cart.expand_nabla(nabla @ T_std))))
    )
    nl_std = td.factor_common(td.collect_terms(reass_std))
    assert nl_std.latex() == nl.latex(), "textbook 2με form ≠ explicit-form endpoint"
    show(
        "4b. textbook form  T = λ tr(ε)I + 2με,  ε = sym(∇u)",
        [
            ("T  (textbook)", T_std),
            ("∇·T  (same endpoint)", nl_std),
        ],
    )

    # ---- 5. verify the endpoint component-by-component ----------------------
    print("\n5. Verification  (component-by-component, via the chart operators)")
    verify(cart, u, lam, mu, I, "Cartesian")
    ws2 = t.Workspace()
    u2 = ws2.field("u", 1)
    lam2 = t.tensor(r"\lambda", 0, ctx=ws2.ctx)
    mu2 = t.tensor(r"\mu", 0, ctx=ws2.ctx)
    I2 = t.identity(ws2.ctx)
    r, th, zc = ws2.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws2.chart(ws2.wcs(), [r, th, zc], [r * t.cos(th), r * t.sin(th), zc])
    verify(cyl, u2, lam2, mu2, I2, "Cylindrical")
    report.append(
        (
            "5. Verification (component-by-component)",
            [
                ("Cartesian frame", r"\checkmark\ \text{all 3 components equal}"),
                ("Cylindrical frame", r"\checkmark\ \text{all 3 components equal}"),
            ],
        )
    )

    # ---- 6. the clean path: evaluate the coordinate-free ∇·T in the chart ----
    # No re-typing with chart.grad/div/rot — the invariant T (u abstract, ∇
    # coordinate-free) is handed straight to chart.evaluate, which lowers each
    # ∇-combination to the curvilinear-correct operators (vibe 000084).
    print("\n6. chart.evaluate(∇·T)  (coordinate-free expression, no hand-rewrite)")
    nabla2 = t.nabla(ctx=ws2.ctx)
    verify_evaluate(cart, nabla, u, lam, mu, I, "Cartesian")
    verify_evaluate(cyl, nabla2, u2, lam2, mu2, I2, "Cylindrical")
    report.append(
        (
            "6. chart.evaluate($\\nabla\\cdot T$) --- coordinate-free, no rewrite",
            [
                ("Cartesian frame", r"\checkmark\ \text{all 3 components equal}"),
                ("Cylindrical frame", r"\checkmark\ \text{all 3 components equal}"),
            ],
        )
    )

    print("\nThe equilibrium operator ∇·T is thus the Navier–Lamé vector")
    print("μΔu + (λ+μ)∇(∇·u) — the same tensor equation in any frame.")

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
        r"\section*{Navier--Lam\'e reduction"
        r" $\nabla\cdot T = \mu\,\Delta u + \nabla((\lambda+\mu)\nabla\cdot u)$}"
        "\n\n"
        "Generated by \\texttt{navier\\_lame.py} (vibe 000080, Increment 8): "
        "$\\nabla\\cdot T$ of the isotropic Hooke stress derived as performed --- "
        "expand $\\nabla$, apply the Leibniz $\\partial$'s, fold $e_i\\cdot I$, "
        "reassemble into $\\nabla$ operators, reverse the distribution --- then "
        "verified.\n\n"
        + "\n\n".join(sections)
        + "\n\n" r"\end{document}" "\n"
    )
    out_dir = pathlib.Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "navier_lame.tex").write_text(doc)


if __name__ == "__main__":
    main()
