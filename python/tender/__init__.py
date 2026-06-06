"""tender — tensor algebra library for computational mechanics."""

from ._tender import (
    # Types
    Rational,
    Expr,
    RationalConst,
    NamedConst,
    SymbolicVar,
    Parameter,
    Sum,
    Scale,
    TensorProduct,
    IdentityTensor,
    LeviCivitaTensor,
    Trace,
    Contract,
    DoubleContract,
    DoubleContractReversed,
    CrossProduct,
    NamedTensor,
    Product,
    FunctionApply,
    Pow,
    MaterialDeriv,
    PatternVar,
    Gradient,
    Divergence,
    Rotor,
    Integral,
    Domain,
    SurfaceDomain,
    VolumeDomain,
    CoordSystem,
    State,
    DerivationStep,
    Derivation,
    Identity,
    Nabla,
    # Expression factories
    tensor,
    scalar,
    parameter,
    named,
    rational,
    make_pattern_var,
    # Algebraic operations
    tp,
    dot,
    ddot,
    ddot2,
    cross,
    trace,
    # Scalar functions
    exp,
    log,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    atan2,
    sinh,
    cosh,
    tanh,
    sqrt,
    pow,
    # Differentiation
    deriv,
    dt,
    ddt,
    # Coordinate systems
    grad,
    make_direct_basis_cs,
    # Symbolic differential operators
    gradient,
    divergence,
    rot,
    # Integral / domain
    make_surface_domain,
    make_volume_domain,
    integral,
    # Derivation rendering
    show,
    show_final,
    # Step factories
    simplify_identity_step,
    expand_step,
    expand_poly_step,
    substitute_step,
    diff_step,
    apply_integration_by_parts_step,
    apply_divergence_theorem_step,
    localize_step,
    apply_identity,
    # Singleton getters (private)
    _identity_singleton,
    _levi_civita_singleton,
    _time_param_singleton,
    _wcs_singleton,
    _cylindrical_cs_singleton,
    _spherical_cs_singleton,
)

# Module-level singletons
I = _identity_singleton()
eps = _levi_civita_singleton()
t = _time_param_singleton()
wcs = _wcs_singleton()
cylindrical_cs = _cylindrical_cs_singleton()
spherical_cs = _spherical_cs_singleton()
nabla = Nabla()

def _label_to_math(label):
    """Render a step label as a LaTeX math-mode fragment.

    Plain identifiers (not part of a \\command) are wrapped in \\mathrm{} so
    they appear upright.  LaTeX commands (\\partial, etc.) are passed through
    unchanged.  Returns a string suitable for use inside \\[...\\].
    """
    import re

    if "\\" not in label:
        # No LaTeX commands — use text mode directly.
        return r"\text{[" + label + r":] }"

    # Match \command sequences first (pass through), then bare alpha runs
    # (wrap in \mathrm{}).  The alternation consumes \partial as one token,
    # preventing "artial" from being re-matched as a separate identifier.
    def _repl(m):
        s = m.group(0)
        return s if s.startswith("\\") else r"\mathrm{" + s + "}"

    math_label = re.sub(r"\\[A-Za-z]+|[A-Za-z]+", _repl, label)
    return r"\text{[}" + math_label + r"\text{:] }"


def to_latex_document(history, title=None):
    """Generate a standalone compilable LaTeX document from a derivation history.

    Each state in history is rendered as a display equation prefixed with
    the step label.  Requires amsmath and amssymb.

    Example::

        history = Derivation([...]).apply(State(expr))
        tex = to_latex_document(history, title="PVW derivation")
        with open("derivation.tex", "w") as f:
            f.write(tex)
    """
    lines = [
        r"\documentclass{article}",
        r"\usepackage{amsmath,amssymb}",
        r"\begin{document}",
    ]
    if title:
        lines.append(r"\section*{" + title + "}")
    for state in history:
        label = state.label or "initial"
        lines.append(
            r"\[" + _label_to_math(label) + r"\quad " + state.expr.latex() + r"\]"
        )
    lines.append(r"\end{document}")
    return "\n".join(lines)


__all__ = [
    # Types
    "Rational", "Expr", "RationalConst", "NamedConst", "SymbolicVar",
    "Parameter", "Sum", "Scale", "TensorProduct", "IdentityTensor",
    "LeviCivitaTensor", "Trace", "Contract", "DoubleContract",
    "DoubleContractReversed", "CrossProduct", "NamedTensor", "Product",
    "FunctionApply", "Pow", "MaterialDeriv", "PatternVar",
    "Gradient", "Divergence", "Rotor", "Integral",
    "Domain", "SurfaceDomain", "VolumeDomain", "CoordSystem",
    "State", "DerivationStep", "Derivation", "Identity", "Nabla",
    # Singletons
    "I", "eps", "t", "wcs", "cylindrical_cs", "spherical_cs", "nabla",
    # Expression factories
    "tensor", "scalar", "parameter", "named", "rational", "make_pattern_var",
    # Algebraic operations
    "tp", "dot", "ddot", "ddot2", "cross", "trace",
    # Scalar functions
    "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "sqrt", "pow",
    # Differentiation
    "deriv", "dt", "ddt",
    # Coordinate systems
    "grad", "make_direct_basis_cs",
    # Symbolic differential operators
    "gradient", "divergence", "rot",
    # Integral / domain
    "make_surface_domain", "make_volume_domain", "integral",
    # Derivation rendering
    "show", "show_final",
    # LaTeX document export
    "to_latex_document",
    # Step factories
    "simplify_identity_step", "expand_step", "expand_poly_step",
    "substitute_step", "diff_step",
    "apply_integration_by_parts_step", "apply_divergence_theorem_step",
    "localize_step", "apply_identity",
]
