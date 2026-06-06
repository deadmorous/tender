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
    # Step factories
    "simplify_identity_step", "expand_step", "expand_poly_step",
    "substitute_step", "diff_step",
    "apply_integration_by_parts_step", "apply_divergence_theorem_step",
    "localize_step", "apply_identity",
]
