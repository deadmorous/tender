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
    collect_step,
    apply_identity,
    find_matches,
    apply_identity_auto,
    matches_at_root,
    _find_and_rewrite_all,
    _capture_step,
    declare_symmetric,
    declare_skew_symmetric,
    simplify_basis_dot_step,
    collect_zero_terms_step,
    reassemble_from_components_step,
    collect_repeated_sum_step,
    reassemble_vector_step,
    reassemble_dot_step,
    # IndexedSum node
    IndexedSum,
    make_indexed_sum,
    # SymBasisVec node
    SymBasisVec,
    make_sym_basis_vec,
    # AbstractComp and AbstractIndexedSum nodes
    AbstractComp,
    AbstractIndexedSum,
    make_abstract_comp,
    make_abstract_indexed_sum,
    alloc_index_id,
    # KroneckerDelta and LeviCivitaSymbol nodes
    KroneckerDelta,
    LeviCivitaSymbol,
    make_kronecker_delta,
    make_levi_civita_symbol,
    # contract_kronecker_step, substitute_index, eps-pair steps
    contract_kronecker_step,
    substitute_index,
    replace_first_lct_step,
    contract_eps_pair_step,
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

_INDEX_LETTERS = ["i", "j", "k", "l", "m", "n"]


def _abstract_index_letters(rank, first_letter=None):
    first = first_letter or "i"
    result = [first]
    for ltr in _INDEX_LETTERS:
        if len(result) >= rank:
            break
        if ltr not in result:
            result.append(ltr)
    for ltr in "abcde":
        if len(result) >= rank:
            break
        if ltr not in result:
            result.append(ltr)
    return result[:rank]


def _abstract_comp_sym(base_sym, cov_list, letters):
    """Build component symbol with grouped indices: "A^ij_k", "A^i_j", etc."""
    result = base_sym
    current_sep = None
    for cov, ltr in zip(cov_list, letters):
        sep = "^" if cov else "_"
        if sep == current_sep:
            result += ltr
        else:
            result += sep + ltr
            current_sep = sep
    return result


def expand_in_basis_step(tensor_expr, cs, covariant=True, abstract=False):
    """Create a DerivationStep that replaces tensor_expr with its basis expansion.

    Concrete mode (default, abstract=False):
      rank-1 covariant:    a  →  a^1 e_1 + a^2 e_2 + a^3 e_3
      rank-1 contravariant: a →  a_1 e^1 + a_2 e^2 + a_3 e^3
      rank-2 covariant:    A  →  Σ_{i,j} A^{ij} e_i ⊗ e_j

    Abstract mode (abstract=True):
      rank-1 covariant:   a → AbstractComp("a", [(id, True)]) ⊗ SBV(id, basis)
      rank-1 contravariant: a → AbstractComp("a", [(id, False)]) ⊗ SBV(id, cobasis)
      rank-2 all-up:    A → AbstractComp("A", [(i, True), (j, True)]) ⊗ SBV_i ⊗ SBV_j
      rank-2 mixed:     A → AbstractComp("A", [(i, True), (j, False)]) ⊗ SBV_i ⊗ SBV^j

    In abstract mode, simplify_basis_dot_step collapses rank-1 dot products
    directly to an AbstractIndexedSum without 9-term expansion.  Index IDs are
    allocated automatically from the global context; latex rendering assigns
    letter names (i, j, k, …) in first-appearance order via enrich().

    covariant: bool (applied to all indices) or list[bool] (one per index).
    """
    sym = tensor_expr.symbol
    r = tensor_expr.rank
    dim = cs.dim

    if abstract:
        if r < 1:
            raise ValueError(
                f"expand_in_basis_step: abstract mode requires rank ≥ 1, got {r}"
            )
        cov_list = ([covariant] * r) if isinstance(covariant, bool) else list(covariant)
        if len(cov_list) != r:
            raise ValueError(
                f"expand_in_basis_step: covariant list length {len(cov_list)} "
                f"does not match tensor rank {r}"
            )
        # Allocate fresh integer index IDs for each index
        index_ids = [alloc_index_id() for _ in range(r)]
        # Build AbstractComp: pairs are (index_id, is_upper) where upper = covariant
        indices = [(idx, cov) for idx, cov in zip(index_ids, cov_list)]
        comp = make_abstract_comp(sym, indices)
        expr = comp
        for idx, cov in zip(index_ids, cov_list):
            expr = expr * make_sym_basis_vec(cs, idx, not cov)
        return substitute_step(tensor_expr, expr)

    if r == 1:
        if covariant:
            comps = [tensor(f"{sym}^{i + 1}") for i in range(dim)]
            vecs = [cs.basis(i) for i in range(dim)]
        else:
            comps = [tensor(f"{sym}_{i + 1}") for i in range(dim)]
            vecs = [cs.cobasis(i) for i in range(dim)]
        terms = [c * v for c, v in zip(comps, vecs)]
    elif r == 2:
        if covariant:
            def _fmt(i, j):
                return f"{sym}^{{{i + 1}{j + 1}}}"
        else:
            def _fmt(i, j):
                return f"{sym}_{{{i + 1}{j + 1}}}"
        terms = [
            tensor(_fmt(i, j)) * (cs.basis(i) * cs.basis(j))
            for i in range(dim)
            for j in range(dim)
        ]
    else:
        raise ValueError(
            f"expand_in_basis_step: unsupported rank {r} "
            f"(only rank 1 and 2 are supported)"
        )

    expansion = terms[0]
    for t in terms[1:]:
        expansion = expansion + t
    return substitute_step(tensor_expr, expansion)


class Theorem:
    """A mathematical statement that has been verified within the system.

    Unlike Identity (an asserted axiom), a Theorem requires a proof callable
    that is invoked at construction.  If the proof raises or returns False,
    construction fails.  A failed proof in tender.lib.theorems makes the
    module unimportable, surfacing regressions immediately.
    """

    def __init__(self, name, lhs, rhs, proof):
        result = proof(lhs, rhs)
        if result is False:
            raise ValueError(f"Theorem '{name}': proof failed")
        self.name = name
        self.lhs = lhs
        self.rhs = rhs

    @classmethod
    def by_components(cls, name, lhs, rhs, lhs_steps, rhs_steps=None):
        """Prove lhs == rhs by reducing both sides to component form.

        Applies lhs_steps to lhs and rhs_steps (or lhs_steps if omitted) to
        rhs, then compares the terminal expressions via _normalize_component_form.
        """
        _rhs_steps = rhs_steps if rhs_steps is not None else lhs_steps

        def proof(l, r):
            prove_equal_by_components(l, r, lhs_steps, _rhs_steps)
            return True

        return cls(name, lhs, rhs, proof)

    @classmethod
    def by_derivation(cls, name, lhs, rhs, steps):
        """Prove lhs == rhs by applying steps to lhs and comparing to rhs.

        The final expression in the derivation is compared to rhs using
        python() string equality.
        """

        def proof(l, r):
            result = Derivation(steps).apply(State(l))[-1].expr
            if result.python() != r.python():
                raise ValueError(
                    f"Theorem '{name}': derivation produced\n"
                    f"  {result.python()}\nbut expected\n  {r.python()}"
                )
            return True

        return cls(name, lhs, rhs, proof)


def expand_identity_step(cs):
    """Replace IdentityTensor with e^{i} ⊗ e_{i} (orthonormal CS only)."""
    if not cs.is_orthonormal:
        raise NotImplementedError(
            "expand_identity_step: non-orthonormal CS not supported"
        )
    idx = alloc_index_id()
    expansion = make_sym_basis_vec(cs, idx, True) * make_sym_basis_vec(cs, idx, False)
    return substitute_step(I, expansion)


def expand_levi_civita_step(cs):
    """Replace LeviCivitaTensor with ε_{ijk} e^{i} ⊗ e^{j} ⊗ e^{k} (orthonormal CS only)."""
    if not cs.is_orthonormal:
        raise NotImplementedError(
            "expand_levi_civita_step: non-orthonormal CS not supported"
        )
    i_id = alloc_index_id()
    j_id = alloc_index_id()
    k_id = alloc_index_id()
    lcs = make_levi_civita_symbol([i_id, j_id, k_id], [False, False, False])
    sbv_i = make_sym_basis_vec(cs, i_id, True)
    sbv_j = make_sym_basis_vec(cs, j_id, True)
    sbv_k = make_sym_basis_vec(cs, k_id, True)
    expansion = lcs * sbv_i * sbv_j * sbv_k
    return substitute_step(eps, expansion)


def expand_levi_civita_first_step(cs):
    """Replace only the FIRST LeviCivitaTensor with ε_{ijk} e^{i} ⊗ e^{j} ⊗ e^{k}."""
    if not cs.is_orthonormal:
        raise NotImplementedError(
            "expand_levi_civita_first_step: non-orthonormal CS not supported"
        )
    i_id = alloc_index_id()
    j_id = alloc_index_id()
    k_id = alloc_index_id()
    lcs = make_levi_civita_symbol([i_id, j_id, k_id], [False, False, False])
    sbv_i = make_sym_basis_vec(cs, i_id, True)
    sbv_j = make_sym_basis_vec(cs, j_id, True)
    sbv_k = make_sym_basis_vec(cs, k_id, True)
    expansion = lcs * sbv_i * sbv_j * sbv_k
    return replace_first_lct_step(expansion)


def doc(entry, format="latex"):
    """Render an Identity (or future Theorem) as LaTeX, plain text, or Jupyter math.

    Parameters
    ----------
    entry : Identity
        The identity to document.
    format : str
        ``"latex"`` (default) — returns a compilable LaTeX snippet.
        ``"plain"`` — returns ASCII text.
        ``"jupyter"`` — displays in Jupyter via IPython.display.Math.
    """
    name = entry.name
    lhs_tex = entry.lhs.latex()
    rhs_tex = entry.rhs.latex()

    if format == "plain":
        return f"[{name}]  {entry.lhs.python()}  =  {entry.rhs.python()}"

    tex = f"\\textbf{{{name}:}}\n\\[\n  {lhs_tex} = {rhs_tex}\n\\]"

    if format == "jupyter":
        try:
            from IPython.display import Math, display
            display(Math(lhs_tex + " = " + rhs_tex))
            return
        except ImportError:
            pass  # fall through to latex

    return tex


def show_jupyter(history):
    """Display a derivation history as formatted LaTeX equations in a Jupyter cell.

    Each step is rendered as a displayed equation with its step label on the
    left, matching the layout produced by :func:`to_latex_document`.

    Parameters
    ----------
    history : list[State]
        The sequence of states returned by :meth:`Derivation.apply`.
    """
    from IPython.display import display, Latex

    blocks = []
    for state in history:
        label = state.label or "initial"
        blocks.append(
            r"\[" + _label_to_math(label) + r"\quad " + state.expr.latex() + r"\]"
        )
    display(Latex("\n".join(blocks)))


def search_apply(target, expr, rules=None, timeout=5.0):
    """Find and apply ``target``, preceded by any necessary preparation rewrites.

    Performs a breadth-first search over sub-expression rewrite steps drawn
    from ``rules``.  Returns a complete list of :class:`DerivationStep` objects
    that, when applied to ``expr`` in order, produce the final result with
    ``target`` applied.  The last step is always the application of ``target``.

    Parameters
    ----------
    target : Identity
        The identity to apply.
    expr : Expr
        Starting expression.
    rules : list[Identity] or None
        Rule library for the BFS.  ``None`` (default) uses all identities in
        ``tender.lib``.
    timeout : float
        Wall-clock time limit in seconds (fractions allowed, e.g. ``0.5``).

    Returns
    -------
    list[DerivationStep]
        Complete steps including the final application of ``target``.
        ``Derivation(steps).apply(State(expr))[-1].expr`` is the result.

    Raises
    ------
    TimeoutError
        No sequence found within ``timeout`` seconds.
    RuntimeError
        Search space exhausted (all reachable expressions visited).
    """
    import time
    from collections import deque

    if rules is None:
        from tender.lib import ALL
        rules = ALL

    def _try_target(e):
        """Return capture step applying target anywhere in e, or None."""
        matches = list(_find_and_rewrite_all(target, e))
        if matches:
            new_e, name = matches[0]
            return _capture_step(name, new_e)
        return None

    direct = _try_target(expr)
    if direct is not None:
        return [direct]

    deadline = time.monotonic() + timeout
    visited = {expr.latex()}
    queue = deque([(expr, [])])

    while queue:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"search_apply: no preparation sequence for '{target.name}' "
                f"found within {timeout:g}s")

        current, steps = queue.popleft()

        for rule in rules:
            for item in _find_and_rewrite_all(rule, current):
                new_expr, step_name = item
                key = new_expr.latex()
                if key in visited:
                    continue
                visited.add(key)
                step = _capture_step(step_name, new_expr)
                new_steps = steps + [step]

                final = _try_target(new_expr)
                if final is not None:
                    return new_steps + [final]

                queue.append((new_expr, new_steps))

    raise RuntimeError(
        f"search_apply: search space exhausted for '{target.name}' — "
        f"no sequence found within the visited rule applications")


def prove_equal_by_components(lhs_expr, rhs_expr, lhs_steps, rhs_steps):
    """Prove lhs_expr == rhs_expr by expanding both to component form.

    Applies ``lhs_steps`` to ``lhs_expr`` and ``rhs_steps`` to ``rhs_expr``,
    then compares the terminal expressions.  Comparison is order-insensitive for
    sums and commutative for scalar products (Product nodes).

    Parameters
    ----------
    lhs_expr, rhs_expr : Expr
        The two invariant expressions to compare.
    lhs_steps, rhs_steps : list[DerivationStep]
        Derivation steps that reduce each side to component form.

    Returns
    -------
    tuple[list[State], list[State]]
        ``(lhs_history, rhs_history)`` on success.

    Raises
    ------
    ValueError
        If the terminal component forms are not equal.
    """
    lhs_history = Derivation(lhs_steps).apply(State(lhs_expr))
    rhs_history = Derivation(rhs_steps).apply(State(rhs_expr))
    lhs_normal = _normalize_component_form(lhs_history[-1].expr)
    rhs_normal = _normalize_component_form(rhs_history[-1].expr)
    if lhs_normal != rhs_normal:
        raise ValueError(
            "Expressions do not reduce to the same component form:\n"
            f"  lhs: {lhs_history[-1].expr.python()}\n"
            f"  rhs: {rhs_history[-1].expr.python()}"
        )
    return lhs_history, rhs_history


def _collect_ac_nodes(expr, out):
    """DFS-collect all AbstractComp nodes in expr."""
    if isinstance(expr, AbstractComp):
        out.append(expr)
    elif isinstance(expr, TensorProduct):
        _collect_ac_nodes(expr.lhs, out)
        _collect_ac_nodes(expr.rhs, out)
    elif isinstance(expr, Product):
        _collect_ac_nodes(expr.lhs, out)
        _collect_ac_nodes(expr.rhs, out)
    elif isinstance(expr, Scale):
        _collect_ac_nodes(expr.expr, out)


def _collect_sbv_nodes(expr, out):
    """DFS-collect all SymBasisVec nodes in expr."""
    if isinstance(expr, SymBasisVec):
        out.append(expr)
    elif isinstance(expr, TensorProduct):
        _collect_sbv_nodes(expr.lhs, out)
        _collect_sbv_nodes(expr.rhs, out)
    elif isinstance(expr, Product):
        _collect_sbv_nodes(expr.lhs, out)
        _collect_sbv_nodes(expr.rhs, out)


def _collect_kd_nodes(expr, out):
    """DFS-collect all KroneckerDelta nodes in expr."""
    if isinstance(expr, KroneckerDelta):
        out.append(expr)
    elif isinstance(expr, TensorProduct):
        _collect_kd_nodes(expr.lhs, out)
        _collect_kd_nodes(expr.rhs, out)
    elif isinstance(expr, Product):
        _collect_kd_nodes(expr.lhs, out)
        _collect_kd_nodes(expr.rhs, out)
    elif isinstance(expr, Scale):
        _collect_kd_nodes(expr.expr, out)


def _count_index_ids_local(expr, counts):
    """Count index ID occurrences within a local expression (not crossing Sum boundaries)."""
    if isinstance(expr, AbstractComp):
        for idx, _ in expr.indices:
            counts[idx] = counts.get(idx, 0) + 1
    elif isinstance(expr, SymBasisVec):
        counts[expr.index_id] = counts.get(expr.index_id, 0) + 1
    elif isinstance(expr, KroneckerDelta):
        counts[expr.lower_id] = counts.get(expr.lower_id, 0) + 1
        counts[expr.upper_id] = counts.get(expr.upper_id, 0) + 1
    elif isinstance(expr, TensorProduct):
        _count_index_ids_local(expr.lhs, counts)
        _count_index_ids_local(expr.rhs, counts)
    elif isinstance(expr, Product):
        _count_index_ids_local(expr.lhs, counts)
        _count_index_ids_local(expr.rhs, counts)
    elif isinstance(expr, Scale):
        _count_index_ids_local(expr.expr, counts)


def _contract_kds_locally(expr):
    """Contract KroneckerDelta nodes within a single expression (locally).

    Repeatedly finds a KD whose one or both indices appear exactly once outside
    any KD in this local expression, then calls substitute_index to rename the
    dummy index (which collapses the KD to RationalConst(1) via make_kronecker_delta).
    """
    for _ in range(20):  # safety cap
        kds = []
        _collect_kd_nodes(expr, kds)
        if not kds:
            break
        kd = kds[0]
        lo, hi = kd.lower_id, kd.upper_id
        counts = {}
        _count_index_ids_local(expr, counts)
        lo_ext = counts.get(lo, 0) - 1  # subtract the KD's own contribution
        hi_ext = counts.get(hi, 0) - 1
        if lo_ext == 1 and hi_ext != 1:
            dummy, free = lo, hi
        elif hi_ext == 1 and lo_ext != 1:
            dummy, free = hi, lo
        elif lo_ext == 1 and hi_ext == 1:
            dummy, free = hi, lo
        else:
            break
        expr = substitute_index(expr, dummy, free)
    return expr


def _try_normalize_rank1_form(expr):
    """Try to normalise a rank-1 component expression to 'Rank1[v,dot(s1,s2)]'.

    First contracts any remaining KroneckerDelta nodes locally (term-by-term),
    then recognises expressions with exactly 1 SymBasisVec and 3 AbstractComps:
    - one AC shares its index ID with the SBV  (identifies the vector symbol)
    - the remaining two ACs share a different index ID  (identifies the dot product)
    Returns None if the pattern is not matched.
    """
    expr = _contract_kds_locally(expr)
    acs = []
    _collect_ac_nodes(expr, acs)
    sbvs = []
    _collect_sbv_nodes(expr, sbvs)

    if len(sbvs) != 1 or len(acs) != 3:
        return None

    sbv_idx = sbvs[0].index_id

    vec_acs = [ac for ac in acs if any(idx == sbv_idx for idx, _ in ac.indices)]
    dot_acs = [ac for ac in acs if not any(idx == sbv_idx for idx, _ in ac.indices)]

    if len(vec_acs) != 1 or len(dot_acs) != 2:
        return None

    dot_ids0 = {idx for idx, _ in dot_acs[0].indices}
    dot_ids1 = {idx for idx, _ in dot_acs[1].indices}
    if not (dot_ids0 & dot_ids1):
        return None

    vec_sym = vec_acs[0].base_sym
    dot_syms = sorted([dot_acs[0].base_sym, dot_acs[1].base_sym])
    return f"Rank1[{vec_sym},dot({','.join(dot_syms)})]"


def _normalize_component_form(expr):
    """Return a canonical string for component-form equality checks.

    Sorts the terms of any Sum (order-insensitive) and treats Product as
    commutative (sorted factors), so that ``a^k b_k`` and ``b_k a^k``
    compare equal.

    IndexedSum nodes are normalised letter-invariantly: ``a^i b_i`` and
    ``a^j b_j`` are considered equal, and factor order is ignored so that
    ``a^i b_i`` and ``b_i a^i`` compare equal.

    Scale nodes are normalised as ``Scale[coeff,inner]``.

    TensorProduct nodes that match the rank-1 pattern ``v*(s1·s2)`` are
    normalised as ``Rank1[v,dot(s1,s2)]``.
    """
    if isinstance(expr, AbstractIndexedSum):
        # Normalise letter-invariantly: only the base symbols matter
        parts = sorted([expr.lhs.base_sym, expr.rhs.base_sym])
        return "AbstractIndexedSum[" + ",".join(parts) + "]"
    if isinstance(expr, IndexedSum):
        parts = sorted([(expr.lhs_sym, expr.lhs_sep), (expr.rhs_sym, expr.rhs_sep)])
        return "IndexedSum[" + ",".join(f"{s}{p}" for s, p in parts) + "]"
    if isinstance(expr, Sum):
        terms = sorted(_normalize_component_form(t) for t in expr.terms)
        return "Sum[" + ",".join(terms) + "]"
    if isinstance(expr, Scale):
        inner = _normalize_component_form(expr.expr)
        return f"Scale[{expr.coeff},{inner}]"
    if isinstance(expr, TensorProduct):
        rank1 = _try_normalize_rank1_form(expr)
        if rank1 is not None:
            return rank1
    if isinstance(expr, Product):
        lhs, rhs = expr.lhs, expr.rhs
        # Detect implicit summation: both factors are AbstractComps sharing index IDs
        if isinstance(lhs, AbstractComp) and isinstance(rhs, AbstractComp):
            lhs_ids = {idx for idx, _ in lhs.indices}
            rhs_ids = {idx for idx, _ in rhs.indices}
            if lhs_ids & rhs_ids:
                parts = sorted([lhs.base_sym, rhs.base_sym])
                return "AbstractIndexedSum[" + ",".join(parts) + "]"
        factors = sorted([
            _normalize_component_form(lhs),
            _normalize_component_form(rhs),
        ])
        return "Prod[" + "*".join(factors) + "]"
    return expr.python()


def _label_to_math(label):
    """Render a step label as a LaTeX math-mode fragment.

    Plain identifiers (not part of a \\command) are wrapped in \\mathrm{} so
    they appear upright.  LaTeX commands (\\partial, etc.) are passed through
    unchanged.  Returns a string suitable for use inside \\[...\\].
    """
    import re

    if "\\" not in label:
        # No LaTeX commands — use text mode directly.
        # Escape ^ and _ so they render literally inside \text{} in display math.
        safe = label.replace("_", r"\_").replace("^", r"\^{}")
        return r"\text{[" + safe + r":] }"

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
        r"\usepackage[utf8]{inputenc}",
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
    "show", "show_final", "show_jupyter",
    # LaTeX document export
    "to_latex_document",
    # Documentation
    "doc",
    # Step factories
    "simplify_identity_step", "expand_step", "expand_poly_step",
    "substitute_step", "diff_step",
    "apply_integration_by_parts_step", "apply_divergence_theorem_step",
    "localize_step", "collect_step",
    "expand_in_basis_step", "simplify_basis_dot_step",
    "collect_zero_terms_step", "reassemble_from_components_step",
    "collect_repeated_sum_step", "reassemble_vector_step", "reassemble_dot_step",
    "contract_kronecker_step", "expand_identity_step", "expand_levi_civita_step",
    "expand_levi_civita_first_step", "contract_eps_pair_step",
    "IndexedSum", "make_indexed_sum",
    "SymBasisVec", "make_sym_basis_vec",
    "AbstractComp", "AbstractIndexedSum",
    "make_abstract_comp", "make_abstract_indexed_sum", "alloc_index_id",
    "KroneckerDelta", "LeviCivitaSymbol",
    "make_kronecker_delta", "make_levi_civita_symbol",
    "substitute_index",
    "Theorem",
    "prove_equal_by_components",
    "apply_identity", "find_matches", "apply_identity_auto", "matches_at_root",
    "search_apply",
    "declare_symmetric", "declare_skew_symmetric",
]
