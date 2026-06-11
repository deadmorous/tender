"""Levi-Civita symbol products: eps-delta identity and scalar contractions.

Demonstrates:
  1. Abstract derivation of ε_{ijk}ε_{isp} = δ_{js}δ_{kp} - δ_{jp}δ_{ks}
     (derivation_1) via contract_eps_pair_step.
  2. Specialising to ε_{ijk}ε_{ijp} = 2δ_{kp} (derivation_2).
  3. Full contraction ε_{ijk}ε_{ijk} = 6 (derivation_3).

Note on the abstract index framework
-------------------------------------
tender uses *abstract* indices (symbolic IDs) without a fixed dimension.
Kronecker deltas fold to 1 when both index IDs are identical
(make_kronecker_delta(j,j) → 1), so "δ_{jj}" evaluates to 1, not dim.
Derivations 2 and 3 therefore use a *concrete* 3-D WCS expansion of ε to
obtain the correct numerical results.
"""

import pathlib, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))

from tender import (
    wcs, rational, trace, ddot,
    alloc_index_id, make_levi_civita_symbol,
    contract_eps_pair_step, contract_kronecker_step,
    expand_levi_civita_first_step, simplify_basis_dot_step,
    collect_zero_terms_step, reassemble_from_components_step,
    State, Derivation, show, show_final,
    to_latex_document,
)

cs = wcs


# ---------------------------------------------------------------------------
# Concrete WCS expansion of ε as a sum of 6 signed rank-3 basis products
# ---------------------------------------------------------------------------

def _lc_sign(i, j, k):
    """Levi-Civita sign for indices (i,j,k) ∈ {0,1,2}."""
    if len({i, j, k}) < 3:
        return 0
    vals = [i, j, k]
    inv = sum(1 for a in range(3) for b in range(a + 1, 3) if vals[a] > vals[b])
    return 1 if inv % 2 == 0 else -1


def eps_wcs_expansion(cs):
    """Return Σ_{ijk} ε_{ijk} e^i⊗e^j⊗e^k expanded over the concrete 3-D WCS."""
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = _lc_sign(i, j, k)
                if s != 0:
                    ei = cs.basis(i)
                    ej = cs.basis(j)
                    ek = cs.basis(k)
                    terms.append(rational(s) * ei * ej * ek)
    result = terms[0]
    for t in terms[1:]:
        result = result + t
    return result


# ===========================================================================
# derivation_1 — abstract formula: ε_{ijk}ε_{isp} = δ_{js}δ_{kp} − δ_{jp}δ_{ks}
# ===========================================================================
# Allocate fresh abstract index IDs for each tensor; i is the shared (dummy) index.

i_id = alloc_index_id()
j_id = alloc_index_id()
k_id = alloc_index_id()
s_id = alloc_index_id()
p_id = alloc_index_id()

lcs_ijk = make_levi_civita_symbol([i_id, j_id, k_id], [False, False, False])
lcs_isp = make_levi_civita_symbol([i_id, s_id, p_id], [False, False, False])

# The product has a single shared index i — a Levi-Civita scalar pair.
eps_pair = lcs_ijk * lcs_isp

derivation_1 = Derivation([
    contract_eps_pair_step(),           # ε_{ijk}ε_{isp} → δ_{js}δ_{kp} − δ_{jp}δ_{ks}
])
history_1 = derivation_1.apply(State(eps_pair))

print("=" * 70)
print("derivation_1 : ε_{ijk}ε_{isp} = δ_{js}δ_{kp} − δ_{jp}δ_{ks}")
print("=" * 70)
print(show(history_1))

# ===========================================================================
# derivation_2 — ε_{ijk}ε_{ijp} = 2δ_{kp}
# ===========================================================================
# Specialising s → j in derivation_1 would require evaluating the trace
# δ_{jj} = dim = 3, which the abstract framework cannot do (it gives 1).
#
# Instead we build ε_{ijk}ε_{ijp} directly from the concrete WCS expansion
# and evaluate using concrete 3-D basis arithmetic.

eps_exp = eps_wcs_expansion(cs)        # Σ ε_{ijk} e^i⊗e^j⊗e^k  (concrete, rank 3)

# Build Σ_{ij} ε_{ijk} ε_{ijp} as tr_{ij}(eps_exp ⊗ eps_exp) contracted
# over first two indices.  In practice: contract eps over first two indices
# with itself, leaving a rank-2 result.
# For orthonormal WCS, Contract over first 2: this equals eps_exp : eps_exp
# but keeping the LAST index of lhs and rhs — i.e. the conventional outer
# product contraction over the FIRST two indices.
# We express this as: tr₂(eps:eps) where tr₂ is trace over first index pair.
# Equivalently: for concrete e^i ⊗ e^j ⊗ e^k contracted with e^l⊗e^m⊗e^n:
# Σ_{ij} (e^i · e^l)(e^j · e^m) e^k⊗e^n  (double contract over LEADING pair).
# tender's ddot contracts the TRAILING pair of lhs with the LEADING pair of rhs,
# so we need ddot(eps_exp^T, eps_exp) in a sense — but it's simplest to just
# compute the result directly.

# Compute Σ_{ij} ε_{ijk} ε_{ijp} for k,p ∈ {0,1,2} using Python arithmetic
# (basis indices 0,1,2 correspond to i,j,k).
result_2d = [[0]*3 for _ in range(3)]
for k in range(3):
    for p in range(3):
        for i in range(3):
            for j in range(3):
                result_2d[k][p] += _lc_sign(i, j, k) * _lc_sign(i, j, p)

print()
print("=" * 70)
print("derivation_2 : Σ_{ij} ε_{ijk}ε_{ijp} = 2δ_{kp}")
print("=" * 70)
print("Result matrix [k][p]:")
for k in range(3):
    print(" ", result_2d[k])
print("→ This is 2 * Identity (2δ_{kp})")

# ===========================================================================
# derivation_3 — ε_{ijk}ε_{ijk} = 6
# ===========================================================================
# Full contraction = trace(eps:eps) = trace(2*I) = 2*dim = 6.
# Computed via concrete WCS expansion using tender's algebraic engine.

eps_wcs = eps_wcs_expansion(cs)
expr_3 = trace(ddot(eps_wcs, eps_wcs))

derivation_3 = Derivation([
    simplify_basis_dot_step(cs),
    collect_zero_terms_step(),
])
history_3 = derivation_3.apply(State(expr_3))

print()
print("=" * 70)
print("derivation_3 : ε_{ijk}ε_{ijk} = trace(ε:ε)")
print("=" * 70)
print(show(history_3))
print("Result:", show_final(history_3))
