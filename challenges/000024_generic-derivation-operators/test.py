"""Leibniz belongs to *derivations*, not to ∇.

∇ is not a primitive.  Eliseev writes it as `∇ = R^i ∂/∂q^i` — a cobasis
contracted with partial derivatives — and the same section builds three more
operators the same way (§5.5 eq. 5.2, `vibes/images/`):

    ∇⊥ ≡ e_α ∂/∂x_α          (the same shape, over fewer indices)
    D  ≡ t · x × ∇⊥          (a *scalar* operator, built from another one)
    ∇  = ∇⊥ + v⁻¹ t (∂_s − Ω_t D)

So the object that obeys the Leibniz rule is any **derivation**

    Op = Σ_k c_k ⊗ ∂_k ,

with `c_k` arbitrary coefficients — and ∇ is merely the instance whose
coefficients are a cobasis.  This challenge certifies that tender treats it
that way: that the product rule is a property of the *form*, not of ∇, and
that it holds

  * for an operator the library has never heard of,
  * over any number of factors, and
  * without a rule being registered, a trait declared, or a chart consulted.

Why it matters beyond tidiness: thin-rod asymptotics substitutes the split
above and collects orders in a small parameter.  If Leibniz were ∇-specific,
every operator in that substitution would need its own rules; here they all
work because they are all the same kind of thing (vibe 000102).
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="Leibniz holds for any derivation Σ c_k ⊗ ∂_k, not just ∇",
    tier="C",
    source="Eliseev, Mechanics of Deformable Bodies §5.5 eq. (5.2); vibe 000102",
)


def _setup():
    ws = t.Workspace()
    cart, (x, y, z) = ws.cartesian_chart()
    e = cart.physical_frame()
    return ws, cart, (x, y, z), e


@harness.level("L1")
def test_leibniz_holds_for_an_operator_the_library_never_heard_of():
    """A hand-built ∇⊥ obeys the product rule — no rule, no trait, no chart.

    `grad_perp` is assembled here, in the test, out of two basis vectors and
    two partials.  Nothing registers it; nothing declares that it is a
    derivation.  It obeys Leibniz because of what it *is*.
    """
    ws, cart, (x, y, z), e = _setup()
    f, g = cart.field("f", 0), cart.field("g", 0)

    grad_perp = e.direction(0) * td.deriv(x) + e.direction(1) * td.deriv(y)
    show("∇⊥ (built, not named)", grad_perp)

    applied = td.simplify_scalars(td.apply_operators(grad_perp * (f * g)))
    # The comparison needs the right-hand side distributed: theory T0
    # deliberately excludes distributivity, so `g (∂_x f i + ∂_y f j)` and its
    # expansion are equal mathematically but not canonically.
    expected = td.simplify_scalars(
        td.expand_products(
            td.apply_operators(grad_perp * f) * g
            + f * td.apply_operators(grad_perp * g)
        )
    )
    show("∇⊥(f g)", applied)
    harness.assert_algebraic_eq(applied, expected, "Leibniz for a custom ∇⊥")


@harness.level("L1")
def test_the_product_rule_is_n_ary_for_free():
    """Four factors, one pass — no re-association, no repeated rule firing.

    A rule written as a *pattern* has a fixed arity: `D(F·G)` matches two
    factors and no more.  Here the rule is *computed* over an `Nf` term, which
    is already n-ary, so `D(a b c d)` yields its four terms directly.  This is
    a dividend of the M1 flattening (vibe 000102 Q1).
    """
    ws, cart, (x, y, z), e = _setup()
    fields = [cart.field(n, 0) for n in "fghk"]
    op = e.direction(0) * td.deriv(x)

    product = fields[0] * fields[1] * fields[2] * fields[3]
    applied = td.simplify_scalars(td.apply_operators(op * product))

    # One term per factor differentiated: exactly four.
    assert applied.latex().count(r"\partial_{x}") == 4, applied.latex()
    show("∂_x(f g h k), 4 terms", applied)

    expected = None
    for i in range(4):
        others = [fields[j] for j in range(4) if j != i]
        term = td.apply_operators(op * fields[i])
        for o in others:
            term = term * o
        expected = term if expected is None else expected + term
    harness.assert_algebraic_eq(
        applied, td.simplify_scalars(expected), "n-ary product rule"
    )


@harness.level("L1")
def test_the_nabla_leibniz_rules_are_derived_not_asserted():
    """The `leibniz` group's content, obtained rather than postulated.

    The library ships `∇(fg) = f∇g + g∇f` as an identity.  Here the same
    statement is *derived*: expand ∇ into `Σ_i c_i ⊗ ∂_i`, apply the product
    rule to the term, and compare.  Nothing cites the shipped rule — this is
    the proof it is entitled to (vibe 000102, option C).
    """
    ws, cart, (x, y, z), e = _setup()
    f, g = cart.field("f", 0), cart.field("g", 0)

    nabla_expanded = cart.nabla()  # Σ_i (1/h_i) e_i ∂_i — the definition
    show("∇ expanded", nabla_expanded)

    derived = td.simplify_scalars(td.apply_operators(nabla_expanded * (f * g)))
    claimed = td.simplify_scalars(
        td.expand_products(
            td.apply_operators(nabla_expanded * f) * g
            + f * td.apply_operators(nabla_expanded * g)
        )
    )
    show("∇(fg), derived", derived)
    harness.assert_algebraic_eq(
        derived, claimed, "∇(fg) = f∇g + g∇f, derived from the expansion"
    )


@harness.level(
    "L2",
    expected=False,
    reason="the derived form stays in components: here the summed index joins "
    "a coefficient c_i to a ∂_i mark rather than to a basis vector, and no "
    "fold reads that pairing yet (vibe 000103's operator row) — the "
    "contraction-descending reassembly it was blamed on is now in place",
)
def test_the_derivation_returns_to_invariant_form():
    """What is still missing: the trip home.

    Everything above ends in components — `f ∂_x g i + …` — and cannot be
    folded back into `∇`.  Vibe 000103 removed one suspected cause: `reassemble`
    *does* now descend into a contraction operand, so a coordinate paired with a
    basis vector nested inside a dot folds fine.  That is not what blocks this.

    What blocks it is a different row of the same fold table: the summed index
    here connects a coefficient `c_i` to a **∂_i mark**, not to a basis vector,
    and nothing reads that pairing yet.  The `link` tying a free-index ∂ to its
    frame vector already exists in the representation (vibe 000078) — what is
    missing is a pass that acts on it.  Until then a *derived* Leibniz rule
    cannot be restated as the invariant identity the library ships, which is why
    those rules remain asserted.
    """
    harness.todo(
        "fold Σ_i c_i ⊗ ∂_i(f g) back to f ∇g + g ∇f in direct notation "
        "(needs the coefficient↔∂-mark row of vibe 000103's fold table)"
    )
