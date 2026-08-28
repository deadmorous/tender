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


@harness.level("L2")
def test_the_derivation_returns_to_invariant_form():
    """The trip home: from components back to direct notation.

    Everything above ends in components — `f ∂ₓg i + …` — because that is what
    applying the operator produces.  Stating the *result* as an identity needs
    the return trip, and `apply_operators` had no inverse.

    `fold_operator` is that inverse, and it is the operator row of vibe 000103's
    fold table: the summed direction joins a coefficient `c_k` to a `∂_k` mark,
    and reading that pairing back is what folds the group into the operator.
    Where the other rows fold one index cluster, this one folds a *complete
    group of addends* — every direction present, all agreeing on the operand,
    the sign, and the company — which is the same completeness argument
    `fold_resolution_of_identity` makes for `Σ_k e_k⊗e_k = I`.

    The caller passes the operator, and with it the claim that this expansion
    *is* that operator.  That is deliberate: nothing in the library knows that
    `i∂ₓ + j∂_y` deserves to be read back as one thing, and ∇⊥ has no name here
    — it was assembled in the test.  This is vibe 000102's Q2 answered the way
    it was posed: the user declares the equivalence and owns it.
    """
    ws, cart, (x, y, z), e = _setup()
    f, g, h = (cart.field(n, 0) for n in "fgh")
    grad_perp = e.direction(0) * td.deriv(x) + e.direction(1) * td.deriv(y)

    applied = td.simplify_scalars(td.apply_operators(grad_perp * (f * g)))
    folded = td.fold_operator(applied, grad_perp)
    show("∇⊥(fg), folded back", folded)
    harness.assert_algebraic_eq(
        folded,
        f * (grad_perp * g) + g * (grad_perp * f),
        "∇⊥(fg) = f ∇⊥g + g ∇⊥f, in direct notation",
    )
    # No derivative marks survive: the ∂'s are back inside the operator.
    assert "partial_{x} f" not in folded.latex()

    # n-ary, likewise — the rule the L1 test above derives, now statable.
    ternary = td.fold_operator(
        td.simplify_scalars(td.apply_operators(grad_perp * (f * g * h))),
        grad_perp,
    )
    show("∇⊥(fgh), folded back", ternary)
    harness.assert_algebraic_eq(
        ternary,
        f * g * (grad_perp * h)
        + f * h * (grad_perp * g)
        + g * h * (grad_perp * f),
        "∇⊥(fgh), the three-factor rule in direct notation",
    )

    # An incomplete group is not this operator: one direction alone stays put
    # rather than being folded into something it is not.
    partial = td.simplify_scalars(
        td.apply_operators(e.direction(0) * td.deriv(x) * f)
    )
    assert td.structural_eq(td.fold_operator(partial, grad_perp), partial)


@harness.level("L2")
def test_the_library_operator_returns_to_its_own_name():
    """The same round trip for ∇, which the library *does* name.

    ∇⊥ folds back to its expansion because it has no name.  ∇ does have one, so
    its trip home ends in the symbol itself: expand it into the frame, let
    Leibniz act, and `reassemble_nabla` reads the frame-vector/∂-mark pairing
    back into a chart-free ∇.  The shipped `leibniz` rules state this identity;
    here it is obtained.
    """
    ws, cart, (x, y, z), e = _setup()
    # expand_nabla needs a uniform ∂_i, so the fields depend on every
    # coordinate rather than on a named subset.
    f, g = ws.field("f", 0), ws.field("g", 0)

    expanded = cart.expand_nabla(ws.nabla() * (f * g))
    show("∇(fg) in the frame", expanded)
    derived = td.simplify_scalars(td.apply_operators(expanded))
    back = cart.reassemble_nabla(derived)
    show("∇(fg), folded back", back)

    harness.assert_algebraic_eq(
        back,
        f * (ws.nabla() * g) + g * (ws.nabla() * f),
        "∇(fg) = f∇g + g∇f, derived and restated invariantly",
    )
    assert "partial" not in back.latex()
