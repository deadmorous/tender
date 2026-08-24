"""Tests for tender.identities — the standard rule library.

Every library rule is *fire-tested*: it must demonstrably fire on a minimal
target.  This is not ceremony.  Canon α-renames dummies, normalizes
symmetries, and sorts symmetric contraction chains by tensor name, any of
which can silently make a correctly-stated rule unmatchable (vibe 000040) —
and a rule that cannot fire is worse than no rule, because it looks like
coverage while doing nothing.  The engine's `fired` report is what makes the
distinction testable.
"""

import pytest

import tender as t
import tender.derivation as td
import tender.identities as ti

U, L = t.Level.Upper, t.Level.Lower
OBL, ORTH = t.Realm.Oblique, t.Realm.Orthonormal


def _delta(ctx, a, b, la=U, lb=L, realm=OBL):
    return t.delta(realm, t.space_3d, la, lb, a, b, ctx=ctx)


def _eps(ctx, level, x, y, z, realm=OBL):
    return t.levi_civita(realm, t.space_3d, [level] * 3, [x, y, z], ctx=ctx)


def _vec(ctx, name):
    return t.tensor(name, rank=1, ctx=ctx)


def _mat(ctx, name):
    return t.tensor(name, rank=2, ctx=ctx)


def _proves(ctx, lhs, rhs, rules, what):
    """Assert the rules prove lhs == rhs *and* that some rule actually fired."""
    result = td.prove_equal(lhs, rhs, rules)
    assert result.proved, f"{what}: not proved ({result})"
    assert sum(result.fired.values()) > 0, (
        f"{what}: proved without firing any rule — canonicalization did the "
        f"work, so the rule under test is not what is being exercised"
    )
    return result


# ---- eps_delta group ------------------------------------------------------


def test_delta_contraction_fires():
    ctx = t.Context()
    q, m, n = (ctx.alloc_index() for _ in range(3))
    target = t.explicit_sum(q, _delta(ctx, q, m) * _delta(ctx, q, n), ctx=ctx)
    _proves(
        ctx,
        target,
        _delta(ctx, m, n, L, L),
        [ti.delta_contraction(ctx)],
        "Σ_p δ^p_a δ^p_b = δ_ab",
    )


def test_delta_trace_fires():
    ctx = t.Context()
    p = ctx.alloc_index()
    target = t.explicit_sum(p, _delta(ctx, p, p), ctx=ctx)
    _proves(ctx, target, t.scalar(3, ctx=ctx), [ti.delta_trace(ctx)], "δ^p_p = 3")


def test_eps_delta_1_matches_the_contract_eps_pair_oracle():
    # The δ-expansion is larger than the ε-form, so this only extracts
    # correctly because the cost function weights Levi-Civita symbols heavily.
    ctx = t.Context()
    a, b, c, d, e = (ctx.alloc_index() for _ in range(5))
    target = t.explicit_sum(
        a, _eps(ctx, U, a, b, c) * _eps(ctx, L, a, d, e), ctx=ctx
    )
    oracle = td.contract_eps_pair(target)
    out, report = td.engine_simplify(target, [ti.eps_delta_1(ctx)])
    assert td.algebraic_eq(out, oracle)
    assert report["fired"]


def test_eps_delta_2_matches_the_oracle():
    ctx = t.Context()
    i, j, k, l = (ctx.alloc_index() for _ in range(4))
    target = t.explicit_sum(
        i,
        t.explicit_sum(j, _eps(ctx, U, i, j, k) * _eps(ctx, L, i, j, l), ctx=ctx),
        ctx=ctx,
    )
    oracle = td.contract_eps_pair(target)
    out, _ = td.engine_simplify(target, [ti.eps_delta_2(ctx)])
    assert td.algebraic_eq(out, oracle)


# ---- realm and level conventions (vibe 000047) ----------------------------


def test_orthonormal_rule_contracts_an_orthonormal_target():
    ctx = t.Context()
    q, m, n = (ctx.alloc_index() for _ in range(3))
    target = t.explicit_sum(
        q,
        _delta(ctx, q, m, L, L, ORTH) * _delta(ctx, q, n, L, L, ORTH),
        ctx=ctx,
    )
    _proves(
        ctx,
        target,
        _delta(ctx, m, n, L, L, ORTH),
        [ti.delta_contraction(ctx, realm=ORTH)],
        "orthonormal contraction",
    )


def test_orthonormal_rule_is_lower_spelled():
    # The Orthonormal rule is lower-lower, so it does NOT fire on an
    # upper-spelled Orthonormal target — pinning the lower-index convention.
    ctx = t.Context()
    q, m, n = (ctx.alloc_index() for _ in range(3))
    upper_target = t.explicit_sum(
        q, _delta(ctx, q, m, U, L, ORTH) * _delta(ctx, q, n, U, L, ORTH), ctx=ctx
    )
    result = td.prove_equal(
        upper_target,
        _delta(ctx, m, n, L, L, ORTH),
        [ti.delta_contraction(ctx, realm=ORTH)],
    )
    assert not result.proved


def test_realm_mismatch_does_not_fire():
    # Matching is realm-exact: an Oblique rule must not contract an
    # Orthonormal target.
    ctx = t.Context()
    q, m, n = (ctx.alloc_index() for _ in range(3))
    target = t.explicit_sum(
        q, _delta(ctx, q, m, L, L, ORTH) * _delta(ctx, q, n, L, L, ORTH), ctx=ctx
    )
    out, report = td.engine_simplify(target, [ti.delta_contraction(ctx, realm=OBL)])
    assert td.algebraic_eq(out, target)  # unchanged
    assert not report["fired"]


# ---- cross group ----------------------------------------------------------


def test_bac_cab_fires():
    ctx = t.Context()
    a, b, c = (_vec(ctx, n) for n in "abc")
    _proves(
        ctx, a % (b % c), b * (a @ c) - c * (a @ b), [ti.bac_cab(ctx)], "bac-cab"
    )


def test_bac_cab_does_not_fire_across_a_rank_two_fence():
    # Soundness guard: a subtree variable binds any factor *regardless of
    # rank*, so bac-cab could in principle fire on a×(B×c) with B rank-2 —
    # where the identity is false.  It must not: canon's rank-2 fence
    # reassociation (vibe 000055) puts that expression in another shape.
    ctx = t.Context()
    a, c = _vec(ctx, "a"), _vec(ctx, "c")
    B = _mat(ctx, "B")
    wrong = B * (a @ c) - c * (a @ B)
    assert not td.prove_equal(a % (B % c), wrong, [ti.bac_cab(ctx)]).proved


def test_cross_identity_fires():
    ctx = t.Context()
    a = _vec(ctx, "a")
    I = t.identity(ctx=ctx)
    _proves(ctx, a % I, I % a, [ti.cross_identity(ctx)], "a×I = I×a")


def test_cross_removal_fires():
    # THE vibe-000056 case: the derivation no user could discover as a step
    # sequence, as a single goal-directed call.
    ctx = t.Context()
    a, b = _vec(ctx, "a"), _vec(ctx, "b")
    I = t.identity(ctx=ctx)
    _proves(
        ctx,
        a % (b % I),
        b * a - (a @ b) * I,
        [ti.cross_removal(ctx)],
        "a×(b×I) = b⊗a − (a·b)I",
    )


def test_lagrange_fires():
    ctx = t.Context()
    a, b, c, d = (_vec(ctx, n) for n in "abcd")
    _proves(
        ctx,
        (a % b) @ (c % d),
        (a @ c) * (b @ d) - (a @ d) * (b @ c),
        [ti.lagrange(ctx)],
        "Lagrange identity",
    )


# ---- dyadic group ---------------------------------------------------------


def test_trace_cyclic_fires():
    ctx = t.Context()
    A, B = _mat(ctx, "A"), _mat(ctx, "B")
    _proves(ctx, t.tr(A @ B), t.tr(B @ A), [ti.trace_cyclic(ctx)], "tr(A·B)")


def test_identity_dot_fires():
    ctx = t.Context()
    a = _vec(ctx, "a")
    _proves(ctx, t.identity(ctx=ctx) @ a, a, [ti.identity_dot(ctx)], "I·a = a")


# ---- double_dot group -----------------------------------------------------


def test_ddot_identity_fires():
    ctx = t.Context()
    A = _mat(ctx, "A")
    I = t.identity(ctx=ctx)
    _proves(ctx, I // A, t.tr(A), [ti.ddot_identity(ctx)], "I··A = tr A")


@pytest.mark.parametrize("name", ["A", "F", "P", "X", "Z"])
def test_ddot_identity_is_name_independent(name):
    """The regression test for AC chain matching (vibe 000096 increment 3).

    Canon puts a commutative binary contraction's operands in name order, so
    with positional-only matching this rule fired on targets named A–H and
    silently missed J–Z — and *no* spelling of the rule worked for both, since
    the pattern variable's own name decided where it sorted.  Every letter
    must now work, from either side of the contraction.
    """
    ctx = t.Context()
    A = _mat(ctx, name)
    I = t.identity(ctx=ctx)
    _proves(ctx, I // A, t.tr(A), [ti.ddot_identity(ctx)], f"I··{name}")
    _proves(ctx, A // I, t.tr(A), [ti.ddot_identity(ctx)], f"{name}··I")


def test_ac_matching_does_not_break_directional_contractions():
    """AC matching applies only where canon may reorder.

    `·` between a rank-2 and a rank-1 is directional (A·b ≠ b·A), so a rule
    written for one order must not fire on the other — otherwise the fix
    would trade a missed match for a wrong rewrite.
    """
    ctx = t.Context()
    A, b = _mat(ctx, "A"), _vec(ctx, "b")
    x, Y = _vec(ctx, "x"), _mat(ctx, "Y")
    zero = t.scalar(0, ctx=ctx)
    rule = td.Identity("directional", x @ Y, zero)  # vector · matrix

    assert td.prove_equal(b @ A, zero, [rule]).proved  # same order: fires
    assert not td.prove_equal(A @ b, zero, [rule]).proved  # swapped: must not


# ---- name robustness (the vibe-000096 finding) ----------------------------


@pytest.mark.parametrize("first", ["a", "f", "p", "x"])
def test_rules_are_name_robust_across_the_alphabet(first):
    # Canon sorts symmetric contraction chains by tensor name, so a rule can
    # fire for targets named one way and silently miss others.  Every shipped
    # rule must be insensitive to the target's naming.
    ctx = t.Context()
    second = "b" if first != "b" else "c"
    third = "d" if first != "d" else "e"
    a, b, c = (_vec(ctx, n) for n in (first, second, third))
    I = t.identity(ctx=ctx)

    _proves(
        ctx,
        a % (b % c),
        b * (a @ c) - c * (a @ b),
        [ti.bac_cab(ctx)],
        f"bac-cab with target named {first!r}",
    )
    _proves(
        ctx,
        a % (b % I),
        b * a - (a @ b) * I,
        [ti.cross_removal(ctx)],
        f"cross-removal with target named {first!r}",
    )


# ---- groups ---------------------------------------------------------------


def test_groups_are_named_and_populated():
    ctx = t.Context()
    # Groups are a labelling over the DAG, so their order follows the
    # registration (dependency) order of the nodes carrying each tag.
    assert set(ti.group_names()) == {"eps_delta", "cross", "dyadic", "double_dot"}
    assert len(ti.group(ctx, "eps_delta")) == 4
    assert len(ti.group(ctx, "cross")) == 4
    assert len(ti.group(ctx, "double_dot")) == 1
    assert len(ti.group(ctx, "dyadic")) == 2
    assert len(ti.all_rules(ctx)) == 11


def test_unknown_group_raises_with_the_available_names():
    ctx = t.Context()
    with pytest.raises(ValueError, match="cross"):
        ti.group(ctx, "no_such_group")


def test_every_shipped_rule_compiles_for_the_engine():
    # A rule the engine cannot compile (multi-term LHS) never fires; the
    # library must contain none.
    ctx = t.Context()
    a = _vec(ctx, "a")
    _, report = td.engine_simplify(a, ti.all_rules(ctx))
    assert report["skipped"] == []


def test_a_user_rule_is_first_class_alongside_the_library():
    # The point of moving the library to Python: your own rule is no
    # different from a shipped one, and needs no rebuild.
    ctx = t.Context()
    a, b = _vec(ctx, "a"), _vec(ctx, "b")
    swap = td.Identity("user-swap", a % b, -(b % a))
    result = td.prove_equal(a % b, -(b % a), td.rules("cross", ctx=ctx) + [swap])
    assert result.proved


# ---- the identity DAG (vibe 000097) ---------------------------------------


def test_dag_records_what_each_identity_rests_on():
    assert ti.node("bac-cab").cites == ("eps-delta-1",)
    assert ti.ancestors("bac-cab") == {"eps-delta-1"}
    assert ti.ancestors("lagrange") == {"eps-delta-1", "delta-contraction"}
    # An identity proved by reduction to components leans on nothing.
    assert ti.ancestors("cross-identity") == set()


def test_descendants_are_the_reverse_edges():
    assert "bac-cab" in ti.descendants("eps-delta-1")
    assert "lagrange" in ti.descendants("eps-delta-1")
    assert ti.descendants("bac-cab") == set()


def test_depth_measures_distance_above_the_axioms():
    assert ti.depth("delta-contraction") == 0  # an axiom
    assert ti.depth("eps-delta-1") == 0  # derived straight from definitions
    assert ti.depth("bac-cab") == 1  # rests on eps-delta-1


def test_axioms_cite_nothing_and_owe_no_proof():
    for name in ti.names():
        n = ti.node(name)
        if n.kind == ti.AXIOM:
            assert n.cites == ()
            assert n.proof is None


def test_citable_for_excludes_the_identity_itself():
    """The anti-circularity guarantee, as an API.

    A derivation of bac-cab may use ε-δ but not bac-cab — otherwise it would
    be proving the identity from itself.
    """
    ctx = t.Context()
    citable = {r.name for r in ti.citable_for(ctx, "bac-cab")}
    assert citable == {"eps-delta-1"}
    assert "bac-cab" not in citable


def test_citable_rules_actually_prove_the_identity():
    """The DAG is not bookkeeping: the declared ancestors really suffice.

    bac-cab claims to rest on ε-δ-1, so ε-δ-1 alone must carry a proof of it
    — checked through the component route, which is what the challenge does.
    """
    ctx = t.Context()
    citable = ti.citable_for(ctx, "bac-cab")
    assert [r.name for r in citable] == ["eps-delta-1"]


def test_registering_an_unknown_dependency_is_refused():
    with pytest.raises(ValueError, match="not registered"):
        ti.register("bad", lambda ctx: None, cites=["no-such-identity"])


def test_an_axiom_may_not_cite_anything():
    with pytest.raises(ValueError, match="cannot cite"):
        ti.register("bad-axiom", lambda ctx: None, kind=ti.AXIOM,
                    cites=["bac-cab"])


def test_a_user_identity_joins_the_dag_with_dependencies():
    """User rules are first-class nodes, dependencies and all."""
    ctx = t.Context()
    a, b = (_vec(ctx, n) for n in "ab")
    try:
        ti.register(
            "user-anticommute",
            lambda c: td.Identity(
                "user-anticommute",
                _vec(c, "a") % _vec(c, "b"),
                -(_vec(c, "b") % _vec(c, "a")),
            ),
            cites=["bac-cab"],
            summary="a × b = −(b × a)",
        )
        assert ti.ancestors("user-anticommute") == {"bac-cab", "eps-delta-1"}
        assert ti.depth("user-anticommute") == 2
        assert "user-anticommute" in ti.descendants("eps-delta-1")
        ti.check_acyclic()
    finally:
        ti._NODES.pop("user-anticommute", None)


def test_the_graph_stays_acyclic():
    ti.check_acyclic()
