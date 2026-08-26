"""The identity DAG and the challenge suite, checked against each other.

tender's library owns the DAG of identities (:mod:`tender.identities`): which
identities exist, what each rests on, and — as inert data — which challenge
derives it.  This suite owns the derivations.  Neither imports the other's
internals; this file is the single place they are reconciled, and it is
development scaffolding, not a library feature.

The obligations checked here, in both directions:

* the graph is acyclic and every citation names a registered identity;
* every ``proves=`` on a challenge names a real identity, and no two
  challenges claim the same one;
* every *derived* identity has a challenge that derives it (the one open gap
  is enumerated as a strict xfail, not hidden);
* an *axiom* has no proof obligation, and cites nothing.
"""

import pathlib
import re

import pytest

import tender.identities as ti


def _challenge_dirs():
    root = pathlib.Path(__file__).resolve().parent
    return sorted(
        d for d in root.iterdir() if d.is_dir() and re.fullmatch(r"\d{6}_.+", d.name)
    )


def _declared():
    """Every challenge's (number, CHALLENGE dict), read without importing.

    Parsing beats importing here: it keeps this meta-test independent of
    whether the challenges themselves pass, so a broken derivation shows up as
    that one challenge failing rather than as the DAG check collapsing too.
    """
    out = {}
    for d in _challenge_dirs():
        text = (d / "test.py").read_text()
        number = d.name.split("_", 1)[0]
        match = re.search(r"proves\s*=\s*(\[[^]]*\]|\"[^\"]+\")", text)
        out[number] = set(re.findall(r'"([^"]+)"', match.group(1))) if match else set()
    return out


# ---- the graph itself -----------------------------------------------------


def test_dag_is_acyclic():
    ti.check_acyclic()


def test_every_citation_names_a_registered_identity():
    for name in ti.names():
        for cited in ti.node(name).cites:
            assert cited in ti.names(), f"{name} cites unregistered {cited!r}"


def test_no_identity_cites_itself_or_a_descendant():
    """The circularity check the DAG exists for.

    An identity's derivation may lean only on identities that stand *below*
    it.  Citing itself, or anything that already rests on it, would make the
    proof circular — and this makes that a detectable bug rather than a matter
    of review.
    """
    for name in ti.names():
        assert name not in ti.ancestors(name), f"{name} transitively cites itself"
        overlap = ti.ancestors(name) & ti.descendants(name)
        assert not overlap, f"{name}: cited identities also depend on it: {overlap}"


def test_axioms_carry_no_proof_obligation():
    for name in ti.names():
        n = ti.node(name)
        if n.kind == ti.AXIOM:
            assert n.cites == (), f"axiom {name} cites {list(n.cites)}"
            assert n.proof is None, f"axiom {name} names a proof ({n.proof})"


def test_every_identity_states_what_it_says():
    for name in ti.names():
        assert ti.node(name).summary, f"{name} has no summary"


# ---- the graph against the suite ------------------------------------------


def test_every_proves_names_a_real_identity():
    for number, proves in _declared().items():
        for name in proves:
            ti.node(name)  # raises ValueError if unknown


def test_no_two_challenges_prove_the_same_identity():
    claimed = {}
    for number, proves in _declared().items():
        for name in proves:
            assert name not in claimed, (
                f"challenges {claimed[name]} and {number} both claim to prove "
                f"{name!r} — an identity has one derivation of record"
            )
            claimed[name] = number


def test_declared_proof_matches_the_challenge_that_claims_it():
    """The node's `proof` field and the challenge's `proves=` must agree.

    They are two halves of the same statement, written on opposite sides of
    the library/suite boundary; letting them drift apart is exactly how the
    obligation would quietly stop meaning anything.
    """
    declared = _declared()
    for name in ti.names():
        expected = ti.node(name).proof
        if expected is None:
            continue
        assert expected in declared, (
            f"identity {name!r} names challenge {expected} as its proof, but "
            f"no such challenge directory exists"
        )
        assert name in declared[expected], (
            f"identity {name!r} names challenge {expected} as its proof, but "
            f"that challenge proves {sorted(declared[expected])}"
        )


def test_every_claim_is_acknowledged_by_the_identity():
    """The other direction of the correspondence.

    `test_declared_proof_matches…` walks nodes that name a proof.  This walks
    challenges that claim one, so a challenge asserting `proves="x"` while
    node `x` still says `proof=None` is caught — a one-way declaration that
    would otherwise pass silently and leave the obligation looking open.
    """
    for number, proves in _declared().items():
        for name in proves:
            assert ti.node(name).proof == number, (
                f"challenge {number} claims to prove {name!r}, but that "
                f"identity names {ti.node(name).proof!r} as its proof"
            )


UNPROVEN = set()  # every derived identity now has a derivation


def test_open_proof_obligations_are_exactly_the_known_ones():
    """Derived identities with no challenge deriving them.

    Not a failure — a visible backlog.  The set is pinned so that adding an
    identity without a derivation, or supplying one at last, both show up as a
    deliberate change here.
    """
    open_now = {
        name
        for name in ti.names()
        if ti.node(name).kind == ti.DERIVED and ti.node(name).proof is None
    }
    assert open_now == UNPROVEN, (
        f"open proof obligations changed: {open_now} (was {UNPROVEN}) — "
        f"update UNPROVEN if this is intended"
    )


def test_every_derived_identity_is_proven():
    for name in ti.names():
        n = ti.node(name)
        if n.kind == ti.DERIVED:
            assert n.proof is not None, f"{name} has no derivation"
