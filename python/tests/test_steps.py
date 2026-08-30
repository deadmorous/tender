"""The step catalogue (vibe 000106).

The catalogue is the answer to "which step do I need?" — every step, its
category, and what it wants besides the expression.  These tests keep it
honest: an entry that has drifted from the function it names, or a step that
exists but was never catalogued, is worse than no catalogue at all.
"""

import pytest

import tender
import tender.basis as tb
import tender.derivation as td
import tender.steps as ts


class TestCatalogueIsHonest:
    def test_every_entry_names_the_real_function(self):
        import importlib

        for name in ts.names():
            s = ts.info(name)
            mod = importlib.import_module(s.home)
            assert getattr(mod, name) is s.fn, name

    def test_every_entry_has_a_known_category_and_a_summary(self):
        for name in ts.names():
            s = ts.info(name)
            assert s.category in ts.CATEGORIES, name
            assert s.summary and not s.summary.endswith("."), name

    def test_every_advertised_step_is_catalogued(self):
        # The reconciliation the identity DAG does for rules: a step on the
        # advertised surface that nobody catalogued is a step nobody can find.
        catalogued = set(ts.names())
        not_steps = {  # types, predicates, factories, combinators
            "Budget", "BudgetExceeded", "Derivation", "Identity", "NoOpStep",
            "PREFER", "ProofResult", "algebraic_eq", "structural_eq",
            "prove_equal", "rules", "rule_groups", "citable_for",
            "default_budget", "set_default_budget", "at", "apply_identity",
            "deriv", "Basis", "Handedness", "Variance", "wcs", "cylindrical",
            "spherical", "polar_2d", "make_orthonormal_basis",
            "make_oblique_basis",
        }
        for mod in (td, tb):
            for name in mod.__all__:
                if name in not_steps:
                    continue
                assert name in catalogued, f"{name} is advertised but not catalogued"

    def test_the_demoted_moves_are_still_importable(self):
        # Demotion is about the *advertised* surface; nothing was removed.
        for mod, name in (
            (td, "canonicalize"), (td, "unroll_sums"), (td, "implicitize"),
            (tb, "simplify_basis_dot"), (tb, "fold_resolution_of_identity"),
        ):
            assert name not in mod.__all__, name
            assert callable(getattr(mod, name)), name

    def test_the_bridge_primaries_are_the_four(self):
        assert [s.name for s in ts.in_category("bridge") if s.primary] == [
            "expand_in_basis", "reassemble", "reduce_frame", "to_concrete",
        ]


class TestStepCall:
    def _expanded(self):
        ctx = tender.Context()
        frame = tb.wcs(ctx)
        a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
        return frame, tb.expand_in_basis(a @ b, frame)

    def test_a_step_takes_its_extra_arguments_from_context(self):
        frame, e = self._expanded()
        direct = tb.reduce_frame(e, frame)
        viaCatalogue = ts.info("reduce_frame")(e, basis=frame)
        assert td.structural_eq(direct, viaCatalogue)

    def test_context_may_carry_more_than_a_step_needs(self):
        # The point of `needs`: one context serves every step.
        frame, e = self._expanded()
        ctx = dict(basis=frame, level=tender.Level.Upper, rules=[])
        assert td.structural_eq(
            ts.info("canonicalize")(e, **ctx), td.canonicalize(e)
        )

    def test_a_missing_need_says_what_is_missing(self):
        frame, e = self._expanded()
        with pytest.raises(TypeError, match="needs.*coord"):
            ts.info("partial")(e, basis=frame)

    def test_optional_arguments_are_passed_when_present(self):
        frame, e = self._expanded()
        comps = tb.reduce_frame(e, frame)
        assert td.structural_eq(
            ts.info("reassemble")(comps, basis=frame, target="a"),
            tb.reassemble(comps, frame, target="a"),
        )

    def test_almost_every_step_runs_from_a_basis_alone(self):
        # The measured claim behind `applicable`: the extra arguments are few
        # and drawn from a closed list, so one context covers nearly everything.
        frame, e = self._expanded()
        needs_more = []
        for name in ts.names():
            try:
                ts.info(name)(e, basis=frame)
            except TypeError:
                needs_more.append(name)
            except Exception:
                pass  # a domain error is fine; the call itself worked
        assert sorted(needs_more) == [
            "engine_simplify", "fold_operator", "insert_metric", "partial",
            "saturate",
        ]


class TestRegistration:
    def test_a_user_can_add_their_own_step(self):
        ts.register(
            "double_it",
            lambda e: e + e,
            category="normalise",
            summary="a test step",
            home="tests",
        )
        try:
            assert "double_it" in ts.names("normalise")
            ctx = tender.Context()
            a = tender.tensor("a", rank=1, ctx=ctx)
            assert td.algebraic_eq(ts.info("double_it")(a), a + a)
        finally:
            ts._STEPS.pop("double_it")

    def test_an_unknown_category_is_refused(self):
        with pytest.raises(ValueError, match="unknown category"):
            ts.register("x", lambda e: e, category="misc", summary="s")

    def test_an_unknown_name_is_refused(self):
        with pytest.raises(ValueError, match="unknown step"):
            ts.info("no_such_step")


def test_describe_covers_every_category():
    text = ts.describe()
    for cat in ts.CATEGORIES:
        assert cat in text
    assert "reduce_frame" in text
