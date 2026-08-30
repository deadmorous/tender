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


# ---------------------------------------------------------------------------
# Feedback: applicable / why_not / explain (vibe 000106 §4)
# ---------------------------------------------------------------------------


class TestApplicable:
    def _setup(self):
        ctx = tender.Context()
        frame = tb.wcs(ctx)
        a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
        return ctx, frame, a, b

    def test_it_reports_the_steps_that_change_the_content(self):
        _, frame, a, b = self._setup()
        e = tb.expand_in_basis(a @ b, frame)
        got = {h.step.name for h in ts.applicable(e, basis=frame).changing}
        # The moves a person would consider here, and no others invented.
        assert {"reduce_frame", "simplify_basis_dot", "reassemble"} <= got

    def test_reshaping_steps_are_separated_from_the_real_options(self):
        # The refinement that makes the report usable: "not a no-op" is too weak
        # a filter, because canonical reordering fires on almost anything.
        _, frame, a, b = self._setup()
        e = tb.expand_in_basis(a @ b, frame)
        report = ts.applicable(e, basis=frame)
        assert len(report.changing) < len(report)
        assert all(h.change for h in report.changing)
        assert all(not h.change for h in report if h.reshapes_only)

    def test_it_reads_the_expression_rather_than_reciting_a_menu(self):
        # A cross and a dot must get different answers.
        _, frame, a, b = self._setup()
        dot = {h.step.name for h in ts.applicable(
            tb.expand_in_basis(a @ b, frame), basis=frame).changing}
        cross = {h.step.name for h in ts.applicable(
            tb.expand_in_basis(a % b, frame), basis=frame).changing}
        assert "simplify_basis_cross" in cross
        assert "simplify_basis_cross" not in dot

    def test_each_hit_carries_the_result_it_would_produce(self):
        _, frame, a, b = self._setup()
        e = tb.expand_in_basis(a @ b, frame)
        hit = next(h for h in ts.applicable(e, basis=frame) if h.step.name == "reduce_frame")
        assert td.structural_eq(hit.result, tb.reduce_frame(e, frame))

    def test_steps_needing_absent_context_are_listed_not_silently_dropped(self):
        _, frame, a, b = self._setup()
        report = ts.applicable(tb.expand_in_basis(a @ b, frame), basis=frame)
        assert "coord" in report.missing
        assert "rules" in report.missing
        assert "not tried" in str(report)

    def test_the_report_prints(self):
        _, frame, a, b = self._setup()
        text = str(ts.applicable(tb.expand_in_basis(a @ b, frame), basis=frame))
        assert "reduce_frame" in text and "only reshape" in text


class TestWhyNot:
    def _setup(self):
        ctx = tender.Context()
        frame = tb.wcs(ctx)
        a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
        return frame, a @ b, tb.expand_in_basis(a @ b, frame)

    def test_missing_context_is_named(self):
        frame, inv, exp = self._setup()
        assert "needs coord" in ts.why_not(exp, "partial")

    def test_a_step_that_explains_itself_says_more_than_a_count(self):
        # contract_delta reports in its own terms, and differently for the two
        # cases — which no external fingerprint could distinguish.
        frame, inv, exp = self._setup()
        assert "no summation" in ts.why_not(inv, "contract_delta")
        assert "no δ in this term" in ts.why_not(exp, "contract_delta")

    def test_a_step_without_a_report_falls_back_to_the_count(self):
        # The synthesized reason, for a step not yet taught to explain itself.
        frame, inv, exp = self._setup()
        msg = ts.why_not(inv, "eval_delta_concrete")
        assert "nothing to act on" in msg and "deltas (needs 1, has 0)" in msg

    def test_a_step_that_would_apply_says_so(self):
        frame, inv, exp = self._setup()
        assert "did apply" in ts.why_not(exp, "reduce_frame", basis=frame)

    def test_a_raising_step_reports_what_it_raised(self):
        frame, inv, exp = self._setup()
        # unroll_sums on an invariant has indices to want but nothing to unroll.
        msg = ts.why_not(inv, "reassemble", basis=frame)
        assert "nothing to act on" in msg or "changed nothing" in msg

    def test_it_accepts_a_step_object_too(self):
        frame, inv, exp = self._setup()
        assert ts.why_not(inv, ts.info("contract_delta")) == ts.why_not(
            inv, "contract_delta"
        )


class TestExplain:
    def test_it_states_the_change(self):
        ctx = tender.Context()
        frame = tb.wcs(ctx)
        a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
        e = tb.expand_in_basis(a @ b, frame)
        text = ts.explain(e, tb.reduce_frame(e, frame))
        assert "before" in text and "after" in text
        assert "basis_vectors-2" in text

    def test_identical_expressions_say_so(self):
        ctx = tender.Context()
        a = tender.tensor("a", rank=1, ctx=ctx)
        assert "none" in ts.explain(a, a)

    def test_a_pure_reordering_is_named_as_such(self):
        ctx = tender.Context()
        a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
        assert "reshaped only" in ts.explain(b @ a, td.canonicalize(b @ a))


class TestStepReport:
    """A step's own account of what it did (vibe 000106).

    The fingerprint measures the result from outside; it can say *that* nothing
    changed, never *why*.  The reason lives in the step's logic, so the step
    reports it.
    """

    def _setup(self):
        ctx = tender.Context()
        frame = tb.wcs(ctx)
        a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
        return frame, a, b, c

    def test_a_reporting_step_gives_different_reasons_for_different_inputs(self):
        # The thing no external measure could do: same step, same "nothing
        # happened", two different causes.
        frame, a, b, _ = self._setup()
        bare = ts.info("contract_delta").run(a @ b)
        expanded = ts.info("contract_delta").run(tb.expand_in_basis(a @ b, frame))
        assert not bare.fired and not expanded.fired
        assert bare.reason != expanded.reason

    def test_reduce_frame_distinguishes_not_expanded_from_nothing_further(self):
        frame, a, b, c = self._setup()
        never = ts.info("reduce_frame").run(a @ b, basis=frame)
        assert "expand_in_basis" in never.reason

        stalled = tb.reduce_frame(tb.expand_in_basis(a % (b % c), frame), frame)
        done = ts.info("reduce_frame").run(stalled, basis=frame)
        assert not done.fired
        assert "ε-pair" in done.reason or "cannot justify" in done.reason

    def test_the_return_value_is_normalised_even_when_nothing_fired(self):
        # The separation StepReport buys: `fired` no longer has to be inferred
        # from the return, so a step can normalise its output *and* report
        # honestly that it did no work.
        frame, a, b, c = self._setup()
        stalled = tb.reduce_frame(tb.expand_in_basis(a % (b % c), frame), frame)
        res = ts.info("reduce_frame").run(stalled, basis=frame)
        assert not res.fired
        assert td.algebraic_eq(res.expr, stalled)

    def test_a_step_without_a_report_still_returns_a_result(self):
        # Uniform from the start: callers need not know which steps have been
        # taught to explain themselves.
        frame, a, b, _ = self._setup()
        res = ts.info("fold_sums").run(a @ b)
        assert isinstance(res, ts.StepResult)
        assert not res.fired and res.reason

    def test_missing_context_is_reported_not_raised(self):
        frame, a, b, _ = self._setup()
        res = ts.info("partial").run(a @ b)
        assert not res.fired and "needs coord" in res.reason

    def test_a_firing_step_reports_no_reason(self):
        frame, a, b, _ = self._setup()
        res = ts.info("reduce_frame").run(tb.expand_in_basis(a @ b, frame), basis=frame)
        assert res.fired and not res.reason
        assert td.structural_eq(res.expr, tb.reduce_frame(tb.expand_in_basis(a @ b, frame), frame))
