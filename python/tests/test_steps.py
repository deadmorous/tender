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
            "deriv", "explore", "rule", "derive", "Basis", "Handedness",
            "Variance", "wcs",
            "cylindrical",
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
            "apply_identity", "engine_simplify", "fold_operator",
            "insert_metric", "partial", "saturate",
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

    def test_a_step_that_only_decorates_is_not_offered(self):
        # collect_terms used to write a unit coefficient back — `a × (a × b)`
        # became `1 · (a × (a × b))` — which reads as an option in a list whose
        # whole value is that everything in it does something.
        _, frame, a, b = self._setup()
        offered = {h.step.name for h in ts.applicable(a % (a % b), basis=frame)}
        assert "collect_terms" not in offered

    def test_each_hit_carries_the_result_it_would_produce(self):
        _, frame, a, b = self._setup()
        e = tb.expand_in_basis(a @ b, frame)
        hit = next(h for h in ts.applicable(e, basis=frame) if h.step.name == "reduce_frame")
        assert td.structural_eq(hit.result, tb.reduce_frame(e, frame))

    def test_steps_needing_absent_context_are_listed_not_silently_dropped(self):
        _, frame, a, b = self._setup()
        report = ts.applicable(tb.expand_in_basis(a @ b, frame), basis=frame)
        # Keyed by step, carrying every unmet need — so a chooser can offer
        # "supply this and see" rather than just naming an absent kind.
        assert report.missing["tender.derivation.partial"] == ("coord",)
        assert "tender.derivation.engine_simplify" in report.blocked_on("rules")
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

    def test_a_step_that_has_learned_to_explain_itself_replaces_the_fallback(self):
        # This asserted the generic "nothing to act on" wording until
        # `reassemble` learned to speak; the step's own reason is better and
        # takes precedence, which is the migration working as intended.
        frame, inv, exp = self._setup()
        msg = ts.why_not(inv, "reassemble", basis=frame)
        assert "nothing here in component form" in msg

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


def test_a_reporting_step_must_agree_with_the_fingerprint():
    """A step's claim is checked against an independent measure.

    The report is what the step *says*; the fingerprint is what the expression
    *shows*.  Keeping both is what makes this test possible — and it is not
    hypothetical: `reduce_frame` once reported `fired=True` on a term it had
    only reordered, and that is exactly the disagreement caught here.
    """
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
    A = tender.tensor("A", rank=2, ctx=ctx)

    exprs = [
        a @ b,
        a % b,
        A @ b,
        tb.expand_in_basis(a @ b, frame),
        tb.expand_in_basis(a % b, frame),
        tb.expand_in_basis(A @ b, frame),
        tb.reduce_frame(tb.expand_in_basis(a @ b, frame), frame),
        tb.reduce_frame(tb.expand_in_basis(a % (b % c), frame), frame),
    ]
    reporting = [n for n in ts.names() if ts.info(n).reported is not None]
    assert reporting, "no step reports yet — this test would be vacuous"

    for name in reporting:
        st = ts.info(name)
        for e in exprs:
            res = st.run(e, basis=frame)
            before, after = ts.shape(e), ts.shape(res.expr)
            changed = before != after
            assert res.fired == changed, (
                f"{name} reported fired={res.fired} but the fingerprint "
                f"{'changed' if changed else 'did not change'}: {e.latex()}"
            )
            assert res.fired != bool(res.reason), (
                f"{name} must give a reason exactly when it did not fire"
            )


class TestReassembleExplainsItself:
    """`reassemble` has the richest refusal set in the library (vibe 000106).

    Its declines were already written out as comments — "a rank ≥ 2 invariant
    cannot be placed at a frame vector nested inside a contraction", "the shared
    indices do not sit on one carrier's trailing slots" — so reporting them
    recovers knowledge the code already had rather than inventing any.

    Each case below is a *different* reason, which is the point: a single
    "nothing happened" would have covered all of them.
    """

    def _frame(self):
        ctx = tender.Context()
        return ctx, tb.wcs(ctx)

    def _why(self, e, frame, **kw):
        msg = ts.why_not(e, "reassemble", basis=frame, **kw)
        return msg[len("tender.basis.reassemble: "):]

    def test_nothing_in_component_form(self):
        ctx, f = self._frame()
        a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
        assert "nothing here in component form" in self._why(a @ b, f)

    def test_an_epsilon_pair_belongs_to_a_different_step(self):
        ctx, f = self._frame()
        a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
        stalled = tb.reduce_frame(tb.expand_in_basis(a % (b % c), f), f)
        assert "ε-pair contraction's business" in self._why(stalled, f)

    def test_a_rank2_invariant_at_a_nested_frame_vector(self):
        ctx, f = self._frame()
        A = tender.tensor("A", rank=2, ctx=ctx)
        b = tender.tensor("b", rank=1, ctx=ctx)
        e = td.canonicalize(tb.expand_in_basis(A, f) @ b)
        assert "unfoldable" in self._why(e, f) or "orientation" in self._why(e, f)

    def test_a_target_that_names_nothing_here(self):
        ctx, f = self._frame()
        a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
        comps = tb.reduce_frame(tb.expand_in_basis(a * b, f), f)
        assert "not the one you named" in self._why(comps, f, target="z")

    def test_an_epsilon_needs_an_orthonormal_right_handed_frame(self):
        ctx = tender.Context()
        oblique = tb.make_oblique_basis(
            [tender.tensor(n, rank=1, ctx=ctx) for n in ("p", "q", "s")],
            tender.space_3d,
        )
        a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
        e = tb.reduce_frame(tb.expand_in_basis(a % b, oblique), oblique)
        msg = self._why(e, oblique)
        assert "orthonormal" in msg or "recognises" in msg or "not read" in msg

    def test_the_reasons_actually_differ(self):
        # The whole point: one "nothing happened" would cover all of these.
        ctx, f = self._frame()
        a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
        A = tender.tensor("A", rank=2, ctx=ctx)
        comps = tb.reduce_frame(tb.expand_in_basis(a * b, f), f)
        reasons = {
            self._why(a @ b, f),
            self._why(tb.reduce_frame(tb.expand_in_basis(a % (b % c), f), f), f),
            self._why(comps, f, target="z"),
        }
        assert len(reasons) == 3, reasons


class TestTheRemainingReporters:
    """`contract_metric`, `insert_metric`, `fold_operator`, `apply_operators`.

    With these, every step whose refusal a person is likely to hit explains
    itself.  The ones left silent are the general normalisers (`simplify`,
    `canonicalize`, `simplify_scalars`), where "it ran and changed nothing" is
    the complete answer.
    """

    def _oblique(self):
        ctx = tender.Context()
        frame = tb.make_oblique_basis(
            [tender.tensor(n, rank=1, ctx=ctx) for n in ("p", "q", "s")],
            tender.space_3d,
        )
        a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
        upper = td.canonicalize(
            tb.simplify_basis_dot(
                tb.expand_in_basis(a @ b, frame, tb.Variance.Contravariant), frame
            )
        )
        return ctx, frame, a, b, upper

    def _why(self, e, name, **kw):
        return ts.why_not(e, name, **kw).split(": ", 1)[1]

    def test_contract_metric_says_when_there_is_no_metric(self):
        ctx, frame, a, b, _ = self._oblique()
        assert "no summation for a metric" in self._why(a @ b, "contract_metric")

    def test_contract_metric_says_when_the_target_is_the_other_factor(self):
        _, _, _, _, upper = self._oblique()
        msg = self._why(upper, "contract_metric", target="c")
        assert "different factor than the one you named" in msg

    def test_insert_metric_says_when_nothing_sits_at_the_wrong_level(self):
        _, _, _, _, upper = self._oblique()
        mixed = td.contract_metric(upper, target="a")  # a^i b_i
        # `a` is already upper, so asking to raise it moves nothing.
        msg = self._why(mixed, "insert_metric", level=tender.Level.Upper, target="a")
        assert "level opposite the one asked for" in msg

    def test_apply_operators_distinguishes_its_two_silences(self):
        ws = tender.Workspace()
        cart, _ = ws.cartesian_chart()
        f = ws.field("f", 0)
        abstract = self._why(ws.nabla() * f, "apply_operators")
        assert "abstract ∇" in abstract and "expand_nabla" in abstract
        none_at_all = self._why(f, "apply_operators")
        assert "no unapplied ∂" in none_at_all
        assert abstract != none_at_all

    def test_fold_operator_rejects_a_thing_that_is_not_an_operator(self):
        ws = tender.Workspace()
        cart, _ = ws.cartesian_chart()
        f = ws.field("f", 0)
        assert "not of the form" in self._why(f, "fold_operator", op=f)

    def test_fold_operator_says_when_the_group_is_incomplete(self):
        ws = tender.Workspace()
        cart, (x, y, z) = ws.cartesian_chart()
        e = cart.physical_frame()
        f = cart.field("f", 0)
        op = e.direction(0) * td.deriv(x) + e.direction(1) * td.deriv(y)
        partial = td.simplify_scalars(
            td.apply_operators(e.direction(0) * td.deriv(x) * f)
        )
        msg = self._why(partial, "fold_operator", op=op)
        assert "no complete group" in msg
        assert "never there" in msg  # why an incomplete fold would be wrong

    def test_every_reporting_step_still_agrees_with_the_fingerprint(self):
        # The invariant from before, re-run over the wider set: a step's claim
        # is checked against an independent measure.
        ctx = tender.Context()
        frame = tb.wcs(ctx)
        a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
        exprs = [
            a @ b,
            tb.expand_in_basis(a @ b, frame),
            tb.reduce_frame(tb.expand_in_basis(a % (b % c), frame), frame),
        ]
        reporting = [n for n in ts.names() if ts.info(n).reported is not None]
        assert len(reporting) >= 7
        for name in reporting:
            st = ts.info(name)
            for e in exprs:
                res = st.run(e, basis=frame, level=tender.Level.Upper)
                changed = ts.shape(e) != ts.shape(res.expr)
                assert res.fired == changed, (name, e.latex())
                assert res.fired != bool(res.reason), name
