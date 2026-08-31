"""The derivation session and its widget (vibe 000108).

The session is the library half of the interactive surface: it fills the extra
arguments from the user's namespace, records the whole tree while showing one
path through it, and emits code that leans on the preamble.  These tests hold
those three promises; the widget tests check that the surface is assembled from
the session and stays in step with it, without a browser.
"""

import pytest

import tender
import tender.basis as tb
import tender.chart as tc
import tender.derivation as td
import tender.explore as tx
import tender.identities as ti
import tender.steps as ts


def _setup():
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
    return ctx, frame, a, b


def _cyl_chart(ctx):
    cart = tb.wcs(ctx)
    r = tender.coordinate("r", chart_id=1, slot=0, nonneg=True, ctx=ctx)
    th = tender.coordinate("\\theta", chart_id=1, slot=1, ctx=ctx)
    z = tender.coordinate("z", chart_id=1, slot=2, ctx=ctx)
    return tc.CoordinateChart(
        cart, [r, th, z], [r * tender.cos(th), r * tender.sin(th), z]
    )


class TestScopeScanning:
    def test_a_basis_in_scope_is_found_under_its_own_name(self):
        _, frame, a, b = _setup()
        found = tx.scan_scope({"frame": frame, "junk": 3})
        assert [x.name for x in found["basis"]] == ["frame"]
        assert found["basis"][0].value is frame

    def test_a_coordinate_is_told_from_a_scalar(self):
        # The distinction the `coord` steps care about, and one no isinstance
        # can make: a coordinate is what `partial` will differentiate by.
        ctx = tender.Context()
        r = tender.coordinate("r", ctx=ctx)
        s = tender.tensor("s", rank=0, ctx=ctx)
        found = tx.scan_scope({"r": r, "s": s})
        assert [x.name for x in found["coord"]] == ["r"]

    def test_a_charts_coordinates_are_offered_under_their_path(self):
        # A notebook holds the chart, not the loose coordinates.
        ctx = tender.Context()
        chart = _cyl_chart(ctx)
        found = tx.scan_scope({"chart": chart})
        assert [x.name for x in found["coord"]] == [
            "chart.coords[0]", "chart.coords[1]", "chart.coords[2]",
        ]

    def test_a_chart_is_found_by_kind_as_well_as_for_its_coordinates(self):
        # The gap of vibe 000108 §14: a chart was scanned only to harvest its
        # coordinates, while the steps that take the chart itself were not in
        # the catalogue at all.
        ctx = tender.Context()
        chart = _cyl_chart(ctx)
        found = tx.scan_scope({"cart": chart})
        assert [x.name for x in found["chart"]] == ["cart"]
        assert found["chart"][0].value is chart

    def test_rules_and_levels_are_recognised_by_kind(self):
        ctx = tender.Context()
        rules = td.rules("eps_delta", ctx=ctx)
        found = tx.scan_scope({"rr": rules, "up": tender.Level.Upper})
        assert [x.name for x in found["rules"]] == ["rr"]
        assert [x.name for x in found["level"]] == ["up"]

    def test_private_names_are_ignored(self):
        _, frame, a, b = _setup()
        assert tx.scan_scope({"_frame": frame})["basis"] == []


class TestContextFilling:
    def test_the_basis_reaches_every_step_that_wants_one(self):
        # The click budget in one assertion: the frame is named once, in the
        # cell, and no call repeats it.
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, scope={"frame": frame})
        s.apply("expand_in_basis").apply("reduce_frame")
        assert td.structural_eq(
            s.current, tb.reduce_frame(tb.expand_in_basis(a @ b, frame), frame)
        )

    def test_use_picks_between_several_candidates(self):
        ctx, frame, a, b = _setup()
        other = tb.cylindrical(ctx)
        s = tx.Session(a @ b, scope={"frame": frame, "cyl": other})
        assert len(s.bindings["basis"]) == 2
        s.use("basis", other)
        assert s.context["basis"].name == "cyl"
        assert s.values["basis"] is other

    def test_needs_may_be_given_explicitly_instead_of_searched_for(self):
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, needs={"basis": frame})
        assert s.values["basis"] is frame
        assert not s.scanned

    def test_an_explicit_dict_excludes_the_search(self):
        # Explicit means explicit: `frame` is in scope and of the right kind,
        # and it still does not appear beside what the caller named.
        ctx, frame, a, b = _setup()
        other = tb.cylindrical(ctx)
        s = tx.Session(a @ b, needs={"basis": other}, scope={"frame": frame})
        assert [x.value for x in s.bindings["basis"]] == [other]

    def test_an_empty_dict_is_a_session_with_nothing_supplied(self):
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, needs={}, scope={"frame": frame})
        assert s.values == {}
        assert "tender.basis.reduce_frame" in s.applicable().blocked_on("basis")

    def test_a_keyword_wins_over_both(self):
        ctx, frame, a, b = _setup()
        other = tb.cylindrical(ctx)
        s = tx.Session(
            a @ b, needs={"basis": frame}, context={"basis": other},
            scope={"frame": frame},
        )
        assert s.values["basis"] is other

    def test_an_unknown_kind_in_the_dict_is_refused(self):
        ctx, frame, a, b = _setup()
        with pytest.raises(ValueError, match="unknown kind"):
            tx.Session(a @ b, needs={"frame": frame})

    def test_an_unknown_kind_is_refused(self):
        ctx, frame, a, b = _setup()
        with pytest.raises(ValueError, match="unknown kind"):
            tx.Session(a @ b).use("nonsense", frame)

    def test_a_step_still_missing_its_argument_is_reported_not_tried(self):
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, scope={"frame": frame})
        assert s.applicable().missing["tender.derivation.partial"] == ("coord",)


class TestChartSteps:
    """A ∇ derivation starts on the chart, so the chooser must reach it."""

    def _nabla_session(self, **kw):
        ws = tender.Workspace()
        cart, _ = ws.cartesian_chart()
        u = tender.field("u", 1, ctx=ws.ctx)
        nabla = tender.nabla(ws.ctx)
        return ws, cart, tx.Session(nabla @ (nabla * u), **kw, scope={
            "ws": ws, "cart": cart,
        })

    def test_the_chart_moves_are_offered(self):
        ws, cart, s = self._nabla_session()
        got = {h.step.name for h in s.applicable().changing}
        assert {"expand_nabla", "evaluate", "expand"} <= got

    def test_without_a_chart_they_are_not_tried_rather_than_absent(self):
        # The failure that prompted this: nothing at all was shown, so absence
        # was indistinguishable from inapplicability.
        ws = tender.Workspace()
        u = tender.field("u", 1, ctx=ws.ctx)
        nabla = tender.nabla(ws.ctx)
        s = tx.Session(nabla @ (nabla * u), needs={})
        assert "tender.chart.expand_nabla" in s.applicable().blocked_on("chart")

    def test_taking_one_emits_a_chart_bound_script(self):
        ws, cart, s = self._nabla_session()
        s.apply("expand_nabla")
        assert "b = ts.using(chart=cart)" in s.script()
        assert "b.expand_nabla," in s.script()

    def test_the_function_form_is_the_method(self):
        ws, cart, s = self._nabla_session()
        import tender.chart as tc

        assert td.structural_eq(
            tc.expand_nabla(s.current, cart), cart.expand_nabla(s.current)
        )


class TestThePath:
    def test_a_step_that_does_nothing_is_refused_with_its_own_reason(self):
        # The history records moves, not attempts; and the reason is the step's.
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, scope={"frame": frame})
        with pytest.raises(tx.DidNotFire, match="contract_delta"):
            s.apply("contract_delta")
        assert len(s.path) == 1

    def test_going_back_keeps_the_branch(self):
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, scope={"frame": frame})
        s.apply("expand_in_basis").apply("reduce_frame").back()
        assert len(s.path) == 2
        assert len(s.path[-1].children) == 1  # the abandoned tail is still there
        assert "reduce_frame" in s.attempts()

    def test_a_step_tried_before_is_remembered_by_the_expression(self):
        # Keyed on the canonical form, so returning by another route still
        # says "you have been here" (vibe 000108 §7).
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, scope={"frame": frame})
        s.apply("expand_in_basis").apply("reduce_frame").back()
        assert "reduce_frame" in s.tried()
        assert "reduce_frame" not in s.tried(s.root.expr)

    def test_goto_truncates_the_shown_path(self):
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, scope={"frame": frame})
        s.apply("expand_in_basis").apply("reduce_frame").goto(0)
        assert td.structural_eq(s.current, a @ b)
        assert len(s.root.children) == 1

    def test_steps_are_available_as_data_for_replay(self):
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, scope={"frame": frame})
        s.apply("expand_in_basis").apply("reduce_frame")
        assert [name for name, _ in s.steps] == ["expand_in_basis", "reduce_frame"]
        e = a @ b
        for name, kwargs in s.steps:
            e = ts.info(name)(e, **kwargs)
        assert td.structural_eq(e, s.current)


class TestScript:
    def test_it_emits_the_derivation_as_a_list_of_steps(self):
        # A derivation is a list, not a chain of assignments: the shared
        # argument is bound once and the list is data you can edit.
        ctx, frame, a, b = _setup()
        e0 = a @ b
        s = tx.Session(e0, scope={"frame": frame, "e0": e0, "td": td, "ts": ts})
        s.apply("expand_in_basis").apply("reduce_frame")
        assert s.script() == (
            "b = ts.using(basis=frame)\n"
            "e = td.derive(e0, [\n"
            "    b.expand_in_basis,\n"
            "    b.reduce_frame,\n"
            "]).current"
        )

    def test_it_imports_what_your_namespace_does_not_have(self):
        # A default alias is a guess about a name in the namespace; when the
        # guess is wrong the pasted script fails on its first line.
        ctx, frame, a, b = _setup()
        e0 = a @ b
        s = tx.Session(e0, scope={"frame": frame, "e0": e0})
        s.apply("expand_in_basis")
        assert s.script().startswith(
            "import tender.derivation as td\nimport tender.steps as ts\n\n"
        )
        # …and says nothing when you already have them.
        held = tx.Session(
            e0, scope={"frame": frame, "e0": e0, "td": td, "ts": ts}
        )
        held.apply("expand_in_basis")
        assert held.script().startswith("b = ts.using(")

    def test_the_assignment_chain_is_still_available(self):
        ctx, frame, a, b = _setup()
        e0 = a @ b
        s = tx.Session(e0, scope={"frame": frame, "e0": e0, "tb": tb, "td": td})
        s.apply("expand_in_basis").apply("reduce_frame")
        assert s.script(style="assign") == (
            "e = e0\n"
            "e = tb.expand_in_basis(e, frame)\n"
            "e = tb.reduce_frame(e, frame)"
        )

    def test_an_unknown_style_is_refused(self):
        ctx, frame, a, b = _setup()
        with pytest.raises(ValueError, match="unknown style"):
            tx.Session(a @ b, needs={}).script(style="prose")

    def test_a_step_with_no_arguments_needs_no_binder(self):
        ctx, frame, a, b = _setup()
        s = tx.Session(a + a, needs={}, name="e0", scope={"td": td})
        s.apply("fold_equal_addends")
        assert s.script() == (
            "e = td.derive(e0, [\n    td.fold_equal_addends,\n]).current"
        )

    def test_a_per_step_argument_stays_per_step(self):
        # The binder holds what every step shares; a target is not that.
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, scope={"frame": frame}, name="e0")
        s.apply("expand_in_basis", variance=tb.Variance.Contravariant)
        assert "b('expand_in_basis', variance=" in s.script()

    def test_it_uses_the_aliases_the_namespace_uses(self):
        ctx, frame, a, b = _setup()
        import tender.basis as basis_module

        s = tx.Session(a @ b, scope={"frame": frame, "basis": basis_module})
        s.apply("expand_in_basis")
        assert "basis.expand_in_basis(e, frame)" in s.script(style="assign")

    def test_an_option_is_emitted_as_a_keyword(self):
        # And an enum is written the way a person writes it, not as its repr.
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, scope={"frame": frame, "tb": tb})
        s.apply("expand_in_basis", variance=tb.Variance.Contravariant)
        assert (
            "tb.expand_in_basis(e, frame, variance=tb.Variance.Contravariant)"
            in s.script(style="assign")
        )

    def test_the_emitted_script_runs_and_reproduces_the_derivation(self):
        # The promise the panel makes: paste it, and it is your derivation.
        ctx, frame, a, b = _setup()
        e0 = a @ b
        s = tx.Session(e0, scope={"frame": frame, "e0": e0})
        s.apply("expand_in_basis").apply("reduce_frame")
        env = {"tb": tb, "td": td, "ts": ts, "frame": frame, "e0": e0}
        exec(s.script(), env)
        assert td.structural_eq(env["e"], s.current)
        exec(s.script(style="assign"), env)
        assert td.structural_eq(env["e"], s.current)


class TestEntryPoint:
    def test_explore_takes_its_context_from_the_calling_scope(self):
        ctx = tender.Context()
        frame = tb.wcs(ctx)
        a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
        s = td.explore(a @ b, gui=False)
        assert s.values["basis"] is frame

    def test_an_explicit_context_overrides_what_was_found(self):
        ctx, frame, a, b = _setup()
        other = tb.cylindrical(ctx)
        s = td.explore(a @ b, scope={"frame": frame}, gui=False, basis=other)
        assert s.values["basis"] is other

    def test_it_is_a_session_outside_a_notebook(self):
        # No frontend, no widget: the session is the object.
        ctx, frame, a, b = _setup()
        assert isinstance(td.explore(a @ b, scope={}, gui=None), tx.Session)

    def test_a_terminal_is_told_why_there_is_no_widget(self):
        # A widget needs a browser to draw in, so a terminal gets the session
        # alone — but silence there looks like a failure, and it is not one.
        ctx, frame, a, b = _setup()
        import io
        import contextlib

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            td.explore(a @ b, scope={})
        said = out.getvalue()
        assert "no derivation widget" in said and "s.applicable()" in said


class TestToCell:
    """The session is scratch; the code is the artifact (vibe 000108 §12)."""

    def _session(self):
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, needs={"basis": frame}, name="e0")
        return s.apply("expand_in_basis")

    def test_it_hands_the_script_to_the_frontend(self):
        s = self._session()
        seen = {}

        class Shell:
            def set_next_input(self, text, replace=False):
                seen["text"], seen["replace"] = text, replace

        real, tx._shell = tx._shell, lambda: Shell()
        try:
            s.to_cell()
        finally:
            tx._shell = real
        assert seen["text"] == s.script()
        assert seen["replace"] is False

    def test_outside_a_kernel_it_prints_the_code(self):
        # The same thing one copy away, rather than a silent no-op.
        import contextlib
        import io

        s = self._session()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            s.to_cell()
        assert out.getvalue().strip() == s.script()


class TestWidget:
    """The surface, driven without a browser: a widget is a Python object.

    ``ipywidgets`` is optional — everything above works without it — so these
    skip rather than fail where it is absent.
    """

    def _widget(self):
        ctx, frame, a, b = _setup()
        gui = pytest.importorskip("tender.gui")

        s = tx.Session(a @ b, scope={"frame": frame})
        return s, gui.build(s)

    def test_it_shows_one_item_per_point_on_the_path(self):
        s, w = self._widget()
        assert len(w.items) == 1
        s.apply("expand_in_basis")
        w.refresh()
        assert len(w.items) == 2

    def test_the_chooser_offers_what_applies_here(self):
        s, w = self._widget()
        offered = {v for _, v in w.items[0].chooser.options if v}
        # …plus the entry that asks a question rather than being a probed move.
        assert offered == {h.step.name for h in s.applicable()} | {"apply_identity"}

    def test_choosing_a_step_advances_the_session(self):
        s, w = self._widget()
        w.items[0].chooser.value = "expand_in_basis"
        assert [name for name, _ in s.steps] == ["expand_in_basis"]
        assert len(w.items) == 2

    def test_choosing_at_an_earlier_item_drops_the_tail(self):
        s, w = self._widget()
        s.apply("expand_in_basis").apply("reduce_frame")
        w.refresh()
        w.items[0].chooser.value = "skew"
        assert [name for name, _ in s.steps] == ["skew"]
        # and the abandoned branch is still recorded
        assert "reduce_frame" in s.attempts()

    def test_a_tried_step_is_marked_in_the_chooser(self):
        s, w = self._widget()
        s.apply("expand_in_basis").apply("reduce_frame").back()
        w.refresh()
        label = next(
            lb for lb, v in w.items[1].chooser.options if v == "reduce_frame"
        )
        assert label.endswith("· tried")

    def test_steps_blocked_for_want_of_an_argument_say_what_is_missing(self):
        s, w = self._widget()
        assert "partial (needs coord)" in w.items[0].note.value

    def test_the_code_panel_tracks_the_session(self):
        s, w = self._widget()
        s.apply("expand_in_basis")
        w.refresh()
        assert w.code.value == s.script()

    def test_a_domain_error_becomes_a_message_rather_than_a_traceback(self):
        s, w = self._widget()
        w._advance(0, "partial", "")
        assert "not tried" in w.status.value or "needs coord" in w.status.value
        assert s.steps == []

    def test_the_target_field_is_hidden_until_asked_for(self):
        # The click budget: a mandatory field costs a click on every step.
        s, w = self._widget()
        item = w.items[0]
        assert item.target.layout.display == "none"
        item.reveal.value = True
        assert item.target.layout.display == ""

    def test_the_whole_catalogue_is_in_view_for_why_not(self):
        s, w = self._widget()
        assert "reduce_frame" in w.box.children[3].children[0].value

    def test_the_items_keep_their_size_as_the_list_grows(self):
        # A flex child shrinks below its content by default, which squeezed a
        # long history until each item hid its own chooser behind a scrollbar.
        s, w = self._widget()
        s.apply("expand_in_basis").apply("reduce_frame")
        w.refresh()
        assert all(i.box.layout.flex == "0 0 auto" for i in w.items)
        assert w.history.layout.overflow == "auto"

    def test_the_list_may_be_made_taller(self):
        import tender.gui as gui

        s, _ = self._widget()
        assert gui.build(s, max_height="900px").history.layout.max_height == "900px"

class TestFilter:
    """Typing narrows all three categories at once, not just the chooser."""

    def _widget(self, expr=None):
        ctx, frame, a, b = _setup()
        gui = pytest.importorskip("tender.gui")
        s = tx.Session(expr if expr is not None else a @ b, needs={"basis": frame})
        return s, gui.build(s)

    def test_it_narrows_the_chooser(self):
        s, w = self._widget()
        item = w.items[0]
        before = {v for _, v in item.chooser.options if v}
        item.filter.value = "basis_"
        after = {v for _, v in item.chooser.options if v}
        assert after < before
        assert all("basis" in n for n in after)

    def test_it_matches_the_module_tail_too(self):
        # The chooser shows `chart.expand`, so the filter matches that — one
        # pattern reaches a whole module's worth of steps.
        ws = tender.Workspace()
        cart, _ = ws.cartesian_chart()
        u = tender.field("u", 1, ctx=ws.ctx)
        nabla = tender.nabla(ws.ctx)
        gui = pytest.importorskip("tender.gui")
        s = tx.Session(nabla @ (nabla * u), needs={"chart": cart})
        item = gui.build(s).items[0]
        item.filter.value = "^chart[.]"
        offered = [lb for lb, v in item.chooser.options if v]
        assert offered and all(lb.startswith("chart.") for lb in offered)

    def test_a_regex_is_a_regex(self):
        s, w = self._widget()
        item = w.items[0]
        item.filter.value = "^expand_"
        assert all(v.startswith("expand_") for _, v in item.chooser.options if v)

    def test_a_half_typed_regex_falls_back_to_a_substring(self):
        # A text box passes through invalid patterns on the way to valid ones;
        # that is not an error to shout about.
        s, w = self._widget()
        item = w.items[0]
        item.filter.value = "expand_("
        assert item.filter.layout.border == "1px solid #a00"
        assert [v for _, v in item.chooser.options if v] == []

    def test_it_reaches_the_steps_that_were_not_tried(self):
        s, w = self._widget()
        item = w.items[0]
        item.filter.value = "partial"
        assert "partial (needs coord)" in item.note.value

    def test_a_single_quiet_step_is_answered_rather_than_listed(self):
        # Typing a name that is not in the list *is* the "why not?" question.
        s, w = self._widget()
        item = w.items[0]
        item.filter.value = "contract_delta"
        assert "no summation" in item.note.value

    def test_several_quiet_steps_are_named(self):
        s, w = self._widget()
        item = w.items[0]
        item.filter.value = "eval_"
        assert "did not fire: " in item.note.value
        assert "eval_delta_concrete" in item.note.value

    def test_a_pattern_matching_nothing_says_so(self):
        s, w = self._widget()
        item = w.items[0]
        item.filter.value = "zzz"
        assert "no step matches that" in item.note.value

    def test_clearing_the_filter_restores_everything(self):
        s, w = self._widget()
        item = w.items[0]
        before = {v for _, v in item.chooser.options if v}
        item.filter.value = "basis"
        item.filter.value = ""
        assert {v for _, v in item.chooser.options if v} == before
        assert item.filter.layout.border == ""

    def test_filtering_does_not_take_a_step(self):
        # Rewriting the options changes the dropdown's value; that must not
        # read as a choice.
        s, w = self._widget()
        w.items[0].filter.value = "reduce"
        assert s.steps == []

    def test_it_asks_about_the_item_it_is_typed_in(self):
        # Filtering above the working end must not answer about the end.
        ctx, frame, a, b = _setup()
        gui = pytest.importorskip("tender.gui")
        s = tx.Session(a @ b, needs={"basis": frame})
        s.apply("expand_in_basis")
        w = gui.build(s)
        w.items[0].filter.value = "contract_delta"
        assert "no summation" in w.items[0].note.value
        w.items[1].filter.value = "contract_delta"
        assert "no δ in this term" in w.items[1].note.value

    def test_the_step_already_taken_survives_the_filter(self):
        # Its value has to stay selectable, or the dropdown holds one that is
        # not among its options.
        s, w = self._widget()
        w.items[0].chooser.value = "expand_in_basis"
        w.items[0].filter.value = "zzz"
        assert w.items[0].chooser.value == "expand_in_basis"
        assert s.steps == [("expand_in_basis", {"basis": s.values["basis"]})]


class TestIdentities:
    """Choosing a rule is a question, not a probed move (vibe 000108 §11)."""

    def _setup_cross(self):
        ws = tender.Workspace()
        ctx = ws.ctx
        a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
        return ws, a % (b % c)

    def test_the_library_is_a_step_set_like_any_other(self):
        # The terminal half: "which identities apply here?" is `applicable`
        # pointed at a different set of steps.
        ws, e = self._setup_cross()
        rs = td.rules("cross", ctx=ws.ctx)
        got = {h.step.name for h in ts.applicable(e, steps=ts.rule_steps(rs))}
        assert got == {"bac-cab"}

    def test_a_rule_that_does_not_match_says_which_pattern_looked(self):
        ws, e = self._setup_cross()
        a = tender.tensor("a", rank=1, ctx=ws.ctx)
        step = ts.rule_steps(td.rules("cross", ctx=ws.ctx))[0]
        why = ts.why_not(a @ a, step)
        assert "did not match" in why and "times" in why or "\\times" in why

    def test_the_session_offers_the_library_from_a_context(self):
        ws, e = self._setup_cross()
        s = tx.Session(e, scope={"ws": ws})
        assert len(s.identities) == len(ti.names())
        assert s.identity_source.name == "ws.ctx"

    def test_a_rules_list_overrides_the_library(self):
        ws, e = self._setup_cross()
        one = [td.rule("bac-cab", ws.ctx)]
        s = tx.Session(e, needs={"ctx": ws.ctx, "rules": one})
        assert [r.name for r in s.identities] == ["bac-cab"]
        assert s.identity_source.kind == "rules"

    def test_with_neither_there_is_nothing_to_choose_from(self):
        # Identities are context-bound and an expression does not carry its
        # context, so this is a real absence rather than an oversight.
        ws, e = self._setup_cross()
        assert tx.Session(e, needs={}).identities == []

    def test_the_session_asks_the_library_the_same_way(self):
        # `applicable` over a different step set, reachable from the session —
        # which is where a user with a session in hand goes looking.
        ws, e = self._setup_cross()
        s = tx.Session(e, scope={"ws": ws})
        got = {h.step.name for h in s.applicable(steps=s.rule_steps())}
        assert got == {"bac-cab"}

    def test_applying_one_records_it_as_an_argument(self):
        ws, e = self._setup_cross()
        s = tx.Session(e, needs={"ctx": ws.ctx})
        s.apply("apply_identity", identity=td.rule("bac-cab", ws.ctx))
        name, kwargs = s.steps[0]
        assert name == "apply_identity"
        assert kwargs["identity"].name == "bac-cab"

    def test_the_script_cites_the_rule_by_name(self):
        ws, e = self._setup_cross()
        s = tx.Session(e, scope={"ws": ws}, name="e0")
        s.apply("apply_identity", identity=td.rule("bac-cab", ws.ctx))
        assert "td.rule('bac-cab', ws.ctx)" in s.script()
        assert (
            "td.apply_identity(e, td.rule('bac-cab', ws.ctx))"
            in s.script(style="assign")
        )

    def test_the_emitted_script_runs(self):
        ws, e = self._setup_cross()
        s = tx.Session(e, scope={"ws": ws, "e0": e})
        s.apply("apply_identity", identity=td.rule("bac-cab", ws.ctx))
        env = {"td": td, "ts": ts, "ws": ws, "e0": e}
        exec(s.script(), env)
        assert td.structural_eq(env["e"], s.current)


class TestIdentityChooser:
    def _widget(self):
        gui = pytest.importorskip("tender.gui")
        ws = tender.Workspace()
        a, b, c = (tender.tensor(n, rank=1, ctx=ws.ctx) for n in "abc")
        s = tx.Session(a % (b % c), scope={"ws": ws})
        return s, gui.build(s)

    def test_it_is_offered_but_not_probed(self):
        # It is in the list because it is a move you can make, not because it
        # was tried — that is what the delimiter above it says.
        s, w = self._widget()
        labels = [lb for lb, _ in w.items[0].chooser.options]
        assert any(lb.startswith("derivation.apply_identity") for lb in labels)
        assert any(set(lb) == {"─"} for lb in labels)
        assert w.items[0].rule_row.layout.display == "none"

    def test_choosing_it_opens_the_rules_and_takes_no_step(self):
        s, w = self._widget()
        w.items[0].chooser.value = "apply_identity"
        assert w.items[0].rule_row.layout.display == ""
        assert s.steps == []

    def test_the_rules_are_annotated_like_steps(self):
        s, w = self._widget()
        w.items[0].chooser.value = "apply_identity"
        offered = {r.name for _, r in w.items[0].rules.options if r}
        assert offered == {"bac-cab"}
        label = next(lb for lb, r in w.items[0].rules.options if r)
        assert "nodes+" in label

    def test_the_rules_that_missed_are_named(self):
        s, w = self._widget()
        w.items[0].chooser.value = "apply_identity"
        note = w.items[0].rule_note.value
        assert "did not match" in note
        assert "bac-cab" not in note  # it did match

    def test_choosing_a_rule_takes_the_step(self):
        s, w = self._widget()
        w.items[0].chooser.value = "apply_identity"
        w.items[0].rules.value = next(r for _, r in w.items[0].rules.options if r)
        assert [n for n, _ in s.steps] == ["apply_identity"]
        assert "bac-cab" in s.script()

    def test_a_taken_rule_is_shown_where_it_was_chosen(self):
        # Reading a derivation back: the rule is on the item, not only in the
        # emitted code.
        s, w = self._widget()
        s.apply("apply_identity", identity=td.rule("bac-cab", s.values["ctx"]))
        w.refresh()
        assert w.items[0].rule_row.layout.display == ""
        assert w.items[0].rules.value.name == "bac-cab"

    def test_the_filter_reaches_the_rules_too(self):
        s, w = self._widget()
        item = w.items[0]
        item.chooser.value = "apply_identity"
        item.filter.value = "zzz"
        assert [r for _, r in item.rules.options if r] == []
        item.filter.value = "bac"
        assert [r.name for _, r in item.rules.options if r] == ["bac-cab"]

    def test_with_no_context_it_says_so_rather_than_sitting_empty(self):
        gui = pytest.importorskip("tender.gui")
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, needs={"basis": frame})
        w = gui.build(s)
        w.items[0].chooser.value = "apply_identity"
        assert "no identities to choose from" in w.items[0].rule_note.value

    def test_it_is_not_listed_as_a_step_that_was_not_tried(self):
        # It is not blocked for want of an argument; it asks for one.
        s, w = self._widget()
        assert "apply_identity" not in w.items[0].note.value


class TestContextRow:
    def _widget(self):
        ctx, frame, a, b = _setup()
        gui = pytest.importorskip("tender.gui")
        s = tx.Session(a @ b, scope={"frame": frame})
        return s, gui.build(s)

    def test_the_pairs_are_delimited(self):
        s, w = self._widget()
        row = w.box.children[0].children[0]
        assert any(",&nbsp;" in c.value for c in row.children if hasattr(c, "value"))

    def test_the_row_says_which_way_the_arguments_arrived(self):
        import tender.gui as gui

        ctx, frame, a, b = _setup()
        scanned = gui.build(tx.Session(a @ b, scope={"frame": frame}))
        given = gui.build(tx.Session(a @ b, needs={"basis": frame}))
        assert "your namespace" in scanned.box.children[0].children[1].value
        assert "as given" in given.box.children[0].children[1].value
        assert "not given" in "".join(
            c.value for c in given.box.children[0].children[0].children
        )

    def test_every_kind_is_named_including_the_empty_ones(self):
        # "How do I give a step what it needs?" is answered where it is asked:
        # a kind with nothing in scope is shown as such, not omitted.
        s, w = self._widget()
        row = "".join(c.value for c in w.box.children[0].children[0].children)
        assert "basis</code> = frame" in row
        for kind in ("coord", "level", "op", "rules"):
            assert f"{kind}</code>" in row and "none in scope" in row
        assert "td.explore(expr, " in w.box.children[0].children[1].value
