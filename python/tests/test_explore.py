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

    def test_an_unknown_kind_is_refused(self):
        ctx, frame, a, b = _setup()
        with pytest.raises(ValueError, match="unknown kind"):
            tx.Session(a @ b).use("nonsense", frame)

    def test_a_step_still_missing_its_argument_is_reported_not_tried(self):
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, scope={"frame": frame})
        assert s.applicable().missing["tender.derivation.partial"] == ("coord",)


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
    def test_it_emits_code_that_leans_on_the_preamble(self):
        ctx, frame, a, b = _setup()
        e0 = a @ b
        s = tx.Session(e0, scope={"frame": frame, "e0": e0, "tb": tb, "td": td})
        s.apply("expand_in_basis").apply("reduce_frame")
        assert s.script() == (
            "e = e0\n"
            "e = tb.expand_in_basis(e, frame)\n"
            "e = tb.reduce_frame(e, frame)"
        )

    def test_it_uses_the_aliases_the_namespace_uses(self):
        ctx, frame, a, b = _setup()
        import tender.basis as basis_module

        s = tx.Session(a @ b, scope={"frame": frame, "basis": basis_module})
        s.apply("expand_in_basis")
        assert "basis.expand_in_basis(e, frame)" in s.script()

    def test_an_option_is_emitted_as_a_keyword(self):
        # And an enum is written the way a person writes it, not as its repr.
        ctx, frame, a, b = _setup()
        s = tx.Session(a @ b, scope={"frame": frame, "tb": tb})
        s.apply("expand_in_basis", variance=tb.Variance.Contravariant)
        assert (
            "tb.expand_in_basis(e, frame, variance=tb.Variance.Contravariant)"
            in s.script()
        )

    def test_the_emitted_script_runs_and_reproduces_the_derivation(self):
        # The promise the panel makes: paste it, and it is your derivation.
        ctx, frame, a, b = _setup()
        e0 = a @ b
        s = tx.Session(e0, scope={"frame": frame, "e0": e0})
        s.apply("expand_in_basis").apply("reduce_frame")
        env = {"tb": tb, "td": td, "frame": frame, "e0": e0}
        exec(s.script(), env)
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
        assert len(w.history.children) == 1
        s.apply("expand_in_basis")
        w.refresh()
        assert len(w.history.children) == 2

    def test_the_chooser_offers_what_applies_here(self):
        s, w = self._widget()
        chooser = w.history.children[0].children[1].children[0]
        offered = {value for _, value in chooser.options if value}
        assert offered == {h.step.name for h in s.applicable()}

    def test_choosing_a_step_advances_the_session(self):
        s, w = self._widget()
        chooser = w.history.children[0].children[1].children[0]
        chooser.value = "expand_in_basis"
        assert [name for name, _ in s.steps] == ["expand_in_basis"]
        assert len(w.history.children) == 2

    def test_choosing_at_an_earlier_item_drops_the_tail(self):
        s, w = self._widget()
        s.apply("expand_in_basis").apply("reduce_frame")
        w.refresh()
        chooser = w.history.children[0].children[1].children[0]
        chooser.value = "skew"
        assert [name for name, _ in s.steps] == ["skew"]
        # and the abandoned branch is still recorded
        assert "reduce_frame" in s.attempts()

    def test_a_tried_step_is_marked_in_the_chooser(self):
        s, w = self._widget()
        s.apply("expand_in_basis").apply("reduce_frame").back()
        w.refresh()
        chooser = w.history.children[1].children[1].children[0]
        label = next(lb for lb, v in chooser.options if v == "reduce_frame")
        assert label.endswith("· tried")

    def test_steps_blocked_for_want_of_an_argument_say_what_is_missing(self):
        s, w = self._widget()
        note = w.history.children[0].children[2].value
        assert "partial (needs coord)" in note

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
        row = w.history.children[0].children[1]
        reveal, target = row.children[1], row.children[2]
        assert target.layout.display == "none"
        reveal.value = True
        assert target.layout.display == ""

    def test_the_whole_catalogue_is_in_view_for_why_not(self):
        s, w = self._widget()
        assert "reduce_frame" in w.box.children[3].children[0].value

    def test_the_items_keep_their_size_as_the_list_grows(self):
        # A flex child shrinks below its content by default, which squeezed a
        # long history until each item hid its own chooser behind a scrollbar.
        s, w = self._widget()
        s.apply("expand_in_basis").apply("reduce_frame")
        w.refresh()
        assert all(item.layout.flex == "0 0 auto" for item in w.history.children)
        assert w.history.layout.overflow == "auto"

    def test_the_list_may_be_made_taller(self):
        import tender.gui as gui

        s, _ = self._widget()
        assert gui.build(s, max_height="900px").history.layout.max_height == "900px"

    def test_every_kind_is_named_including_the_empty_ones(self):
        # "How do I give a step what it needs?" is answered where it is asked:
        # a kind with nothing in scope is shown as such, not omitted.
        s, w = self._widget()
        row = "".join(c.value for c in w.box.children[0].children[0].children)
        assert "basis</code> = frame" in row
        for kind in ("coord", "level", "op", "rules"):
            assert f"{kind}</code>" in row and "none in scope" in row
        assert "td.explore(expr, basis=" in w.box.children[0].children[1].value
