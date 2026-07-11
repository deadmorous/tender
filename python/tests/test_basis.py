"""Tests for the tender.basis Python bindings."""

import pytest

import tender
import tender.basis as tb
import tender.derivation as td
from tender import Realm


# ---------------------------------------------------------------------------
# Basis construction and accessors
# ---------------------------------------------------------------------------


class TestBasis:
    def test_make_orthonormal_basis(self):
        ctx = tender.Context()
        i = tender.tensor("i", rank=1, ctx=ctx)
        j = tender.tensor("j", rank=1, ctx=ctx)
        k = tender.tensor("k", rank=1, ctx=ctx)
        b = tb.make_orthonormal_basis([i, j, k], tender.space_3d)
        assert b.dim == 3
        assert b.is_orthonormal
        assert b.realm == Realm.Orthonormal
        assert b.space is tender.space_3d
        assert b.vector_symbol == "e"

    def test_custom_symbol(self):
        ctx = tender.Context()
        u = tender.tensor("u", rank=1, ctx=ctx)
        v = tender.tensor("v", rank=1, ctx=ctx)
        b = tb.make_orthonormal_basis([u, v], tender.space_2d, symbol="g")
        assert b.vector_symbol == "g"
        assert b.dim == 2

    def test_cardinality_mismatch_raises(self):
        ctx = tender.Context()
        i = tender.tensor("i", rank=1, ctx=ctx)
        j = tender.tensor("j", rank=1, ctx=ctx)
        with pytest.raises(Exception):
            tb.make_orthonormal_basis([i, j], tender.space_3d)

    def test_symbolic_emission(self):
        ctx = tender.Context()
        b = tb.wcs(ctx)
        m = ctx.alloc_index()
        e_m = b.covariant_vector(m)
        # e_m renders with the generic symbol and the given index name.
        assert "e" in e_m.latex()


# ---------------------------------------------------------------------------
# Coordinate systems
# ---------------------------------------------------------------------------


class TestCoordSystems:
    def test_wcs(self):
        ctx = tender.Context()
        b = tb.wcs(ctx)
        assert b.dim == 3
        assert b.is_orthonormal
        assert "i" in b.basis(0).latex()

    def test_cylindrical(self):
        b = tb.cylindrical(tender.Context())
        assert b.dim == 3
        assert "r" in b.basis(0).latex()
        assert "theta" in b.basis(1).latex()

    def test_spherical(self):
        b = tb.spherical(tender.Context())
        assert b.dim == 3
        assert "phi" in b.basis(2).latex()

    def test_polar_2d(self):
        b = tb.polar_2d(tender.Context())
        assert b.dim == 2
        assert b.space is tender.space_2d


# ---------------------------------------------------------------------------
# Basis-parameterized steps
# ---------------------------------------------------------------------------


class TestBasisIdentity:
    def test_same_invariant_in_two_bases_is_distinct(self):
        # vibe 000067: expanding the same vector in two different bases yields
        # results that look identical but are algebraically distinct (the
        # coordinates/vectors carry different basis ids).
        ctx = tender.Context()
        b1 = tb.wcs(ctx)
        b2 = tb.cylindrical(ctx)
        a = tender.tensor("a", rank=1, ctx=ctx)
        e1 = tb.expand_in_basis(a, b1, tb.Variance.Covariant)
        e2 = tb.expand_in_basis(a, b2, tb.Variance.Covariant)
        assert e1.latex() == e2.latex()  # rendering of the tag is increment 4
        assert not td.algebraic_eq(e1, e2)
        assert td.algebraic_eq(e1, e1)

    def test_concrete_indices_render_with_coordinate_letters(self):
        # vibe 000067 increment 4: once unrolled to concrete directions, a
        # cylindrical expansion reads in r, θ, z rather than 1, 2, 3.
        ctx = tender.Context()
        cyl = tb.cylindrical(ctx)
        a = tender.tensor("a", rank=1, ctx=ctx)
        exp = td.canonicalize(tb.expand_in_basis(a, cyl, tb.Variance.Covariant))
        unrolled = td.unroll_sums(exp)
        assert unrolled.latex() == (
            r"a_{r} \, \mathbf{e}_{r} + a_{\theta} \, \mathbf{e}_{\theta}"
            r" + a_{z} \, \mathbf{e}_{z}"
        )

    def test_wcs_unrolls_to_ijk_vectors(self):
        # vibe 000067 4b: WCS frame vectors print as the classic standalone
        # i, j, k, with x, y, z coordinate components.
        ctx = tender.Context()
        wcs = tb.wcs(ctx)
        a = tender.tensor("a", rank=1, ctx=ctx)
        exp = td.unroll_sums(
            td.canonicalize(tb.expand_in_basis(a, wcs, tb.Variance.Covariant))
        )
        assert exp.latex() == (
            r"a_{x} \, \mathbf{i} + a_{y} \, \mathbf{j} + a_{z} \, \mathbf{k}"
        )

    def test_dot_with_frame_vector_contracts(self):
        # vibe 000068 P1/P3: dotting an expansion with a concrete frame vector
        # cs.basis(0) now contracts; contract_delta (or unroll + eval) finishes
        # I·e_1 = e_1.
        ctx = tender.Context()
        cs = tb.wcs(ctx)
        I = tender.identity(ctx=ctx)
        term = tb.expand_in_basis(I, cs, tb.Variance.Covariant) @ cs.basis(0)

        # The result is the direction-1 vector e_1, which renders as the frame
        # letter "i" (it is the symbolic-concrete form, not the frame vector
        # object, so we compare the rendered result).
        want = cs.basis(0).latex()  # "\mathbf{i}"

        # P1: symbolic δ_{i1}, then contract_delta.
        p1 = td.contract_delta(tb.simplify_basis_dot(term, cs))
        assert td.canonicalize(p1).latex() == want

        # P3: unroll first, concrete δ, then eval_delta_concrete.
        unrolled = td.unroll_sums(term)
        p3 = td.fold_arithmetic(
            td.eval_delta_concrete(tb.simplify_basis_dot(unrolled, cs))
        )
        assert td.canonicalize(p3).latex() == want

    def test_reassemble_also_does_completeness(self):
        # vibe 000068 P2: reassemble finishes I·e_1 = e_1 in one call, without
        # the caller reaching for reassemble_completeness.
        ctx = tender.Context()
        cs = tb.wcs(ctx)
        I = tender.identity(ctx=ctx)
        term = tb.expand_in_basis(I, cs, tb.Variance.Covariant) @ cs.basis(0)
        assert td.structural_eq(tb.reassemble(term, cs), cs.basis(0))

    def test_reassemble_ignores_foreign_basis(self):
        # vibe 000067 increment 3: a step keyed to one basis only acts on that
        # basis's coordinates/vectors.  An expansion in cylindrical is not
        # reassembled by WCS; the matching basis does fold it back.
        ctx = tender.Context()
        wcs = tb.wcs(ctx)
        cyl = tb.cylindrical(ctx)
        v = tender.tensor("v", rank=1, ctx=ctx)
        exp = td.canonicalize(tb.expand_in_basis(v, cyl, tb.Variance.Covariant))
        assert not td.structural_eq(
            td.canonicalize(tb.reassemble(exp, wcs)), v
        )
        assert td.structural_eq(tb.reassemble(exp, cyl), v)


class TestBasisSteps:
    def test_expand_reassemble_round_trip(self):
        ctx = tender.Context()
        b = tb.wcs(ctx)
        a = tender.tensor("a", rank=1, ctx=ctx)
        expanded = td.canonicalize(
            tb.expand_in_basis(a, b, tb.Variance.Covariant)
        )
        assert td.structural_eq(tb.reassemble(expanded, b), a)

    def test_rank2_round_trip(self):
        ctx = tender.Context()
        b = tb.wcs(ctx)
        A = tender.tensor("A", rank=2, ctx=ctx)
        expanded = td.canonicalize(
            tb.expand_in_basis(A, b, tb.Variance.Covariant)
        )
        assert td.structural_eq(tb.reassemble(expanded, b), A)

    def test_identity_round_trip(self):
        # I -> Σ_i e_i ⊗ e^i -> I.
        ctx = tender.Context()
        b = tb.wcs(ctx)
        I = tender.identity(ctx=ctx)
        expanded = td.canonicalize(
            tb.expand_in_basis(I, b, tb.Variance.Covariant)
        )
        assert td.structural_eq(tb.reassemble(expanded, b), I)

    def test_oblique_identity_round_trip_and_metric(self):
        ctx = tender.Context()
        a = tender.tensor("a", rank=1, ctx=ctx)
        b = tender.tensor("b", rank=1, ctx=ctx)
        c = tender.tensor("c", rank=1, ctx=ctx)
        basis = tb.make_oblique_basis([a, b, c], tender.space_3d)
        assert basis.realm == tender.Realm.Oblique
        assert not basis.is_orthonormal

        I = tender.identity(ctx=ctx)
        expanded = td.canonicalize(
            tb.expand_in_basis(I, basis, tb.Variance.Covariant)
        )
        assert td.structural_eq(tb.reassemble(expanded, basis), I)

        # I_ij = e_i·I·e_j reduces to the metric g_ij.
        i = ctx.alloc_index()
        j = ctx.alloc_index()
        coord = (basis.covariant_vector(i) @ I) @ basis.covariant_vector(j)
        reduced = tb.simplify_basis_dot(td.contract_identity(coord), basis)
        assert "g" in reduced.latex()

    def test_reassemble_no_op_on_foreign(self):
        ctx = tender.Context()
        b = tb.wcs(ctx)
        # A plain tensor is not an expansion: reassemble leaves it unchanged.
        x = tender.tensor("x", rank=1, ctx=ctx)
        assert td.structural_eq(tb.reassemble(x, b), x)

    def test_per_slot_variance_round_trip(self):
        # Mixed variance A^i_j: a list of Variance, one per slot.
        ctx = tender.Context()
        b = tb.wcs(ctx)
        A = tender.tensor("A", rank=2, ctx=ctx)
        expanded = td.canonicalize(
            tb.expand_in_basis(
                A, b, [tb.Variance.Covariant, tb.Variance.Contravariant]
            )
        )
        assert td.structural_eq(tb.reassemble(expanded, b), A)

    def test_variance_count_mismatch_raises(self):
        ctx = tender.Context()
        b = tb.wcs(ctx)
        a = tender.tensor("a", rank=1, ctx=ctx)
        with pytest.raises(Exception):
            tb.expand_in_basis(
                a, b, [tb.Variance.Covariant, tb.Variance.Contravariant]
            )

    def test_cross_gives_levi_civita(self):
        # e_i × e_j → ε_{ijk} e_k (orthonormal).
        ctx = tender.Context()
        b = tb.wcs(ctx)
        i = ctx.alloc_index()
        j = ctx.alloc_index()
        res = tb.simplify_basis_cross(
            b.covariant_vector(i) % b.covariant_vector(j), b
        )
        assert "varepsilon" in res.latex()
        assert "-1" not in res.latex()  # right-handed: √g = +1, no sign

    def test_left_handed_cross_flips_sign(self):
        ctx = tender.Context()
        b = tb.make_orthonormal_basis(
            [
                tender.tensor("i", rank=1, ctx=ctx),
                tender.tensor("j", rank=1, ctx=ctx),
                tender.tensor("k", rank=1, ctx=ctx),
            ],
            tender.space_3d,
            handedness=tb.Handedness.Left,
        )
        i = ctx.alloc_index()
        j = ctx.alloc_index()
        res = tb.simplify_basis_cross(
            b.covariant_vector(i) % b.covariant_vector(j), b
        )
        assert "-1" in res.latex()  # left-handed: √g = -1

    def test_cross_with_identity_commutes(self):
        # a × I = I × a, derived through the basis.
        ctx = tender.Context()
        frame = tb.wcs(ctx)
        a = tender.tensor("a", rank=1, ctx=ctx)
        I = tender.identity(ctx=ctx)

        def reduce(e):
            # simplify_basis_cross distributes over the identity dyad itself.
            e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
            e = tb.simplify_basis_cross(e, frame)
            return td.canonicalize(e)

        assert td.algebraic_eq(reduce(a % I), reduce(I % a))

    def test_dot_product_commutes(self):
        ctx = tender.Context()
        b = tb.wcs(ctx)
        a = tender.tensor("a", rank=1, ctx=ctx)
        c = tender.tensor("b", rank=1, ctx=ctx)

        def reduce(expr):
            expr = tb.expand_in_basis(expr, b, tb.Variance.Covariant)
            expr = tb.simplify_basis_dot(expr, b)
            expr = td.canonicalize(expr)
            expr = td.unroll_sums(expr)
            expr = td.eval_delta_concrete(expr)
            expr = td.fold_arithmetic(expr)
            return td.fold_sums(td.canonicalize(expr))

        assert td.algebraic_eq(reduce(a @ c), reduce(c @ a))


def test_dimension_aware_identity_is_orthogonal_to_slots():
    # vibe 000082: dimension-awareness is a TensorObject attribute orthogonal to
    # the index slots (not fake unbound slots).  The identity is slotless, so it
    # renders as a clean I and expands in a basis normally.  It is BEARING (part
    # of identity, so `tr` stays a congruence: a 2-D I ≠ a 3-D I), and there is
    # no dimension-agnostic identity — the default is 3-D.
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    default = tender.identity(ctx=ctx)  # 3-D by default
    three_d = tender.identity(ctx=ctx, space=tender.space_3d)
    two_d = tender.identity(ctx=ctx, space=tender.space_2d)
    assert default.latex() == three_d.latex() == two_d.latex() == r"\mathbf{I}"
    assert td.structural_eq(default, three_d)  # default IS 3-D
    assert not td.structural_eq(three_d, two_d)  # bearing: distinct objects
    # tr reads the dimension (congruent — distinct I's, distinct traces)
    assert td.expand_dyad_ops(tender.tr(three_d)).latex() == "3"
    assert td.expand_dyad_ops(tender.tr(two_d)).latex() == "2"
    # slotless: expands in a matching basis normally, no fake index bullets
    exp = td.canonicalize(tb.expand_in_basis(three_d, frame, tb.Variance.Covariant))
    assert r"\bullet" not in exp.latex()
    # but a dimension mismatch (2-D I on a 3-D frame) is refused
    with pytest.raises(ValueError, match="dimension does not match"):
        tb.expand_in_basis(two_d, frame, tb.Variance.Covariant)


def test_vec_of_identity_is_zero():
    # vec(I) = 0 through the basis: I = Σ e_i⊗e_i, vec → Σ e_i×e_i, each = 0.
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    e = tender.vec(tender.identity(ctx=ctx))
    e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
    e = td.expand_dyad_ops(e)
    e = tb.simplify_basis_cross(e, frame)
    e = td.unroll_sums(e)
    e = td.eval_eps_concrete(e)
    e = td.canonicalize(td.fold_arithmetic(e))
    assert td.algebraic_eq(e, tender.scalar(0, ctx=ctx))


def test_user_ddot_identity_fires_on_basis_expansion():
    # A hand-written dyad identity fires on the basis-expanded I:I via
    # subtree variables + binder-to-top canonicalization (vibe 000051).
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    c = tender.tensor("c", rank=1, ctx=ctx)
    d = tender.tensor("d", rank=1, ctx=ctx)
    I = tender.identity(ctx=ctx)
    expand_ddot = td.Identity("ddot", (a * b).ddot(c * d), (a @ c) * (b @ d))

    y = tb.expand_in_basis(I.ddot(I), frame, tb.Variance.Covariant)
    y = td.apply_identity(expand_ddot)(y)   # fires now
    y = tb.simplify_basis_dot(y, frame)
    y = td.contract_delta(y)
    y = td.unroll_sums(y)
    y = td.fold_arithmetic(td.eval_delta_concrete(y))
    assert td.algebraic_eq(td.canonicalize(y), tender.scalar(3, ctx=ctx))


def test_bac_cab():
    # a × (b × c) = b(a·c) − c(a·b), via ε-pair contraction in a product.
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    c = tender.tensor("c", rank=1, ctx=ctx)

    def reduce_cross(e):
        e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
        e = tb.simplify_basis_cross(e, frame)
        e = td.canonicalize(e)
        e = td.contract_eps_pair(e)
        e = td.expand_products(e)
        e = td.canonicalize(e)
        e = td.unroll_sums(e)
        e = td.eval_delta_concrete(e)
        e = td.fold_arithmetic(e)
        return td.canonicalize(e)

    def reduce_dot(e):
        e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
        e = tb.simplify_basis_dot(e, frame)
        e = td.canonicalize(e)
        e = td.unroll_sums(e)
        e = td.eval_delta_concrete(e)
        e = td.fold_arithmetic(e)
        return td.canonicalize(e)

    lhs = reduce_cross(a % (b % c))
    rhs = reduce_dot(b * (a @ c) - c * (a @ b))
    assert td.algebraic_eq(lhs, rhs)


def test_cross_identity_cross():
    # a × I × b = b ⊗ a − (a·b) I — the rank-2 cross identity.
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    I = tender.identity(ctx=ctx)

    def reduce_cross(e):
        e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
        e = tb.simplify_basis_cross(e, frame)
        e = td.canonicalize(e)
        e = td.contract_eps_pair(e)
        e = td.expand_products(e)
        e = td.canonicalize(e)
        e = td.unroll_sums(e)
        e = td.eval_delta_concrete(e)
        e = td.fold_arithmetic(e)
        return td.canonicalize(e)

    def reduce_dot(e):
        e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
        e = tb.simplify_basis_dot(e, frame)
        e = td.canonicalize(e)
        e = td.unroll_sums(e)
        e = td.eval_delta_concrete(e)
        e = td.fold_arithmetic(e)
        return td.canonicalize(e)

    lhs = reduce_cross((a % I) % b)
    rhs = reduce_dot(b * a - (a @ b) * I)
    assert td.algebraic_eq(lhs, rhs)
    # Not the transpose.
    assert not td.algebraic_eq(lhs, reduce_dot(a * b - (a @ b) * I))


def test_cross_identity_cross_via_reassembly():
    # Second proof of a × I × b: pattern matcher (bac-cab as a reusable
    # identity) + completeness reassembly, landing in pure direct notation.
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    I = tender.identity(ctx=ctx)
    x = tender.tensor("x", rank=1, ctx=ctx)
    y = tender.tensor("y", rank=1, ctx=ctx)
    z = tender.tensor("z", rank=1, ctx=ctx)
    baccab = td.Identity("bac-cab", x % (y % z), y * (x @ z) - z * (x @ y))

    I_exp = tb.expand_in_basis(I, frame, tb.Variance.Covariant)
    s = td.canonicalize(a % (b % I_exp))
    s = td.distribute_contraction(s)
    s = td.canonicalize(s)
    s = td.apply_identity(baccab)(s)
    s = td.expand_products(s)
    s = td.canonicalize(s)
    s = tb.reassemble_completeness(s, frame)
    s = td.canonicalize(s)

    want = td.canonicalize(b * a - (a @ b) * I)
    assert td.structural_eq(s, want)


def test_cross_reassociation_exposes_identity():
    # a % I % b parses left-associated as (a%I)%b, so the subterm I%b the commute
    # identity needs is not a node.  Because I is a rank-2 fence, canon re-
    # associates to a%(I%b); apply_identity (which canonicalizes first) then fires
    # I%x = x%I on the exposed I%b, reaching a%(b%I) regardless of bracketing.
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    I = tender.identity(ctx=ctx)
    x = tender.tensor("x", rank=1, ctx=ctx)
    commute = td.Identity("I-commute", I % x, x % I)

    got = td.apply_identity(commute)((a % I) % b)
    want = td.canonicalize(a % (b % I))
    assert td.structural_eq(got, want)
