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
