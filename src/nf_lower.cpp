#include <tender/nf_lower.hpp>

#include <tender/derivation.hpp> // infer_rank
#include <tender/rewrite.hpp>    // rewrite_tree (fold_forced_zeros)
#include <tender/summation.hpp>  // contracted_ids, substitute_index_ids, …
#include <tender/tensor_symmetry.hpp> // canon_symmetry_slots

#include <mpk/mix/util/overloads.hpp>

#include <algorithm>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <variant>

namespace tender::nf
{

using mpk::mix::Overloads;

// ---- pass 2: additive flatten (C3) -------------------------------------

namespace
{

void flatten(Expr const* e, int sign, std::vector<SignedExpr>& out)
{
    if (auto const* s = std::get_if<Sum>(&e->node))
    {
        flatten(s->left, sign, out);
        flatten(s->right, sign, out);
    }
    else if (auto const* d = std::get_if<Difference>(&e->node))
    {
        flatten(d->left, sign, out);
        flatten(d->right, -sign, out);
    }
    else if (auto const* n = std::get_if<Negate>(&e->node))
    {
        flatten(n->operand, -sign, out);
    }
    else
    {
        // Any non-additive node (including a product whose interior contains a
        // sum) is an opaque leaf — no distribution.
        out.push_back({sign, e});
    }
}

} // namespace

auto additive_flatten(Expr const* e) -> std::vector<SignedExpr>
{
    std::vector<SignedExpr> out;
    flatten(e, +1, out);
    return out;
}

// ---- pass 3a: multiplicative flatten (C4) ------------------------------

namespace
{

void flatten_product(Expr const* e, ProductParts& out)
{
    if (auto const* p = std::get_if<TensorProduct>(&e->node))
    {
        flatten_product(p->left, out);
        flatten_product(p->right, out);
    }
    else if (auto const* sl = std::get_if<ScalarLiteral>(&e->node))
    {
        out.coeff *= sl->value;
    }
    else if (auto const* n = std::get_if<Negate>(&e->node))
    {
        out.coeff *= Rational{-1};
        flatten_product(n->operand, out);
    }
    else if (auto const* d = std::get_if<ScalarDiv>(&e->node);
             d != nullptr
             && std::holds_alternative<ScalarLiteral>(d->right->node))
    {
        out.coeff /= std::get<ScalarLiteral>(d->right->node).value;
        flatten_product(d->left, out);
    }
    else
    {
        // A contraction / cross / sum / non-numeric division node is one
        // opaque factor; encapsulation into an Nf Factor is C5/C6.
        out.factors.push_back(e);
    }
}

} // namespace

auto multiplicative_flatten(SignedExpr const& term) -> ProductParts
{
    ProductParts out{.coeff = Rational{term.sign}, .factors = {}};
    flatten_product(term.body, out);
    return out;
}

// ---- pass 3b: factor encapsulation (C5) --------------------------------

namespace
{

// The contraction-family operator of `e`, if it is one.
auto contraction_op(Expr const* e) -> std::optional<COp>
{
    if (std::holds_alternative<Dot>(e->node))
        return COp::Dot;
    if (std::holds_alternative<DDot>(e->node))
        return COp::DDot;
    if (std::holds_alternative<DDotAlt>(e->node))
        return COp::DDotAlt;
    return std::nullopt;
}

auto binop_operands(Expr const* e) -> std::pair<Expr const*, Expr const*>
{
    return visit(
        Overloads{
            [](Dot const& d) { return std::pair{d.left, d.right}; },
            [](DDot const& d) { return std::pair{d.left, d.right}; },
            [](DDotAlt const& d) { return std::pair{d.left, d.right}; },
            [](auto const&) -> std::pair<Expr const*, Expr const*>
            { throw std::logic_error("binop_operands: not a contraction"); }},
        *e);
}

// A flattened contraction in left-fold order, annotated with where its result's
// free legs live so a parent `·` can splice its operands correctly.
//
// The flat `Contraction` model reads `factors` as a left fold
// `(((f0·f1)·f2)…)`. The bracketing of the *source* tree is immaterial only for
// a clean matrix chain (the 000057 interface theorem): a rank-1 "connector" —
// as in `a·(b·T)`, where `b` is fully consumed contracting into `T` — fans `a`
// onto an *interior* leg of `T`, so the naive `flatten(l)++[op]++flatten(r)`
// reads `(a·b)·T`, silently changing which legs meet (vibe 000078 bug 3b:
// `a·(b·T)` came out `a·b T`, rank 0 → 2).  We therefore track, per sub-chain,
// whether its result exposes a free leg at each physical tip, and orient /
// reorder its operands so the contracted legs actually meet — transposing only
// when a rank-≤1 sub-chain must be read backwards (a vector equals its
// transpose; a scalar is orientation-free), which never disturbs a matrix
// chain.
struct FlatChain final
{
    std::vector<Expr const*> seq;
    std::vector<COp> ops;
    int rank = 0;
    bool free_front = false; // result exposes a free leg at seq.front()
    bool free_back = false;  // …and/or at seq.back()
};

// Reverse a contraction chain and transpose it as a whole: `(A·B·C)ᵀ =
// Cᵀ·Bᵀ·Aᵀ`.  Rank-1 factors are left as-is (a vector equals its transpose;
// skipping the notation avoids a spurious `aᵀ`).  Only ever applied to a
// rank-≤1 result, where reading the chain backwards is benign.
auto reverse_transpose(Context& ctx, FlatChain c) -> FlatChain
{
    FlatChain out;
    out.rank = c.rank;
    out.free_front = c.free_back;
    out.free_back = c.free_front;
    for (auto it = c.seq.rbegin(); it != c.seq.rend(); ++it)
        out.seq.push_back(
            infer_rank(*it) == std::optional<int>{1} ?
                *it :
                make_transpose(ctx, *it));
    out.ops.assign(c.ops.rbegin(), c.ops.rend());
    return out;
}

// `[left…] op [right…]`, junction `left.back` · `right.front`; no
// reorientation.
auto concat_chains(
    FlatChain left, COp op, FlatChain right, int rank, bool ff, bool fb)
    -> FlatChain
{
    left.ops.push_back(op);
    left.seq.insert(left.seq.end(), right.seq.begin(), right.seq.end());
    left.ops.insert(left.ops.end(), right.ops.begin(), right.ops.end());
    left.rank = rank;
    left.free_front = ff;
    left.free_back = fb;
    return left;
}

auto flatten_chain(Context& ctx, Expr const* e) -> FlatChain
{
    auto op = contraction_op(e);
    if (!op)
    {
        auto r = infer_rank(e);
        int rank = r.value_or(-1);
        bool leg = !r || *r >= 1; // a leaf's leg(s) sit at both tips
        return {{e}, {}, rank, leg, leg};
    }
    auto [le, re] = binop_operands(e);
    FlatChain x = flatten_chain(ctx, le);
    FlatChain y = flatten_chain(ctx, re);

    // Double contractions (`:` / `··`) act on rank-≥2 operands — no rank-1
    // connector is possible — so naive concatenation is already faithful.
    // Unknown-rank operands fall back the same way (we cannot leg-track without
    // ranks).  Both preserve the historical behaviour.
    if (*op != COp::Dot || x.rank < 0 || y.rank < 0)
    {
        int rank = (x.rank < 0 || y.rank < 0) ?
                       -1 :
                       x.rank + y.rank - (*op == COp::Dot ? 2 : 4);
        bool leg = rank >= 1;
        return concat_chains(std::move(x), *op, std::move(y), rank, leg, leg);
    }

    int const rank = x.rank + y.rank - 2;
    bool const xff = x.rank >= 2; // an x on the left keeps a free leg up front
    bool const yfb = y.rank >= 2; // a y on the right keeps a free leg at back

    // x contributes its LAST free leg, y its FIRST.  A leg's tip: for a rank-≤1
    // chain the (single) free leg can serve either role; a rank-≥2 chain's
    // first leg is its front, last leg its back — fixed.
    bool const x_last_back = x.free_back; // x's last leg at back
    bool const x_last_front = x.free_front && x.rank <= 1; // …or at front
                                                           // (rank≤1)
    bool const y_first_front = y.free_front;              // y's first leg front
    bool const y_first_back = y.free_back && y.rank <= 1; // …or at back
                                                          // (rank≤1)

    // Prefer the two transpose-free splices; else read one rank-≤1 side back.
    if (x_last_back && y_first_front) // [x…]·[y…]
        return concat_chains(
            std::move(x), COp::Dot, std::move(y), rank, xff, yfb);
    if (x_last_front && y_first_back) // [y…]·[x…]
        return concat_chains(
            std::move(y), COp::Dot, std::move(x), rank, yfb, xff);
    if (x_last_back && y_first_back) // [x…]·[yᵀ…]
    {
        FlatChain yt = reverse_transpose(ctx, std::move(y));
        return concat_chains(
            std::move(x), COp::Dot, std::move(yt), rank, xff, yfb);
    }
    // x_last_front && y_first_front: [xᵀ…]·[y…]
    FlatChain xt = reverse_transpose(ctx, std::move(x));
    return concat_chains(std::move(xt), COp::Dot, std::move(y), rank, xff, yfb);
}

// Flatten a contraction tree into operand / op sequences in faithful left-fold
// order (`operands.size() == ops.size() + 1`).  See `flatten_chain`.
void flatten_contraction(
    Context& ctx,
    Expr const* e,
    std::vector<Expr const*>& operands,
    std::vector<COp>& ops)
{
    FlatChain c = flatten_chain(ctx, e);
    operands.insert(operands.end(), c.seq.begin(), c.seq.end());
    ops.insert(ops.end(), c.ops.begin(), c.ops.end());
}

// A bare rank-1 vector (the only operand cross anticommutation applies to).
auto is_rank1_vector(Expr const* e) -> bool
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    return t && t->rank && *t->rank == 1 && t->slots.empty();
}

auto has_rank(Expr const* e, int n) -> bool
{
    return infer_rank(e) == std::optional<int>{n};
}

// Whether a binary contraction with operator `op` over Expr operands `l`, `r`
// is commutative, so its two operands may be ordered canonically.  A single
// contraction joins adjacent interface legs; swapping the operands swaps which
// legs meet, so it is symmetric *only* when both operands have exactly the
// contracted rank (the result is then a scalar):
//   - `·` removes one leg from each side, so it commutes only between two
//     rank-1 vectors (`a·b = b·a`); `A·b`, a matrix-vector product, does not;
//   - `:` / `··` remove two legs from each side, so they commute only between
//     two rank-2 tensors (`A:B = B:A`).  Higher-rank double contractions such
//     as `C:ε` (rank-4 `:` rank-2, e.g. stress = stiffness : strain) are
//     directional — `C:ε ≠ ε:C` — and must keep their operand order.
auto contraction_commutes(COp op, Expr const* l, Expr const* r) -> bool
{
    switch (op)
    {
        case COp::Dot: return is_rank1_vector(l) && is_rank1_vector(r);
        case COp::DDot:
        case COp::DDotAlt: return has_rank(l, 2) && has_rank(r, 2);
    }
    return false;
}

// Re-associate a cross around a rank-≥2 fence: `(x×M)×z → x×(M×z)` when M is
// rank ≥ 2 (the ⊗ inside M fences the two crosses onto disjoint legs, so the
// bracketing is immaterial — 000055).  Returns the re-associated `Expr`, or
// nullptr when the pattern does not apply.  Mirrors derivation.cpp's helper of
// the same name (kept local to avoid disturbing that translation unit).
auto reassociate_cross_fence(Context& ctx, Expr const* l, Expr const* r)
    -> Expr const*
{
    // `Cross` / `make_cross` here are the Expr-level ones (tender::), not the
    // Nf factor of the same name in this namespace.
    auto const* inner = std::get_if<tender::Cross>(&l->node);
    if (!inner)
        return nullptr;
    auto const rx = infer_rank(inner->left);
    auto const rm = infer_rank(inner->right);
    auto const rz = infer_rank(r);
    if (rx == std::optional<int>{1} && rm && *rm >= 2
        && rz == std::optional<int>{1})
        return tender::make_cross(
            ctx, inner->left, tender::make_cross(ctx, inner->right, r));
    return nullptr;
}

} // namespace

auto encapsulate(Context& ctx, Expr const* factor) -> SignedFactor
{
    if (auto const* t = std::get_if<TensorObject>(&factor->node))
    {
        // Symmetry-orbit canonicalization (vibe 000047): put a symmetric /
        // antisymmetric tensor's slots into orbit-minimal order, folding the
        // antisymmetric sign out.  Sign 0 means the object is identically zero
        // (e.g. ε with a repeated index) — carried as a 0 multiplier so the
        // whole term collects to coeff 0 and drops out.
        auto [slots, sign] = canon_symmetry_slots(*t);
        if (sign == 0)
            return {0, make_atom(ctx, *t)};
        TensorObject obj = *t;
        obj.slots = std::move(slots);
        return {sign, make_atom(ctx, std::move(obj))};
    }

    if (contraction_op(factor))
    {
        std::vector<Expr const*> operands;
        std::vector<COp> ops;
        flatten_contraction(ctx, factor, operands, ops);
        std::vector<Factor const*> encapsulated;
        encapsulated.reserve(operands.size());
        int sign = +1;
        for (auto const* o: operands)
        {
            auto sf = encapsulate(ctx, o);
            sign *= sf.sign;
            encapsulated.push_back(sf.factor);
        }
        // Interior order: a binary commutative contraction gets its two
        // operands in canonical order (no sign change — unlike cross).
        if (encapsulated.size() == 2
            && contraction_commutes(ops[0], operands[0], operands[1])
            && compare(*encapsulated[0], *encapsulated[1]) > 0)
            std::swap(encapsulated[0], encapsulated[1]);
        return {
            sign,
            make_contraction(ctx, std::move(encapsulated), std::move(ops))};
    }

    // Unary invariants: linear, so the operand's lifted sign passes through.
    if (auto const* u = std::get_if<Trace>(&factor->node))
    {
        auto sf = encapsulate(ctx, u->operand);
        return {sf.sign, make_unary(ctx, UnaryOp::Trace, sf.factor)};
    }
    if (auto const* u = std::get_if<VectorInvariant>(&factor->node))
    {
        auto sf = encapsulate(ctx, u->operand);
        return {sf.sign, make_unary(ctx, UnaryOp::VectorInvariant, sf.factor)};
    }
    if (auto const* u = std::get_if<Transpose>(&factor->node))
    {
        // Transpose is an involution: `(Xᵀ)ᵀ = X`.  A transpose-of-transpose
        // undoes itself — fold it away before the symmetric-fold / wrap logic
        // (otherwise the inner Tᵀ encapsulates to `Transpose(X)` and the outer
        // wraps it again into `Transpose(Transpose(X))`, which survives into
        // the nf form and renders as stacked superscripts).
        if (auto const* inner = std::get_if<Transpose>(&u->operand->node))
            return encapsulate(ctx, inner->operand);
        // A rank-2 tensor symmetric under its slot swap folds `Tᵀ = T`
        // (antisymmetric `Tᵀ = −T`): transpose is exactly that swap.  So
        // `εᵀ = ε`, and — since the swap leaves the ∂-marks untouched —
        // `(∂∂ε)ᵀ = ∂∂ε`, which the strain-compatibility reduction needs.  The
        // invariant form carries no index slots, so probe the symmetry with two
        // synthetic ones and compare the canonical orbits.
        if (auto const* t = std::get_if<TensorObject>(&u->operand->node);
            t && t->traits && t->rank && *t->rank == 2)
        {
            TensorObject probe = *t;
            if (probe.slots.size() != 2)
                probe.slots = {
                    SlotBinding{
                        IndexSlot{},
                        IndexAssoc{CountableIndex{ctx.alloc_index_id()}}},
                    SlotBinding{
                        IndexSlot{},
                        IndexAssoc{CountableIndex{ctx.alloc_index_id()}}}};
            TensorObject swapped = probe;
            std::swap(swapped.slots[0], swapped.slots[1]);
            auto [cp, sp] = canon_symmetry_slots(probe);
            auto [cs, ss] = canon_symmetry_slots(swapped);
            if (sp != 0 && ss != 0 && slot_seq_cmp(cp, cs) == 0)
            {
                // Same orbit ⇒ the swap is a symmetry: `Tᵀ = (sp·ss)·T`.
                auto sf = encapsulate(ctx, u->operand);
                return {sp * ss * sf.sign, sf.factor};
            }
        }
        auto sf = encapsulate(ctx, u->operand);
        return {sf.sign, make_unary(ctx, UnaryOp::Transpose, sf.factor)};
    }
    // A ∂ operator (vibe 000077): its wrt-object is a bare TensorObject (a
    // coordinate today); carry it into the nf Deriv factor.
    if (auto const* d = std::get_if<tender::Deriv>(&factor->node))
    {
        auto const* w = std::get_if<TensorObject>(&d->wrt->node);
        if (!w)
            throw std::invalid_argument(
                "encapsulate: ∂ operator wrt-object must be a bare tensor "
                "object (a coordinate)");
        return {1, make_deriv(ctx, *w)};
    }
    // The chart-free ∇ operator (vibe 000078): a childless operator factor.
    if (std::holds_alternative<tender::Nabla>(factor->node))
        return {1, nf::make_nabla(ctx)};

    if (auto const* c = std::get_if<tender::Cross>(&factor->node))
    {
        // Anticommutation: a rank-1 pair is ordered canonically, lifting the
        // sign `a×b = -(b×a)`.  Mirrors the canon Cross arm.
        if (is_rank1_vector(c->left) && is_rank1_vector(c->right))
        {
            auto sl = encapsulate(ctx, c->left);
            auto sr = encapsulate(ctx, c->right);
            int sign = sl.sign * sr.sign;
            if (compare(*sl.factor, *sr.factor) > 0)
                return {-sign, make_cross(ctx, {sr.factor, sl.factor})};
            return {sign, make_cross(ctx, {sl.factor, sr.factor})};
        }
        // Rank-≥2 fence: re-associate, then encapsulate the result.
        if (auto const* ra = reassociate_cross_fence(ctx, c->left, c->right))
            return encapsulate(ctx, ra);
        // General binary cross (e.g. a nested cross operand): structural.
        auto sl = encapsulate(ctx, c->left);
        auto sr = encapsulate(ctx, c->right);
        return {sl.sign * sr.sign, make_cross(ctx, {sl.factor, sr.factor})};
    }

    // A bare `Negate` factor (e.g. a contraction operand `A·(−b)`): lift the
    // sign and encapsulate the operand.  (A top-level / multiplicative `Negate`
    // is already folded into `coeff` by multiplicative_flatten; this arm only
    // fires for a `Negate` reached through a contraction / cross operand.)
    if (auto const* n = std::get_if<Negate>(&factor->node))
    {
        auto sf = encapsulate(ctx, n->operand);
        return {-sf.sign, sf.factor};
    }

    // A *genuine sum* factor (`b + c`, `b − c`) — never distributed — becomes a
    // `Paren` whose interior is the recursively canonicalized `Nf` (000057 /
    // C10).  Its sign rides in the inner terms' coeffs, so the lifted sign here
    // is `+1`.
    if (std::holds_alternative<Sum>(factor->node)
        || std::holds_alternative<Difference>(factor->node))
        return {+1, make_paren(ctx, canonicalize_nf(ctx, factor))};

    // A symbolic division (`A / B`, non-numeric divisor — a numeric one folds
    // into `coeff` in multiplicative_flatten) becomes a `Div` over recursively
    // canonical numerator / denominator `Nf`s.
    if (auto const* d = std::get_if<ScalarDiv>(&factor->node))
        return {
            +1,
            make_div(
                ctx,
                canonicalize_nf(ctx, d->left),
                canonicalize_nf(ctx, d->right))};

    // Scalar fields (vibe 000069): an elementary function / power becomes the
    // corresponding scalar Factor over recursively canonical operand `Nf`s.
    // (`tender::ScalarFn` / `tender::Pow` are the Expr nodes; the unqualified
    // `make_scalar_fn` / `make_pow` here are this namespace's Factor builders.)
    if (auto const* fn = std::get_if<tender::ScalarFn>(&factor->node))
        return {
            +1,
            make_scalar_fn(ctx, fn->kind, canonicalize_nf(ctx, fn->operand))};
    if (auto const* pw = std::get_if<tender::Pow>(&factor->node))
        return {
            +1,
            make_pow(
                ctx,
                canonicalize_nf(ctx, pw->base),
                canonicalize_nf(ctx, pw->exponent))};

    // An *operator* ⊗-fence — a `∇⊗X` (grad) kept intact because ∇ is an
    // operator, not a vector, so `distribute_contraction` refused to float it
    // (`∇·(∇⊗X)` must stay nested, not collapse to `(∇·∇)⊗X`; vibe 000085).
    // The nf model's ⊗ lives only at the term level, so carry the fence
    // opaquely as a `Paren` over its canonical sub-`Nf`, exactly like a genuine
    // sum (line ~472).  It round-trips: `∇·(∇⊗X)` raises back to
    // `Dot(∇, ∇⊗X)`, which renders `ΔX`.  The barrier keys on a fence *leg*
    // being ∇, so any fence reaching here has one at an immediate leg; a fence
    // WITHOUT an operator should already be distributed — the throw below still
    // catches that genuine bug.
    if (auto const* p = std::get_if<TensorProduct>(&factor->node);
        p
        && (std::holds_alternative<tender::Nabla>(p->left->node)
            || std::holds_alternative<tender::Nabla>(p->right->node)))
        return {+1, make_paren(ctx, canonicalize_nf(ctx, factor))};

    throw std::invalid_argument(
        "encapsulate: unsupported factor node (a nested ⊗ inside an operand "
        "awaits fence distribution)");
}

// ---- pass 4: region placement (C5) -------------------------------------

auto place_factors(Context& ctx, ProductParts const& pp) -> Term
{
    Term t{.coeff = pp.coeff};
    // A differential operator acts on everything to its right (vibe 000077),
    // so once a term contains one, *no* factor may be commuted — a scalar to an
    // operator's right is its operand (`∂_x x`) and must stay distinct from one
    // to its left (`x ∂_x`).  Detect an operator anywhere in the product and,
    // if present, keep every factor positional (the tensor region) so the whole
    // left-to-right order is preserved.
    bool const has_operator = std::any_of(
        pp.factors.begin(),
        pp.factors.end(),
        [](Expr const* f)
        {
            return std::holds_alternative<tender::Deriv>(f->node)
                   || std::holds_alternative<tender::Nabla>(f->node);
        });
    for (auto const* f: pp.factors)
    {
        // Region by *result* rank: a known scalar (rank 0) joins the
        // commutative scalar region; everything else — a rank-≥1 tensor *or* an
        // abstract tensor of unknown rank — goes positionally into the tensor
        // region.  Unknown rank defaults to tensors because the scalar region
        // is commutative: only a factor we *know* to be scalar may be reordered
        // there, so an unknown stays positional (conservative, never wrong).
        auto rank = infer_rank(f);
        auto enc = encapsulate(ctx, f);
        t.coeff *= Rational{enc.sign}; // lift anticommutation sign into coeff
        if (rank == std::optional<int>{0} && !has_operator)
            t.scalars.push_back(enc.factor);
        else
            t.tensors.push_back(enc.factor);
    }
    // Region 2 is commutative: sort the scalars into canonical order so that
    // like terms collide (tensors stay positional — ⊗ is non-commutative).
    std::sort(
        t.scalars.begin(),
        t.scalars.end(),
        [](Factor const* x, Factor const* y) { return compare(*x, *y) < 0; });
    return t;
}

// ---- pass 5: summation resolution (C8) ---------------------------------

namespace
{

enum class BinderKind
{
    Sum,
    NoSum,
};

struct RawBinder final
{
    int id;
    BinderKind kind;
    Expr const* range = nullptr; // non-null: an explicit `Σ_{i∈range}` bound
};

// Strip the leading `ExplicitSum` / `NoSum` binder stack off a term, recording
// each `(id, kind, range)`.  A ranged `ExplicitSum` carries its symbolic bound
// in `range`.
auto strip_binders(Expr const* e, std::vector<RawBinder>& out) -> Expr const*
{
    for (;;)
    {
        if (auto const* s = std::get_if<ExplicitSum>(&e->node))
        {
            out.push_back({s->index.id, BinderKind::Sum, s->bound});
            e = s->body;
        }
        else if (auto const* n = std::get_if<NoSum>(&e->node))
        {
            out.push_back({n->index.id, BinderKind::NoSum, nullptr});
            e = n->body;
        }
        else
            return e;
    }
}

// Whether the realm rule would *implicitly* contract index `id`, given its free
// occurrences in the term (`uses`, from `collect_term_uses`).  Unlike
// `contracted_ids`, this never throws — the ill-formed cases (an index that
// needs an explicit override) simply answer "no", since the override already
// dictates the mode.
auto realm_contracts(std::map<int, std::vector<IndexUse>> const& uses, int id)
    -> bool
{
    auto it = uses.find(id);
    if (it == uses.end())
        return false;
    auto const& us = it->second;
    switch (us.front().realm)
    {
        case Realm::Oblique:
            return us.size() == 2 && us[0].level != us[1].level;
        case Realm::Orthonormal: return us.size() == 2;
        case Realm::Collection:
        case Realm::Label: return false;
    }
    return false;
}

// A summed dummy that will be α-renamed to a canonical id.
struct Dummy final
{
    int id;
    SumMode mode;              // Default (realm-implicit) or Sum (explicit)
    Nf const* range = nullptr; // explicit symbolic range, else null
};

} // namespace

// Fold every subexpression forced to zero by a literal-0 operand down to a
// bare 0, and drop additive zeros, in one bottom-up pass.  A differentiated
// operand (∂_i e_j = 0 in a constant frame) or a projected-out basis leg leaves
// zeros nested inside ⊗ / contraction fences — e.g. `e_j ⊗ (0·∂_iε)` — which
// `encapsulate` has no arm for.  Collapsing them here turns a zero *term* into
// coeff 0 (so `collect_terms` erases it) and a zero *factor* inside a live term
// into a bare 0 that `multiplicative_flatten` folds into the coefficient — the
// algebraic zero law, applied before the all-`*` encapsulation runs.  Bottom-up
// so a deep zero (`Dot(0,·)` under a `⊗`) collapses its ancestors in the same
// pass; a genuine sum keeps its non-zero addend rather than vanishing.
auto fold_forced_zeros(Context& ctx, Expr const* e) -> Expr const*
{
    auto is0 = [](Expr const* z)
    {
        auto const* s = std::get_if<ScalarLiteral>(&z->node);
        return s && s->value.is_zero();
    };
    // Unqualified `Cross` / `Dot` / … resolve to this namespace's `nf` Factor
    // types; the Expr nodes and builders are the `tender::` ones.
    return rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* n) -> Expr const*
        {
            Expr const* zero = nullptr;
            auto to_zero = [&]() -> Expr const* {
                return zero ? zero :
                              (zero = tender::make_scalar(c, Rational{0}));
            };
            // Multiplicative / contraction nodes: a 0 operand forces 0.
            if (auto const* d = std::get_if<tender::Dot>(&n->node))
                return (is0(d->left) || is0(d->right)) ? to_zero() : n;
            if (auto const* d = std::get_if<tender::DDot>(&n->node))
                return (is0(d->left) || is0(d->right)) ? to_zero() : n;
            if (auto const* d = std::get_if<tender::DDotAlt>(&n->node))
                return (is0(d->left) || is0(d->right)) ? to_zero() : n;
            if (auto const* x = std::get_if<tender::Cross>(&n->node))
                return (is0(x->left) || is0(x->right)) ? to_zero() : n;
            if (auto const* p = std::get_if<tender::TensorProduct>(&n->node))
                return (is0(p->left) || is0(p->right)) ? to_zero() : n;
            if (auto const* q = std::get_if<tender::ScalarDiv>(&n->node))
                return is0(q->left) ? to_zero() : n; // 0 / s = 0
            // Linear unaries of 0 are 0.
            if (auto const* u = std::get_if<tender::Negate>(&n->node))
                return is0(u->operand) ? to_zero() : n;
            if (auto const* u = std::get_if<tender::Transpose>(&n->node))
                return is0(u->operand) ? to_zero() : n;
            if (auto const* u = std::get_if<tender::Trace>(&n->node))
                return is0(u->operand) ? to_zero() : n;
            if (auto const* u = std::get_if<tender::VectorInvariant>(&n->node))
                return is0(u->operand) ? to_zero() : n;
            // Additive zeros drop the vanishing addend (0 + x → x, x − 0 → x,
            // 0 − x → −x) — the enclosing node is *not* forced to zero.
            if (auto const* s = std::get_if<tender::Sum>(&n->node))
                return is0(s->left) ? s->right : is0(s->right) ? s->left : n;
            if (auto const* s = std::get_if<tender::Difference>(&n->node))
                return is0(s->right) ? s->left :
                       is0(s->left)  ? tender::make_negate(c, s->right) :
                                       n;
            return n;
        });
}

// ---- per-term lowering (passes 3+4+5) ----------------------------------

auto lower_term(Context& ctx, SignedExpr const& term) -> Term
{
    // 1. Strip explicit head binders, then push every operator through its ⊗
    //    fences so no ⊗ stays buried in an operand (the all-`*` model needs
    //    flat factors).  A rank-2 invariant of a dyad expands by definition
    //    (`tr(a⊗b) → a·b`, `vec(a⊗b) → a×b`, `(a⊗b)^T → b⊗a`); a double dot of
    //    dyads expands (`(a⊗b):(c⊗d) → (a·c)(b·d)`); then the single
    //    contractions `·` / `×` push through their adjacent ⊗ leg.  Each
    //    iterates to a fixpoint and never distributes over a genuine sum.
    std::vector<RawBinder> binders;
    auto const* body = strip_binders(term.body, binders);
    auto const* distributed = fold_forced_zeros(
        ctx,
        steps::distribute_contraction(
            ctx,
            steps::expand_double_dot(ctx, steps::expand_dyad_ops(ctx, body))));

    // 2. Census the free index occurrences (for mode classification), and
    //    collect the term's bound indices:
    //      - implicit realm contractions  → Default (α-renamed);
    //      - an explicit Σ                → Default if it merely confirms the
    //        realm default, else Sum      (α-renamed);
    //      - a NoSum suppressing a real contraction → a free override, kept
    //        with its original id (not α-renamed); a redundant NoSum is
    //        dropped.
    std::map<int, std::vector<IndexUse>> uses;
    collect_term_uses(distributed, {}, uses);
    std::set<int> explicit_ids;
    for (auto const& b: binders)
        explicit_ids.insert(b.id);

    std::vector<Dummy> dummies;
    for (int id: contracted_ids(distributed, explicit_ids))
        dummies.push_back({id, SumMode::Default});
    std::vector<BoundIndex> nosum_free;
    for (auto const& b: binders)
    {
        bool const c = realm_contracts(uses, b.id);
        if (b.kind == BinderKind::Sum)
        {
            // A ranged Σ is always an explicit Sum (its range makes it so);
            // an unranged Σ is Default when it merely confirms the realm rule.
            if (b.range)
                dummies.push_back(
                    {b.id, SumMode::Sum, canonicalize_nf(ctx, b.range)});
            else
                dummies.push_back(
                    {b.id, c ? SumMode::Default : SumMode::Sum, nullptr});
        }
        else if (c)
            nosum_free.push_back({CountableIndex{b.id}, SumMode::NoSum});
    }
    std::sort(
        nosum_free.begin(),
        nosum_free.end(),
        [](BoundIndex const& x, BoundIndex const& y)
        { return x.index.id < y.index.id; });
    // Deterministic base order for the (k > 6) fallback that skips the search.
    std::sort(
        dummies.begin(),
        dummies.end(),
        [](Dummy const& x, Dummy const& y) { return x.id < y.id; });

    // 3. α-canonicalize the summed dummies: assign canonical (negative) ids,
    //    choosing the permutation that minimizes the resulting term under
    //    `compare` (Fubini — the binders are interchangeable).  Substitution is
    //    at the Expr level; minimization at the Nf level.  Then flatten +
    //    place.
    int const k = static_cast<int>(dummies.size());
    std::vector<int> order(static_cast<std::size_t>(k));
    std::iota(order.begin(), order.end(), 0);
    bool const search = k <= 6; // k! permutations; tall stacks keep their order
    Term best;
    bool have_best = false;
    do
    {
        std::map<int, int> remap;
        std::vector<BoundIndex> bound;
        bound.reserve(static_cast<std::size_t>(k) + nosum_free.size());
        for (int p = 0; p < k; ++p)
        {
            auto const& d = dummies[static_cast<std::size_t>(
                order[static_cast<std::size_t>(p)])];
            int const cid = bound_canon_id(p);
            remap[d.id] = cid;
            bound.push_back({CountableIndex{cid}, d.mode, d.range});
        }
        bound.insert(bound.end(), nosum_free.begin(), nosum_free.end());

        auto const* renamed = substitute_index_ids(ctx, distributed, remap);
        auto pp = multiplicative_flatten(SignedExpr{term.sign, renamed});
        Term cand = place_factors(ctx, pp);
        cand.bound = std::move(bound);
        if (!have_best || compare(cand, best) < 0)
        {
            best = std::move(cand);
            have_best = true;
        }
    } while (search && std::next_permutation(order.begin(), order.end()));
    return best;
}

// ---- pass 6: like-term collection + term-set ordering (C9) --------------

auto collect_terms(std::vector<Term> terms) -> std::vector<Term>
{
    // Sort by the like-term key so equal keys are adjacent; their `coeff`s then
    // merge in one linear sweep.  (Key order ignores `coeff`, and the surviving
    // merged terms have distinct keys, so this is also the canonical term-set
    // order — the `coeff` tiebreak in `compare` never separates two of them.)
    std::stable_sort(
        terms.begin(),
        terms.end(),
        [](Term const& x, Term const& y)
        { return compare_term_key(x, y) < 0; });

    std::vector<Term> out;
    out.reserve(terms.size());
    for (auto& t: terms)
    {
        if (!out.empty() && compare_term_key(out.back(), t) == 0)
            out.back().coeff += t.coeff; // 2a + 3a → 5a
        else
            out.push_back(std::move(t));
    }
    // Cancellation: a merged zero coefficient means the term vanishes.
    std::erase_if(out, [](Term const& t) { return t.coeff == Rational{0}; });
    return out;
}

// ---- binder sinking (keep the additive layer above the binders) --------

namespace
{

// Rebuild a binder of the same kind as `binder` around `body`.
auto rewrap_binder(Context& ctx, Expr const* binder, Expr const* body)
    -> Expr const*
{
    if (auto const* s = std::get_if<ExplicitSum>(&binder->node))
        return make_explicit_sum(ctx, s->index, body, s->bound);
    auto const& s = std::get<NoSum>(binder->node);
    return make_no_sum(ctx, s.index, body);
}

// Push a binder over an *additive* body to each addend — summation is linear:
//   Σ_i(X + Y) → Σ_iX + Σ_iY,  Σ_i(X − Y) → Σ_iX − Σ_iY,  Σ_i(−X) → −Σ_iX.
// A binder over a non-additive body (a product / contraction / atom) wraps it
// unchanged.  `binder` carries the kind + index to re-emit.
auto distribute_binder(Context& ctx, Expr const* binder, Expr const* body)
    -> Expr const*
{
    if (auto const* s = std::get_if<Sum>(&body->node))
        return make_sum(
            ctx,
            distribute_binder(ctx, binder, s->left),
            distribute_binder(ctx, binder, s->right));
    if (auto const* d = std::get_if<Difference>(&body->node))
        return make_difference(
            ctx,
            distribute_binder(ctx, binder, d->left),
            distribute_binder(ctx, binder, d->right));
    if (auto const* n = std::get_if<Negate>(&body->node))
        return make_negate(ctx, distribute_binder(ctx, binder, n->operand));
    return rewrap_binder(ctx, binder, body);
}

// Sink every summation binder below the additive layer, so the result is a
// (possibly nested) Sum/Difference/Negate tree of binder-headed *terms* — the
// shape `additive_flatten` needs to split a `Σ_i(X+Y)` into separate terms.
// Binders buried inside a product/contraction are left in place (they become a
// term's bound set or a `Paren`).
auto sink_binders(Context& ctx, Expr const* e) -> Expr const*
{
    return visit(
        Overloads{
            [&](Sum const& s) -> Expr const*
            {
                return make_sum(
                    ctx, sink_binders(ctx, s.left), sink_binders(ctx, s.right));
            },
            [&](Difference const& d) -> Expr const*
            {
                return make_difference(
                    ctx, sink_binders(ctx, d.left), sink_binders(ctx, d.right));
            },
            [&](Negate const& n) -> Expr const*
            { return make_negate(ctx, sink_binders(ctx, n.operand)); },
            [&](ExplicitSum const&) -> Expr const*
            {
                auto const& s = std::get<ExplicitSum>(e->node);
                return distribute_binder(ctx, e, sink_binders(ctx, s.body));
            },
            [&](NoSum const&) -> Expr const*
            {
                auto const& s = std::get<NoSum>(e->node);
                return distribute_binder(ctx, e, sink_binders(ctx, s.body));
            },
            // Products / contractions / atoms: a binder inside stays put.
            [&](auto const&) -> Expr const* { return e; },
        },
        *e);
}

} // namespace

// ---- entry point: lower `Expr → Nf` (C10) ------------------------------

auto canonicalize_nf(Context& ctx, Expr const* e) -> Nf const*
{
    // Sink summation binders below the additive layer (`Σ_i(X+Y) → Σ_iX +
    // Σ_iY`), so the outermost layer is a sum of binder-headed terms.  Then
    // expand that additive layer into signed terms, lower each (multiplicative
    // flatten + encapsulate + region placement + summation resolution), and
    // collect like terms.  A genuine-sum factor *inside* a product recurses
    // back through `encapsulate` → `make_paren`.
    std::vector<Term> lowered;
    for (auto const& st: additive_flatten(sink_binders(ctx, e)))
        lowered.push_back(lower_term(ctx, st));
    return make_nf(ctx, collect_terms(std::move(lowered)));
}

// ---- raise: Nf → Expr (C12) --------------------------------------------

namespace
{

// Rebuild an `Expr` from a `Factor`.  Composites recurse; an `Atom` re-wraps
// its stored `TensorObject` (traits and slots preserved); a `Paren` raises its
// sub-`Nf`.  Contraction / cross chains are rebuilt left-associated — the
// interface theorem makes the bracketing immaterial, so re-lowering recovers
// the same flat factor.
auto raise_factor(Context& ctx, Factor const& f) -> Expr const*
{
    return visit(
        Overloads{
            [&](Atom const& a) -> Expr const* { return ctx.make<Expr>(a.obj); },
            [&](Contraction const& c) -> Expr const*
            {
                auto join =
                    [&](COp op, Expr const* l, Expr const* r) -> Expr const*
                {
                    switch (op)
                    {
                        case COp::Dot: return make_dot(ctx, l, r);
                        case COp::DDot: return make_ddot(ctx, l, r);
                        case COp::DDotAlt: return make_ddot_alt(ctx, l, r);
                    }
                    return make_dot(ctx, l, r); // unreachable
                };
                Expr const* acc = raise_factor(ctx, *c.factors[0]);
                for (std::size_t i = 0; i + 1 < c.factors.size(); ++i)
                    acc = join(
                        c.ops[i], acc, raise_factor(ctx, *c.factors[i + 1]));
                return acc;
            },
            [&](Cross const& c) -> Expr const*
            {
                Expr const* acc = raise_factor(ctx, *c.factors[0]);
                for (std::size_t i = 1; i < c.factors.size(); ++i)
                    acc =
                        make_cross(ctx, acc, raise_factor(ctx, *c.factors[i]));
                return acc;
            },
            [&](Paren const& p) -> Expr const* { return raise(ctx, *p.body); },
            [&](Div const& d) -> Expr const* {
                return make_scalar_div(
                    ctx, raise(ctx, *d.num), raise(ctx, *d.den));
            },
            [&](Unary const& u) -> Expr const*
            {
                auto const* operand = raise_factor(ctx, *u.operand);
                switch (u.op)
                {
                    case UnaryOp::Trace: return make_trace(ctx, operand);
                    case UnaryOp::VectorInvariant:
                        return make_vector_invariant(ctx, operand);
                    case UnaryOp::Transpose:
                        return make_transpose(ctx, operand);
                }
                return operand; // unreachable
            },
            // Scalar fields: rebuild the Expr node (the qualified `tender::`
            // factories — the unqualified names are this namespace's Factor
            // builders) over raised operand Nfs.
            [&](ScalarFn const& s) -> Expr const* {
                return tender::make_scalar_fn(
                    ctx, s.kind, raise(ctx, *s.operand));
            },
            [&](Pow const& p) -> Expr const* {
                return tender::make_pow(
                    ctx, raise(ctx, *p.base), raise(ctx, *p.exponent));
            },
            // ∂ operator (vibe 000077): rebuild the surface Deriv over its
            // wrt-object wrapped back into an Expr.
            [&](Deriv const& d) -> Expr const*
            { return tender::make_deriv(ctx, ctx.make<Expr>(d.wrt)); },
            // ∇ operator (vibe 000078): rebuild the surface Nabla.
            [&](Nabla const&) -> Expr const*
            { return tender::make_nabla(ctx); },
        },
        f);
}

// Rebuild one term: the ⊗-product of its factors (scalars then tensors), with
// the coefficient as a leading literal / `Negate`, and the bound indices as
// head binders.  A `Default` (realm-implicit) index is materialized as an
// `ExplicitSum` exactly like a `Sum` override — the raised `Expr` then carries
// the same explicit binders the existing `Expr` pipeline (materialize / canon /
// reassemble / unroll_sums) expects, and lowering re-classifies it back to
// `Default` (so `canonicalize_nf(raise(nf)) == nf` still holds).  Only a
// `NoSum` differs, becoming a `NoSum` binder.
auto raise_term(Context& ctx, Term const& t) -> Expr const*
{
    Expr const* body = nullptr;
    auto append = [&](Expr const* x)
    { body = body ? make_tensor_product(ctx, body, x) : x; };
    for (auto const* f: t.scalars)
        append(raise_factor(ctx, *f));
    for (auto const* f: t.tensors)
        append(raise_factor(ctx, *f));

    bool const neg = t.coeff < Rational{0};
    Rational const mag = neg ? -t.coeff : t.coeff;
    // Emit the magnitude unless it is a redundant unit factor on a non-empty
    // product (a bare term keeps its `1` so it survives as `ScalarLiteral{1}`).
    if (!(mag == Rational{1}) || !body)
        body = body ? make_tensor_product(ctx, make_scalar(ctx, mag), body) :
                      make_scalar(ctx, mag);
    if (neg)
        body = make_negate(ctx, body);

    for (auto it = t.bound.rbegin(); it != t.bound.rend(); ++it)
    {
        if (it->mode == SumMode::NoSum)
            body = make_no_sum(ctx, it->index, body);
        else // Default or Sum → materialized ExplicitSum binder (with its
             // range)
            body = make_explicit_sum(
                ctx,
                it->index,
                body,
                it->range ? raise(ctx, *it->range) : nullptr);
    }
    return body;
}

} // namespace

auto raise(Context& ctx, Nf const& nf) -> Expr const*
{
    if (nf.terms.empty())
        return make_scalar(ctx, Rational{0});
    Expr const* acc = nullptr;
    for (auto const& t: nf.terms)
    {
        auto const* term = raise_term(ctx, t);
        acc = acc ? make_sum(ctx, acc, term) : term;
    }
    return acc;
}

} // namespace tender::nf
