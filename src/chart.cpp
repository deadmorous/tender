#include <tender/chart.hpp>

#include <tender/context.hpp>
#include <tender/derivation.hpp>
#include <tender/nf_lower.hpp>
#include <tender/rewrite.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace tender
{

namespace
{

// Flatten a TensorProduct tree into its leaf factors.
void flatten(Expr const* e, std::vector<Expr const*>& out)
{
    if (auto const* p = std::get_if<TensorProduct>(&e->node))
    {
        flatten(p->left, out);
        flatten(p->right, out);
    }
    else
        out.push_back(e);
}

auto is_scalar(Expr const* e) -> bool
{
    return infer_rank(e) == std::optional<int>{0};
}

// Left-fold factors into a TensorProduct (one factor returns itself, none
// returns the scalar 1).
auto product_of(Context& ctx, std::vector<Expr const*> const& factors)
    -> Expr const*
{
    Expr const* p = nullptr;
    for (auto const* f: factors)
        p = p ? make_tensor_product(ctx, p, f) : f;
    return p ? p : make_scalar(ctx, Rational{1});
}

// Bilinearly expand a contraction `op` (· or ×) of (sums/differences/negations
// of) vector terms: (A + B) op C → (A op C) + (B op C), so the leaf
// basis-vector contractions surface for the basis steps.  A leaf-vs-leaf pair
// stays a single `op` node.
using BinOp = Expr const* (*)(Context&, Expr const*, Expr const*);

auto distribute_bilinear(Context& ctx, Expr const* l, Expr const* r, BinOp op)
    -> Expr const*
{
    if (auto const* s = std::get_if<Sum>(&l->node))
        return make_sum(
            ctx,
            distribute_bilinear(ctx, s->left, r, op),
            distribute_bilinear(ctx, s->right, r, op));
    if (auto const* d = std::get_if<Difference>(&l->node))
        return make_difference(
            ctx,
            distribute_bilinear(ctx, d->left, r, op),
            distribute_bilinear(ctx, d->right, r, op));
    if (auto const* n = std::get_if<Negate>(&l->node))
        return make_negate(ctx, distribute_bilinear(ctx, n->operand, r, op));
    // A scalar quotient commutes with the contraction: (X/s) op r → (X op r)/s
    // — so a symmetrized term like (A + Aᵀ)/2 does not fence the reduction
    // (vibe 000075).
    if (auto const* q = std::get_if<ScalarDiv>(&l->node))
        return make_scalar_div(
            ctx, distribute_bilinear(ctx, q->left, r, op), q->right);
    if (auto const* s = std::get_if<Sum>(&r->node))
        return make_sum(
            ctx,
            distribute_bilinear(ctx, l, s->left, op),
            distribute_bilinear(ctx, l, s->right, op));
    if (auto const* d = std::get_if<Difference>(&r->node))
        return make_difference(
            ctx,
            distribute_bilinear(ctx, l, d->left, op),
            distribute_bilinear(ctx, l, d->right, op));
    if (auto const* n = std::get_if<Negate>(&r->node))
        return make_negate(ctx, distribute_bilinear(ctx, l, n->operand, op));
    if (auto const* q = std::get_if<ScalarDiv>(&r->node))
        return make_scalar_div(
            ctx, distribute_bilinear(ctx, l, q->left, op), q->right);
    return op(ctx, l, r);
}

// Divide a vector field (sum of scalar ⊗ frame-vector terms) by the scalar h:
// recurse over the additive structure and, at each leaf term, fold the scalar
// coefficient into a quotient (scalar)/h while keeping the frame vector leg.
auto divide_vector(Context& ctx, Expr const* vec, Expr const* h) -> Expr const*
{
    if (auto const* s = std::get_if<Sum>(&vec->node))
        return make_sum(
            ctx,
            divide_vector(ctx, s->left, h),
            divide_vector(ctx, s->right, h));
    if (auto const* d = std::get_if<Difference>(&vec->node))
        return make_difference(
            ctx,
            divide_vector(ctx, d->left, h),
            divide_vector(ctx, d->right, h));
    if (auto const* n = std::get_if<Negate>(&vec->node))
        return make_negate(ctx, divide_vector(ctx, n->operand, h));

    std::vector<Expr const*> facs;
    flatten(vec, facs);
    std::vector<Expr const*> scalars;
    std::vector<Expr const*> legs;
    for (auto const* f: facs)
        (is_scalar(f) ? scalars : legs).push_back(f);
    Expr const* coeff = make_scalar_div(ctx, product_of(ctx, scalars), h);
    legs.insert(legs.begin(), coeff);
    return product_of(ctx, legs);
}

// The integer square root of n if n is a perfect square, else nullopt.
auto perfect_isqrt(std::int64_t n) -> std::optional<std::int64_t>
{
    if (n < 0)
        return std::nullopt;
    auto r = static_cast<std::int64_t>(std::llround(std::sqrt((double)n)));
    for (auto c: {r - 1, r, r + 1})
        if (c >= 0 && c * c == n)
            return c;
    return std::nullopt;
}

// √g_ii taken as the positive root under the chart's domain convention
// (decision 2/3): factor the radicand into base^e with a numeric coefficient,
// and if every exponent is even and the coefficient a perfect square, return
// the product of the positive half-powers (e.g. r² → r, r² sin²θ → r sin θ).
// Otherwise fall back to a symbolic √ pushed through simplify_scalars.
auto positive_sqrt(Context& ctx, Expr const* g) -> Expr const*
{
    std::vector<Expr const*> facs;
    flatten(g, facs);
    Rational coeff{1};
    std::vector<std::pair<Expr const*, std::int64_t>> bases;
    auto add_base = [&](Expr const* base, std::int64_t exp)
    {
        for (auto& kv: bases)
            if (structural_eq(kv.first, base))
            {
                kv.second += exp;
                return;
            }
        bases.push_back({base, exp});
    };
    bool clean = true;
    for (auto const* f: facs)
    {
        if (auto const* s = std::get_if<ScalarLiteral>(&f->node))
            coeff = coeff * s->value;
        else if (auto const* p = std::get_if<Pow>(&f->node))
        {
            auto const* ne = std::get_if<ScalarLiteral>(&p->exponent->node);
            if (ne && ne->value.is_integer())
                add_base(p->base, ne->value.num());
            else
            {
                clean = false;
                add_base(f, 1);
            }
        }
        else
            add_base(f, 1);
    }
    auto root_num = perfect_isqrt(coeff.num());
    auto root_den = perfect_isqrt(coeff.den());
    for (auto const& kv: bases)
        if (kv.second % 2 != 0)
            clean = false;
    if (clean && root_num && root_den)
    {
        std::vector<Expr const*> out;
        Rational const root{*root_num, *root_den};
        if (!(root == Rational{1}))
            out.push_back(make_scalar(ctx, root));
        for (auto const& kv: bases)
        {
            auto const half = kv.second / 2;
            if (half == 1)
                out.push_back(kv.first);
            else if (half != 0)
                out.push_back(
                    make_pow(ctx, kv.first, make_scalar(ctx, Rational{half})));
        }
        return product_of(ctx, out);
    }
    return steps::simplify_scalars(
        ctx, make_scalar_fn(ctx, ScalarFnKind::Sqrt, g));
}

// The invariant dot u·v of two tensors expressed in the orthonormal reference
// frame `ref`, reduced: distribute the dot bilinearly over the sums so the
// basis-vector dots surface, turn the concrete frame-vector legs into Kronecker
// deltas, evaluate them, and simplify.  For two vectors the result is a scalar.
auto reduce_dot(Context& ctx, Basis const& ref, Expr const* u, Expr const* v)
    -> Expr const*
{
    Expr const* e = distribute_bilinear(ctx, u, v, &make_dot);
    // A dot with a zero operand is zero — a fully projected-out operand (e.g.
    // the row e_r·(e_θ⊗e_θ) → 0 fed back in by component_matrix) makes 0·e_j,
    // which is malformed for encapsulate.  Fold it first, as reduce_cross
    // does for 0×b (vibe 000074).
    e = rewrite_tree(
        ctx,
        e,
        [](Context& c, Expr const* node) -> Expr const*
        {
            auto const* d = std::get_if<Dot>(&node->node);
            if (!d)
                return node;
            auto is_zero = [](Expr const* z)
            {
                auto const* s = std::get_if<ScalarLiteral>(&z->node);
                return s && s->value.is_zero();
            };
            return (is_zero(d->left) || is_zero(d->right)) ?
                       make_scalar(c, Rational{0}) :
                       node;
        });
    // Push the dot through any ⊗ fence it now straddles: a connection term such
    // as e_θ·∂_θ(e_θ⊗e_r) → e_θ·((−e_r)⊗e_r) leaves a raw ⊗ inside the dot
    // operand, which canonicalize's encapsulate rejects.  Fence-distribute
    // e_θ·(u⊗v) → (e_θ·u) v before the basis reduction (vibe 000073).
    e = steps::distribute_contraction(ctx, e);
    e = simplify_basis_dot(ctx, e, ref);
    e = steps::canonicalize(ctx, e);
    e = steps::eval_delta_concrete(ctx, e);
    e = steps::fold_arithmetic(ctx, e);
    e = steps::canonicalize(ctx, e);
    return steps::simplify_scalars(ctx, e);
}

// The invariant cross u×v reduced in the frame `fb`, symbolically: distribute
// bilinearly, turn each frame-vector cross into its Levi-Civita expansion
// (s e_i)×(t e_j) → s t √g ε_{ijk} e^k, materialise the Σ_k, evaluate the
// concrete ε, and simplify.  Works on the symbolic e_i atoms (vibe 000071), so
// e_r × e_θ = e_z with no reference frame.
auto reduce_cross(Context& ctx, Basis const& fb, Expr const* u, Expr const* v)
    -> Expr const*
{
    Expr const* e = distribute_bilinear(ctx, u, v, &make_cross);
    // A cross with a zero operand is zero — the differentiator emits 0×b + a×0
    // when it Leibniz-differentiates a cross of constants; fold it before the
    // basis reduction so the term vanishes cleanly.
    e = rewrite_tree(
        ctx,
        e,
        [](Context& c, Expr const* node) -> Expr const*
        {
            auto const* x = std::get_if<Cross>(&node->node);
            if (!x)
                return node;
            auto is_zero = [](Expr const* z)
            {
                auto const* s = std::get_if<ScalarLiteral>(&z->node);
                return s && s->value.is_zero();
            };
            return (is_zero(x->left) || is_zero(x->right)) ?
                       make_scalar(c, Rational{0}) :
                       node;
        });
    e = simplify_basis_cross(ctx, e, fb);
    e = steps::canonicalize(ctx, e);
    e = steps::unroll_sums(ctx, e);
    e = steps::eval_eps_concrete(ctx, e);
    e = steps::fold_arithmetic(ctx, e);
    e = steps::canonicalize(ctx, e);
    return steps::simplify_scalars(ctx, e);
}

// The coefficients γ^k_{ij} of ∂_{q^j} e_i = Σ_k γ^k_{ij} e_k, given a
// precomputed physical basis `pb` (so callers that need many of them — the
// operators — do not rebuild the frame each time).  e_i lives in the constant
// reference frame, so ∂ acts on its scalar coefficients alone; the orthonormal
// projection (∂_j e_i)·e_k gives the k-th coefficient.  A vanishing derivative
// (rank-0 scalar 0) yields all zeros.
auto connection_at(
    Context& ctx, CoordinateChart const& chart, Basis const& pb, int i, int j)
    -> std::vector<Expr const*>
{
    int const n = static_cast<int>(chart.coords.size());
    Expr const* de = steps::simplify_scalars(
        ctx, steps::partial(ctx, pb.basis(i), chart.coords[j]));
    if (infer_rank(de) == std::optional<int>{0})
        return std::vector<Expr const*>(n, make_scalar(ctx, Rational{0}));
    std::vector<Expr const*> out;
    out.reserve(n);
    for (int k = 0; k < n; ++k)
        out.push_back(reduce_dot(ctx, chart.reference, de, pb.basis(k)));
    return out;
}

// (1/h) ⊗ x as a scalar quotient, simplified.
auto over_h(Context& ctx, Expr const* x, Expr const* h) -> Expr const*
{
    return steps::simplify_scalars(ctx, make_scalar_div(ctx, x, h));
}

// The directional derivative (1/h_i) ∂_{q^i} T scaled out: the scalar 1/h_i.
auto inv_h(Context& ctx, Expr const* h) -> Expr const*
{
    return over_h(ctx, make_scalar(ctx, Rational{1}), h);
}

} // namespace

void validate_chart(CoordinateChart const& chart)
{
    int const n = static_cast<int>(chart.coords.size());
    if (n == 0)
        throw std::invalid_argument("CoordinateChart: no coordinates given");
    if (static_cast<int>(chart.embedding.size()) != chart.reference.dim())
        throw std::invalid_argument(
            "CoordinateChart: embedding must have one component per reference "
            "direction (embedding.size() == reference.dim())");
    int chart_id = 0;
    for (int i = 0; i < n; ++i)
    {
        auto const* t = std::get_if<TensorObject>(&chart.coords[i]->node);
        if (!t || !t->traits || !t->traits->coordinate)
            throw std::invalid_argument(
                "CoordinateChart: every coord must be a coordinate() atom");
        auto const& cr = *t->traits->coordinate;
        if (i == 0)
            chart_id = cr.chart_id;
        else if (cr.chart_id != chart_id)
            throw std::invalid_argument(
                "CoordinateChart: all coords must share one chart_id");
        if (cr.slot != i)
            throw std::invalid_argument(
                "CoordinateChart: coords must occupy slots 0..n-1 in list order "
                "(coord i must have slot i); use Workspace.coords() to mint them "
                "with slots assigned automatically");
    }
}

auto radius_vector(Context& ctx, CoordinateChart const& chart) -> Expr const*
{
    Expr const* R = nullptr;
    for (int a = 0; a < chart.reference.dim(); ++a)
    {
        Expr const* term = make_tensor_product(
            ctx, chart.embedding[a], chart.reference.basis(a));
        R = R ? make_sum(ctx, R, term) : term;
    }
    if (!R)
        throw std::invalid_argument("radius_vector: empty embedding");
    return steps::canonicalize(ctx, R);
}

auto tangent_vector(Context& ctx, CoordinateChart const& chart, int i)
    -> Expr const*
{
    if (i < 0 || i >= static_cast<int>(chart.coords.size()))
        throw std::out_of_range(
            "tangent_vector: coordinate index out of range");
    return steps::partial(ctx, radius_vector(ctx, chart), chart.coords[i]);
}

auto metric_component(Context& ctx, CoordinateChart const& chart, int i, int j)
    -> Expr const*
{
    Expr const* gi = tangent_vector(ctx, chart, i);
    Expr const* gj = tangent_vector(ctx, chart, j);
    return reduce_dot(ctx, chart.reference, gi, gj);
}

auto scale_factor(Context& ctx, CoordinateChart const& chart, int i)
    -> Expr const*
{
    return positive_sqrt(ctx, metric_component(ctx, chart, i, i));
}

// The chart these coordinates belong to (all coords share one chart_id).
auto chart_id_of(CoordinateChart const& chart) -> int
{
    auto const* c0 = std::get_if<TensorObject>(&chart.coords.front()->node);
    if (!c0 || !c0->traits || !c0->traits->coordinate)
        throw std::invalid_argument(
            "chart coordinate is not a coordinate variable");
    return c0->traits->coordinate->chart_id;
}

// A structural fingerprint of a chart: its coordinate and embedding Expr
// pointers, which are stable under hash-consing, so two charts with the same
// geometry share one fingerprint and two different geometries never do (vibe
// 000072 Obs 2).
auto chart_fingerprint(CoordinateChart const& chart) -> std::vector<Expr const*>
{
    std::vector<Expr const*> fp;
    fp.reserve(chart.coords.size() + chart.embedding.size());
    fp.insert(fp.end(), chart.coords.begin(), chart.coords.end());
    fp.insert(fp.end(), chart.embedding.begin(), chart.embedding.end());
    return fp;
}

auto physical_basis(Context& ctx, CoordinateChart const& chart) -> Basis
{
    // Idempotent per chart (vibe 000073 Gap 3): reuse the cached frame so that
    // physical_basis and physical_frame return the *same* Basis identity.  Each
    // e_i atom carries the basis id in its slot tag, so a fresh build would
    // mint structurally distinct e_r — silently defeating simplify_basis_dot
    // when a field lives on one and an operator result on the other.  Keyed by
    // chart id, validated by the geometry fingerprint (vibe 000072 Obs 2).
    int const chart_id = chart_id_of(chart);
    auto fingerprint = chart_fingerprint(chart);
    if (auto const* cf = ctx.chart_frame(chart_id);
        cf && cf->fingerprint == fingerprint)
        if (auto const* b = ctx.basis(cf->basis_id))
            return *b;

    std::vector<Expr const*> vectors;
    std::vector<IndexName> value_names;
    for (int i = 0; i < static_cast<int>(chart.coords.size()); ++i)
    {
        Expr const* gi = tangent_vector(ctx, chart, i);
        Expr const* hi = scale_factor(ctx, chart, i);
        vectors.push_back(
            steps::simplify_scalars(ctx, divide_vector(ctx, gi, hi)));
        auto const* t = std::get_if<TensorObject>(&chart.coords[i]->node);
        if (!t)
            throw std::invalid_argument(
                "physical_basis: coordinate is not a "
                "coordinate variable");
        value_names.push_back(make_index_name(t->name.v.view()));
    }
    // vibe 000072 Obs 1/8: when the physical frame coincides with the reference
    // basis — an identity/Cartesian chart where e_i == reference.basis(i) for
    // every i — reuse the reference basis itself.  Then the frame prints under
    // the reference's own vector names (i, j, k) and shares its basis id, so
    // express / to_reference / grad all agree and a completed resolution of
    // identity folds without a naming split.  A genuinely curvilinear frame
    // (e_r ≠ i) falls through to a fresh "e" basis.
    if (static_cast<int>(vectors.size()) == chart.reference.dim())
    {
        bool coincides = true;
        for (int i = 0; i < static_cast<int>(vectors.size()); ++i)
            if (!structural_eq(
                    steps::canonicalize(ctx, vectors[i]),
                    chart.reference.basis(i)))
            {
                coincides = false;
                break;
            }
        if (coincides)
        {
            ctx.register_chart_frame(
                chart_id, chart.reference.basis_id(), std::move(fingerprint));
            return chart.reference;
        }
    }
    Basis const basis = make_orthonormal_basis(
        ctx,
        chart.reference.space(),
        std::move(vectors),
        make_tensor_name("e"),
        Handedness::Right,
        BasisNaming{.value_names = std::move(value_names)});
    ctx.register_chart_frame(chart_id, basis.basis_id(), std::move(fingerprint));
    return basis;
}

auto basis_derivative(Context& ctx, CoordinateChart const& chart, int i, int j)
    -> Expr const*
{
    int const n = static_cast<int>(chart.coords.size());
    if (i < 0 || i >= n || j < 0 || j >= n)
        throw std::out_of_range("basis_derivative: index out of range");
    // e_i lives in the constant reference frame, so ∂_{q^j} differentiates only
    // its scalar coefficients (the reference vectors are constant → 0).
    Expr const* ei = physical_basis(ctx, chart).basis(i);
    return steps::simplify_scalars(
        ctx, steps::partial(ctx, ei, chart.coords[j]));
}

auto connection_coefficients(
    Context& ctx,
    CoordinateChart const& chart,
    int i,
    int j) -> std::vector<Expr const*>
{
    int const n = static_cast<int>(chart.coords.size());
    if (i < 0 || i >= n || j < 0 || j >= n)
        throw std::out_of_range("connection_coefficients: index out of range");
    return connection_at(ctx, chart, physical_basis(ctx, chart), i, j);
}

auto physical_frame(Context& ctx, CoordinateChart const& chart) -> Basis
{
    int const n = static_cast<int>(chart.coords.size());

    // physical_basis is the cached, idempotent frame builder (vibe 000073 Gap
    // 3), so the user's fields and the operators share the same e_i atoms.
    // physical_frame adds the connection table on top; skip rebuilding it if it
    // is already registered for this basis.
    Basis const basis = physical_basis(ctx, chart);
    if (ctx.connection(basis.basis_id()) != nullptr)
        return basis;
    int const chart_id = chart_id_of(chart);

    // Pre-express ∂_{q^j} e_i on the symbolic e_k atoms: Σ_k γ^k_{ij} e_k, with
    // the γ from the connection (M5).  The e_k are this basis's own direction
    // atoms, so the derivative stays intrinsic to the frame.
    auto* conn = ctx.make<BasisConnection>();
    conn->chart_id = chart_id;
    conn->basis_id = basis.basis_id();
    auto const vals = basis.space()->values();
    conn->values.assign(vals.begin(), vals.end());
    // The reference-frame (WCS) expansion of each frame vector, e.g.
    // e_r = cos θ i + sin θ j — physical_basis's concrete vectors (vibe 000071
    // P4).
    conn->reference_expansion.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        conn->reference_expansion.push_back(basis.basis(i));
    conn->deriv.assign(n, std::vector<Expr const*>(n, nullptr));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            auto const gamma = connection_coefficients(ctx, chart, i, j);
            Expr const* d = nullptr;
            for (int k = 0; k < n; ++k)
            {
                if (auto const* s = std::get_if<ScalarLiteral>(&gamma[k]->node);
                    s && s->value.is_zero())
                    continue;
                Expr const* term =
                    make_tensor_product(ctx, gamma[k], basis.direction(ctx, k));
                d = d ? make_sum(ctx, d, term) : term;
            }
            conn->deriv[i][j] =
                d ? steps::simplify_scalars(ctx, steps::canonicalize(ctx, d)) :
                    make_scalar(ctx, Rational{0});
        }
    ctx.register_connection(conn, basis.basis_id());
    return basis;
}

// ---- differential operators (vibe 000071, intrinsic) -------------------
//
// ∇ = Σ_i (1/h_i) e_i ∂_{q^i}, applied *intrinsically in the chart's own
// physical frame* — the symbolic e_i atoms, whose derivatives come from the
// connection (∂_j e_i = γ^k_{ij} e_k, vibe 000071 P2), never expanded in WCS.
// Each operator is
//
//     ∇ ⊙ T = Σ_i (1/h_i) e_i ⊙ ∂_{q^i} T          (⊙ = ⊗, ·, or ×)
//
// with ∂_{q^i} acting by Leibniz on the curvilinear components (fields) and the
// basis vectors (connection), and the result staying on e_i.  grad raises the
// rank by one, div lowers it by one (via e_i·e_j = δ_ij / g_ij), rot keeps it.

// Expand every identity tensor I in `e` into the frame's resolution of identity
// Σ_k e_k⊗e_k on the symbolic direction atoms (vibe 000071), so a field
// carrying I (e.g. R×I) reduces to clean dyad form.
auto expand_identity_frame(Context& ctx, Basis const& fb, Expr const* e)
    -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&node->node);
            if (!t || !t->traits
                || t->traits->well_known
                       != std::optional{WellKnownKind::Identity})
                return node;
            Expr const* sum = nullptr;
            for (int k = 0; k < fb.dim(); ++k)
            {
                Expr const* d = make_tensor_product(
                    c, fb.direction(c, k), fb.direction(c, k));
                sum = sum ? make_sum(c, sum, d) : d;
            }
            return sum ? sum : node;
        });
}

// Bring a field into clean dyad form on the frame before an operator
// differentiates it (vibe 000071): expand any identity tensor, then reduce
// every frame-vector cross to frame vectors via its Levi-Civita expansion — so
// a field built with I or an unreduced cross (R×I) does not trip the
// differentiator or canonicalize.  A fixpoint over fence-distribution +
// basis-cross reduction.
auto reduce_field(Context& ctx, Basis const& fb, Expr const* T) -> Expr const*
{
    T = expand_identity_frame(ctx, fb, T);
    T = steps::expand_products(ctx, T);
    Expr const* prev = nullptr;
    for (int guard = 0; T != prev && guard <= fb.dim() + 2; ++guard)
    {
        prev = T;
        T = steps::distribute_contraction(ctx, T); // fence a×(u⊗v) → (a×u)⊗v
        T = simplify_basis_cross(ctx, T, fb); // frame crosses → √g ε e^k
        T = simplify_basis_dot(ctx, T, fb); // frame dots → δ (a tr ε operand
                                            // leaves e_i·e_j, vibe 000075)
        T = steps::canonicalize(ctx, T);
        T = steps::unroll_sums(ctx, T);
        T = steps::eval_eps_concrete(ctx, T);
        T = steps::eval_delta_concrete(ctx, T);
        T = steps::fold_arithmetic(ctx, T);
        T = steps::canonicalize(ctx, T);
    }
    return steps::simplify_scalars(ctx, T);
}

// A tensor *field* worth expanding into frame components: slot-less, rank ≥ 1,
// carrying a field trait (a value varying in space), and not well-known.  The
// field trait is what distinguishes the physical field T we mean to expand from
// a constant abstract vector such as a reference axis i, j, k (which `express`
// handles, not `expand`).
auto is_expandable_leaf(TensorObject const& t) -> bool
{
    return t.slots.empty() && t.rank && *t.rank >= 1 && t.traits
           && t.traits->field && !t.traits->well_known;
}

// Rewrite every abstract tensor-field leaf of `v` into its components on frame
// `fb`, Σ T_ij e_i ⊗ e_j (vibe 000073).  A field derivative ∂_q T is expanded
// by Leibniz: the base is expanded to CONCRETE frame vectors, then
// differentiated through the connection ∂_q e_i (which requires `fb` to carry a
// connection — i.e. to be a chart's physical_frame).  The structural core
// shared by `expand` and the operators' operand pre-expansion; leaves basis
// contractions unreduced.
auto expand_fields(Context& ctx, Basis const& fb, Expr const* v) -> Expr const*
{
    Expr const* out = rewrite_tree(
        ctx,
        v,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&node->node);
            if (!t || !is_expandable_leaf(*t))
                return node;
            auto const derivs = t->deriv_marks;
            // The base field with its ∂ marks stripped (slots stay empty).
            // Built field-by-field so the whole-object copy does not confuse
            // -O3's maybe-uninitialized analysis.
            Expr const* base = c.make<Expr>(TensorObject{
                .name = t->name,
                .rank = t->rank,
                .traits = t->traits,
                .slots = t->slots,
                .deriv_marks = {}});
            Expr const* ex = steps::unroll_sums(
                c, expand_in_basis(c, base, fb, Variance::Covariant));
            for (auto const& dd: derivs)
            {
                Expr const* q = make_coordinate(
                    c,
                    dd.coord_name,
                    dd.wrt.chart_id,
                    dd.wrt.slot,
                    dd.wrt.nonneg);
                ex = steps::partial(c, ex, q);
            }
            // Fold the differentiated expansion before it meets an outer
            // contraction: a vanishing connection (∂_r e_i = 0) leaves T_ij·0
            // terms that, if an outer dot distributes into them, become the
            // malformed e_r·0 (vector·scalar) that canonicalize rejects.  ex
            // has no contractions yet, so folding here is safe.
            if (!derivs.empty())
                ex = steps::simplify_scalars(c, steps::canonicalize(c, ex));
            return ex;
        });
    // A tr/vec/transpose wrapper around a field is opaque to the
    // differentiator once the field inside has become an explicit
    // Σ T_ij e_i ⊗ e_j — open it on the dyads now (tr(a⊗b) → a·b, …), so
    // grad(tr ε) and friends self-prepare (vibe 000075).  Only needed when
    // something actually expanded; an already-explicit operand is unchanged.
    if (out != v)
        out = steps::expand_dyad_ops(ctx, out);
    return out;
}

// Σ_i (1/h_i) e_i ⊙ ∂_{q^i} T on the physical frame `fb`.  `combine(e_i, ∂_i
// T)` chooses ⊗ / · / ×.  When `fold_identity` (default), a resolution of
// identity Σ_k e_k⊗e_k folds back to I (so ∇R = I).
template <typename Combine>
auto del_apply(
    Context& ctx,
    CoordinateChart const& chart,
    Basis const& fb,
    Expr const* T,
    Combine combine,
    bool fold_identity) -> Expr const*
{
    int const n = static_cast<int>(chart.coords.size());
    // Expand an abstract field operand into frame components first (vibe 000073
    // Route A): the operator then differentiates the explicit Σ T_ij e_i e_j,
    // picking up the connection terms — expand-then-differentiate, the only
    // correct order in a moving frame.  A no-op on an already-explicit operand.
    T = expand_fields(ctx, fb, T);
    // Reduce the field to clean dyad form on the frame first: expand any I and
    // reduce frame crosses (so R×I etc. differentiate without an atomic I or an
    // un-fenced cross tripping canonicalize), then distribute into monomials.
    T = reduce_field(ctx, fb, T);
    T = steps::canonicalize(ctx, steps::expand_products(ctx, T));
    Expr const* acc = make_scalar(ctx, Rational{0});
    for (int i = 0; i < n; ++i)
    {
        // ∂_{q^i} T through the first-class operator (vibe 000077 step C):
        // apply_operators(∂_{q^i} ⊗ T) is the operator ∂_{q^i} acting on T,
        // which is exactly the partial derivative — so ∇ = Σ_i (1/h_i) e_i
        // ∂_{q^i} now drives grad/div/rot, each a particular product ⊙ of e_i
        // with ∂_i T.
        Expr const* di = steps::simplify_scalars(
            ctx,
            steps::apply_operators(
                ctx,
                make_tensor_product(ctx, make_deriv(ctx, chart.coords[i]), T)));
        // A coordinate the field does not depend on contributes nothing.
        if (auto const* s = std::get_if<ScalarLiteral>(&di->node);
            s && s->value.is_zero())
            continue;
        Expr const* leg = combine(fb.direction(ctx, i), di);
        Expr const* term = make_tensor_product(
            ctx, inv_h(ctx, scale_factor(ctx, chart, i)), leg);
        acc = make_sum(ctx, acc, term);
    }
    acc = steps::expand_products(ctx, acc);
    acc = steps::simplify_scalars(ctx, steps::canonicalize(ctx, acc));
    if (fold_identity)
        acc = fold_resolution_of_identity(ctx, acc, fb);
    return acc;
}

auto del(Context& ctx, CoordinateChart const& chart) -> Expr const*
{
    Basis const fb = physical_frame(ctx, chart);
    int const n = static_cast<int>(chart.coords.size());
    Expr const* acc = nullptr;
    for (int i = 0; i < n; ++i)
    {
        // The i-th operator term (1/h_i) e_i ∂_{q^i}: scalar · vector ·
        // operator, the ∂ last so it acts rightward on whatever ∇ is applied
        // to.
        Expr const* term = make_tensor_product(
            ctx,
            make_tensor_product(
                ctx,
                inv_h(ctx, scale_factor(ctx, chart, i)),
                fb.direction(ctx, i)),
            make_deriv(ctx, chart.coords[i]));
        acc = acc ? make_sum(ctx, acc, term) : term;
    }
    return acc ? acc : make_scalar(ctx, Rational{0});
}

auto expand_nabla(Context& ctx, CoordinateChart const& chart, Expr const* e)
    -> Expr const*
{
    Basis const fb = physical_frame(ctx, chart);
    int const cid = chart_id_of(chart);
    int const n = static_cast<int>(chart.coords.size());
    // The free-index expansion keeps ∇ = e_i ∂_i as a *single*
    // implicitly-summed term (not the n-fold concrete unrolling of `del`), so a
    // nested ∇×(∇×ε)ᵀ stays abstract in ε.  A moving frame's ∂_i e_j ≠ 0 and
    // its per-direction scale factors 1/h_i cannot ride a summed index
    // uniformly, so this targets a constant unit-scale (Cartesian/WCS) frame;
    // assert that here.
    for (int i = 0; i < n; ++i)
    {
        auto const* h = std::get_if<ScalarLiteral>(
            &steps::simplify_scalars(ctx, scale_factor(ctx, chart, i))->node);
        if (!h || h->value != Rational{1})
            throw std::invalid_argument(
                "expand_nabla: the free-index ∇ expansion currently supports "
                "only a constant unit-scale (Cartesian) frame; use the chart "
                "operators (grad/div/rot) for curvilinear charts");
    }
    Expr const* replaced = rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* nd) -> Expr const*
        {
            if (!std::holds_alternative<Nabla>(nd->node))
                return nd;
            // A fresh shared direction index i: the indexed frame vector e_i
            // and a free-index ∂_i carrying the same id, which sum together
            // under the existing Einstein machinery.  ∂ last, acting rightward.
            CountableIndex const i{c.alloc_index_id()};
            Expr const* ei = fb.covariant_vector(c, i);
            IndexSlot const slot =
                std::get<TensorObject>(ei->node).slots[0].slot;
            Expr const* di = make_deriv(
                c,
                make_coordinate_direction(
                    c, make_tensor_name("q"), cid, i, slot));
            return make_tensor_product(c, ei, di);
        });
    Expr const* out = steps::apply_operators(ctx, replaced);
    // ∂_i of a constant frame vector (∂_i e_j = 0) leaves zeros buried inside ⊗
    // / contraction / transpose fences — `Cross(0, …)`, and `(e_i ⊗ 0 ⊗ ∂_jε)ᵀ`
    // from a transposed grad-div, which a nested-cross / transpose interior
    // blocks canonicalize from reaching (that is Phase-1's job).  Fold the
    // algebraic zero law here so the free-index interior is clean and keeps its
    // true rank (the stray zero-product term was rank-inflating the result).
    return nf::fold_forced_zeros(ctx, out);
}

namespace
{

// The first free-index direction id occurring on any ∂-mark in `e`, if any.
auto first_free_link(Context& ctx, Expr const* e) -> std::optional<int>
{
    std::optional<int> found;
    rewrite_tree(
        ctx,
        e,
        [&](Context&, Expr const* n) -> Expr const*
        {
            if (auto const* t = std::get_if<TensorObject>(&n->node))
                for (auto const& m: t->deriv_marks)
                    if (m.free && !found)
                        found = m.link;
            return n;
        });
    return found;
}

// Fix the free direction index `link_id` to concrete direction `value`:
// concretize every frame-vector slot carrying it (e_i → e_value) and turn every
// free ∂-mark tied to it into the concrete ∂_{coord} mark.
auto concretize_dir(
    Context& ctx,
    Expr const* e,
    int link_id,
    int value,
    TensorName coord_name,
    CoordinateRef coord_ref) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* n) -> Expr const*
        {
            // A summation binder over the index being fixed loses its purpose —
            // drop it, keeping the (already-concretized) body.
            if (auto const* s = std::get_if<ExplicitSum>(&n->node))
                if (s->index.id == link_id)
                    return s->body;
            if (auto const* s = std::get_if<NoSum>(&n->node))
                if (s->index.id == link_id)
                    return s->body;
            auto const* t = std::get_if<TensorObject>(&n->node);
            if (!t)
                return n;
            auto slots = t->slots;
            bool changed = false;
            for (auto& sb: slots)
                if (sb.index)
                    if (auto const* ci =
                            std::get_if<CountableIndex>(&*sb.index))
                        if (ci->id == link_id)
                        {
                            sb.index = ConcreteIndex{value};
                            changed = true;
                        }
            auto marks = t->deriv_marks;
            for (auto& m: marks)
                if (m.free && m.link == link_id)
                {
                    m.free = false;
                    m.coord_name = coord_name;
                    m.wrt = coord_ref;
                    m.link = 0;
                    m.free_slot = {};
                    changed = true;
                }
            if (!changed)
                return n;
            std::sort(
                marks.begin(),
                marks.end(),
                [](DerivMark const& a, DerivMark const& b)
                {
                    if (a.wrt.chart_id != b.wrt.chart_id)
                        return a.wrt.chart_id < b.wrt.chart_id;
                    if (a.wrt.slot != b.wrt.slot)
                        return a.wrt.slot < b.wrt.slot;
                    return a.link < b.link;
                });
            TensorObject obj = *t;
            obj.slots = std::move(slots);
            obj.deriv_marks = std::move(marks);
            return c.make<Expr>(std::move(obj));
        });
}

} // namespace

auto componentize_nabla(
    Context& ctx, CoordinateChart const& chart, Expr const* e) -> Expr const*
{
    Basis const fb = physical_frame(ctx, chart);
    auto const id = first_free_link(ctx, e);
    if (!id)
        return e; // fully concrete already
    auto const vals = fb.space()->values();
    int const n = fb.dim();
    Expr const* acc = nullptr;
    for (int d = 0; d < n; ++d)
    {
        auto const* ct = std::get_if<TensorObject>(&chart.coords[d]->node);
        Expr const* term = concretize_dir(
            ctx,
            e,
            *id,
            vals[static_cast<std::size_t>(d)],
            ct->name,
            *ct->traits->coordinate);
        // Recurse for any remaining free direction indices (nested ∂_i∂_j …).
        term = componentize_nabla(ctx, chart, term);
        acc = acc ? make_sum(ctx, acc, term) : term;
    }
    return acc ? acc : make_scalar(ctx, Rational{0});
}

// ---- Phase-2 reassembly: free-index ∂ → ∇ operators (vibe 000078) -------

namespace
{

// A physical-frame direction vector e_ℓ carrying a countable (summation) index;
// returns ℓ.  These are the abstract directions a reduced ∇-expression pairs
// with ∂-marks — each folds back into a `Nabla`.
auto frame_dir_index(Expr const* e, Basis const& fb) -> std::optional<int>
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    if (!t || t->name.v.view() != fb.vector_symbol().v.view()
        || t->slots.size() != 1 || !t->slots[0].index
        || t->slots[0].slot.basis_id != fb.basis_id())
        return std::nullopt;
    if (auto const* ci = std::get_if<CountableIndex>(&*t->slots[0].index))
        return ci->id;
    return std::nullopt;
}

auto is_identity_tensor(Expr const* e) -> bool
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    return t && t->traits && t->traits->well_known == WellKnownKind::Identity;
}

// Strip every free ∂-mark (they are the operator indices, becoming ∇'s).
auto strip_free_marks(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context& c, Expr const* n) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&n->node);
            if (!t)
                return n;
            std::vector<DerivMark> keep;
            for (auto const& m: t->deriv_marks)
                if (!m.free)
                    keep.push_back(m);
            if (keep.size() == t->deriv_marks.size())
                return n;
            TensorObject o = *t;
            o.deriv_marks = std::move(keep);
            return c.make<Expr>(std::move(o));
        });
}

// Fold the divergence legs of one operand blob: a frame vector contracted (`·`)
// with the field is a divergence, so `e_ℓ·T` / `T·e_ℓ` → `∇·(folded T)`.  A
// `tr` of the (mark-stripped) field is its scalar invariant.  Everything else
// is the bare field (its ∂-marks stripped — the ∇'s now carry them).
auto fold_divergences(
    Context& ctx,
    Basis const& fb,
    Expr const* nabla,
    Expr const* n) -> Expr const*
{
    if (auto const* d = std::get_if<Dot>(&n->node))
    {
        if (frame_dir_index(d->left, fb))
            return make_dot(
                ctx, nabla, fold_divergences(ctx, fb, nabla, d->right));
        if (frame_dir_index(d->right, fb))
            return make_dot(
                ctx, nabla, fold_divergences(ctx, fb, nabla, d->left));
        return make_dot(
            ctx,
            fold_divergences(ctx, fb, nabla, d->left),
            fold_divergences(ctx, fb, nabla, d->right));
    }
    if (auto const* u = std::get_if<Trace>(&n->node))
        return make_trace(ctx, fold_divergences(ctx, fb, nabla, u->operand));
    return strip_free_marks(ctx, n);
}

// Does `e` carry the field being reassembled — a frame-direction vector, or a
// ∂-marked tensor — anywhere in its tree?  A ⊗-factor that carries neither is a
// plain scalar coefficient (a Lamé constant λ/μ, a numeric factor), which must
// ride through reassembly rather than be mistaken for the operand blob (vibe
// 000080, Increment 8: reassemble_nabla dropped leading scalars).
auto carries_field(Context& ctx, Expr const* e, Basis const& fb) -> bool
{
    bool found = false;
    rewrite_tree(
        ctx,
        e,
        [&](Context&, Expr const* n) -> Expr const*
        {
            if (frame_dir_index(n, fb))
                found = true;
            else if (auto const* t = std::get_if<TensorObject>(&n->node);
                     t && !t->deriv_marks.empty())
                found = true;
            return n;
        });
    return found;
}

// True if any tensor in `e` carries a ∂-mark (an applied derivative).  A term
// with frame vectors but no ∂ is a plain expanded tensor (e.g. `i_i e_i`), not
// a ∇-expression: reassembly must leave it untouched rather than mistake its
// frame vectors for gradient legs and fabricate a spurious ∇⊗1 (vibe 000081,
// I10/I11).
auto has_deriv_mark(Context& ctx, Expr const* e) -> bool
{
    bool found = false;
    rewrite_tree(
        ctx,
        e,
        [&](Context&, Expr const* n) -> Expr const*
        {
            if (auto const* t = std::get_if<TensorObject>(&n->node);
                t && !t->deriv_marks.empty())
                found = true;
            return n;
        });
    return found;
}

// The free ∂-mark link ids (the gradient/operator indices) carried anywhere in
// `e`.  An operand `∂_i X` owns one such id per applied frame derivative; used
// to tell which operand a frame-direction vector `e_ℓ` belongs to (vibe
// 000087).
auto free_mark_ids(Context& ctx, Expr const* e) -> std::vector<int>
{
    std::vector<int> ids;
    rewrite_tree(
        ctx,
        e,
        [&](Context&, Expr const* n) -> Expr const*
        {
            if (auto const* t = std::get_if<TensorObject>(&n->node))
                for (auto const& m: t->deriv_marks)
                    if (m.free)
                        ids.push_back(m.link);
            return n;
        });
    return ids;
}

// Try to reassemble a *bilinear* term — two ∂-marked operands whose gradient
// legs are contracted by a frame-dot, e.g. `(∂_j u)(e_i·e_j)(∂_i e)` = the
// cross term `(∇u)·(∇⊗e)` of the second-order Leibniz rule Δ(u e) (vibe
// 000087).  The single-operand `reassemble_term` classifier would misread the
// inter-gradient `e_i·e_j` as a Laplacian δ-pair *and* drop one field, so
// intercept this shape here.  Returns nullptr (caller falls back) unless the
// term is exactly: two operands with one free gradient leg each, one frame-dot
// joining those two legs, and only plain scalar coefficients besides.
auto try_reassemble_bilinear(
    Context& ctx,
    Basis const& fb,
    Expr const* nabla,
    std::vector<Expr const*> const& factors) -> Expr const*
{
    std::vector<Expr const*> operands;
    std::vector<Expr const*> coefficients;
    std::vector<std::pair<int, int>> dots;
    for (Expr const* f: factors)
    {
        if (auto const* d = std::get_if<Dot>(&f->node))
        {
            auto a = frame_dir_index(d->left, fb);
            auto b = frame_dir_index(d->right, fb);
            if (a && b)
            {
                dots.push_back({*a, *b});
                continue;
            }
            return nullptr; // a non-frame dot: not this shape
        }
        if (frame_dir_index(f, fb))
            return nullptr; // a free gradient leg: leave to the single path
        if (is_identity_tensor(f))
            return nullptr;
        if (carries_field(ctx, f, fb))
            operands.push_back(f);
        else
            coefficients.push_back(f);
    }
    if (operands.size() != 2 || dots.size() != 1)
        return nullptr;
    auto ids0 = free_mark_ids(ctx, operands[0]);
    auto ids1 = free_mark_ids(ctx, operands[1]);
    if (ids0.size() != 1 || ids1.size() != 1)
        return nullptr; // focused case: exactly one gradient leg per operand
    auto owner = [&](int id) -> int
    {
        if (id == ids0.front())
            return 0;
        if (id == ids1.front())
            return 1;
        return -1;
    };
    int const oa = owner(dots.front().first);
    int const ob = owner(dots.front().second);
    if (!((oa == 0 && ob == 1) || (oa == 1 && ob == 0)))
        return nullptr; // the dot must join the two *different* operands' legs

    // Contract the two gradients at their gradient legs: `(∇⊗A)ᵀ·(∇⊗B)`.  ∇⊗X
    // carries its gradient leg first; transposing the left operand brings it
    // adjacent to the right's, so the two gradient legs meet.  A rank-1
    // gradient (a scalar operand's ∇u) equals its transpose, so drop the ᵀ
    // there.
    auto grad = [&](Expr const* op)
    { return make_tensor_product(ctx, nabla, strip_free_marks(ctx, op)); };
    Expr const* gA = grad(operands[0]);
    Expr const* gB = grad(operands[1]);
    Expr const* left =
        infer_rank(gA).value_or(0) >= 2 ? make_transpose(ctx, gA) : gA;
    Expr const* res = make_dot(ctx, left, gB);
    for (Expr const* c: coefficients)
        res = make_tensor_product(ctx, c, res);
    return res;
}

// Reassemble one additive term (a ⊗-product) into ∇ operators.
auto reassemble_term(
    Context& ctx,
    Basis const& fb,
    Expr const* nabla,
    Expr const* term) -> Expr const*
{
    // No applied derivative in this term ⇒ nothing to reassemble: a bare frame
    // vector / expanded basis (`i`, `i_i e_i`) is not a ∇-expression.  Leave it
    // as-is instead of inventing a gradient ∇⊗1 (vibe 000081, I10/I11).
    if (!has_deriv_mark(ctx, term))
        return term;

    // Flatten the top-level ⊗ chain into factors.
    std::vector<Expr const*> factors;
    std::function<void(Expr const*)> flat = [&](Expr const* n)
    {
        if (auto const* p = std::get_if<TensorProduct>(&n->node))
        {
            flat(p->left);
            flat(p->right);
        }
        else
            factors.push_back(n);
    };
    flat(term);

    // A bilinear cross term (two ∂-marked operands joined by an inter-gradient
    // dot) needs the multi-operand path; the single-operand classifier below
    // would mis-fold it (vibe 000087).
    if (Expr const* bilinear = try_reassemble_bilinear(ctx, fb, nabla, factors))
        return bilinear;

    // Classify factors: δ-pairs `e_ℓ·e_m` (two directions ⇒ a Laplacian), free
    // frame vectors (gradient legs), the identity, and the operand blob (the
    // one factor carrying the field).
    int laplacians = 0;
    std::vector<Expr const*> identities;
    std::vector<std::pair<int, Expr const*>> coefficients; // (position, factor)
    std::vector<int> grad_positions;
    Expr const* operand = nullptr;
    int operand_pos = -1;
    for (int i = 0; i < static_cast<int>(factors.size()); ++i)
    {
        Expr const* f = factors[static_cast<std::size_t>(i)];
        if (auto const* d = std::get_if<Dot>(&f->node);
            d && frame_dir_index(d->left, fb) && frame_dir_index(d->right, fb))
        {
            ++laplacians;
            continue;
        }
        if (frame_dir_index(f, fb))
        {
            grad_positions.push_back(i);
            continue;
        }
        if (is_identity_tensor(f))
        {
            identities.push_back(f);
            continue;
        }
        // A factor with no field/frame content is a coefficient — a scalar
        // (λ, μ, a number) or an *undifferentiated* tensor factor (`b` in
        // (Δa)⊗b).  Keep it aside *with its position* to reattach on the
        // correct side of the operand, so its ⊗ order is preserved (vibe
        // 000087).
        if (!carries_field(ctx, f, fb))
        {
            coefficients.push_back({i, f});
            continue;
        }
        operand = f;
        operand_pos = i;
    }
    Expr const* cur = operand ? fold_divergences(ctx, fb, nabla, operand) :
                                make_scalar(ctx, Rational{1});
    // Gradient legs: a leg left of the operand is ∇⊗cur; a leg to its right is
    // (∇⊗cur)ᵀ when cur already carries a leg (a scalar operand commutes, so no
    // transpose there).  Exception — the scalar Hessian: when cur is itself the
    // gradient of a scalar (∇⊗θ, θ rank 0), ∇⊗cur = ∇∇θ is symmetric (mixed
    // partials commute), so the right-leg transpose is redundant and we drop it
    // (vibe 000080 Increment 5).
    auto is_scalar_gradient = [&](Expr const* x) -> bool
    {
        auto const* p = std::get_if<TensorProduct>(&x->node);
        return p && std::holds_alternative<Nabla>(p->left->node)
               && infer_rank(p->right) == std::optional<int>{0};
    };
    for (int pos: grad_positions)
    {
        bool const left = pos < operand_pos;
        Expr const* g = make_tensor_product(ctx, nabla, cur);
        bool const transpose = !left && infer_rank(cur).value_or(0) >= 1
                               && !is_scalar_gradient(cur);
        cur = transpose ? make_transpose(ctx, g) : g;
    }
    // Laplacians: Δ = ∇·(∇⊗·).
    for (int k = 0; k < laplacians; ++k)
        cur = make_dot(ctx, nabla, make_tensor_product(ctx, nabla, cur));
    for (Expr const* id: identities)
        cur = make_tensor_product(ctx, cur, id);
    // Coefficients keep their ⊗ order relative to the operand: a factor left of
    // the operand attaches on the left (λ ∇(∇·u), μ Δu, …), one to its right on
    // the right ((Δa)⊗b) — so an undifferentiated tensor factor keeps its leg
    // order.  A scalar commutes, so its side is immaterial (canonicalize pools
    // it).  With no operand blob (operand_pos < 0) every factor rides left, the
    // historical behaviour.
    for (auto const& [pos, coef]: coefficients)
        cur = (operand_pos >= 0 && pos > operand_pos) ?
                  make_tensor_product(ctx, cur, coef) :
                  make_tensor_product(ctx, coef, cur);
    return cur;
}

// Stamp the chart's dimension onto every identity in the reassembled result.
// The result is chart-bound, so its abstract I takes the chart's space (letting
// `tr(c·I) → c·n` fold, vibe 000081 B1) — overwriting any mismatched default a
// user's I may have carried in (vibe 000082).  Dimension-awareness is
// orthogonal to the slots (`TensorObject::dim`), so the I stays slotless.
auto dimension_identities(Context& ctx, Expr const* e, IndexSpace const* space)
    -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* n) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&n->node);
            if (t && t->traits
                && t->traits->well_known == WellKnownKind::Identity
                && t->dim != space)
                return make_identity(c, space);
            return n;
        });
}

} // namespace

auto reassemble_nabla(Context& ctx, CoordinateChart const& chart, Expr const* e)
    -> Expr const*
{
    Basis const fb = physical_frame(ctx, chart);
    Expr const* const nabla = make_nabla(ctx);
    // Summation binders are structural noise here (the frame-vector ↔ ∂-mark
    // pairing is what carries the Einstein sum); drop them, then fold each
    // additive term.
    Expr const* body = rewrite_tree(
        ctx,
        e,
        [](Context&, Expr const* n) -> Expr const*
        {
            if (auto const* s = std::get_if<ExplicitSum>(&n->node))
                return s->body;
            if (auto const* s = std::get_if<NoSum>(&n->node))
                return s->body;
            return n;
        });
    std::function<Expr const*(Expr const*, int)> go =
        [&](Expr const* n, int sign) -> Expr const*
    {
        if (auto const* s = std::get_if<Sum>(&n->node))
            return make_sum(ctx, go(s->left, sign), go(s->right, sign));
        if (auto const* s = std::get_if<Difference>(&n->node))
            return make_sum(ctx, go(s->left, sign), go(s->right, -sign));
        if (auto const* s = std::get_if<Negate>(&n->node))
            return go(s->operand, -sign);
        Expr const* r = reassemble_term(ctx, fb, nabla, n);
        return sign < 0 ? make_negate(ctx, r) : r;
    };
    return dimension_identities(ctx, go(body, +1), fb.space());
}

auto gradient(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* f,
    bool fold_identity) -> Expr const*
{
    Basis const fb = physical_frame(ctx, chart);
    // grad T = Σ_i (1/h_i) e_i ⊗ ∂_i T (rank + 1).
    return del_apply(
        ctx,
        chart,
        fb,
        f,
        [&](Expr const* ei, Expr const* di)
        { return make_tensor_product(ctx, ei, di); },
        fold_identity);
}

auto divergence(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* v,
    bool fold_identity) -> Expr const*
{
    Basis const fb = physical_frame(ctx, chart);
    // div v = Σ_i (1/h_i) e_i · ∂_i v (rank − 1).
    return del_apply(
        ctx,
        chart,
        fb,
        v,
        [&](Expr const* ei, Expr const* di)
        { return reduce_dot(ctx, fb, ei, di); },
        fold_identity);
}

auto laplacian(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* f,
    bool fold_identity) -> Expr const*
{
    return divergence(
        ctx, chart, gradient(ctx, chart, f, fold_identity), fold_identity);
}

auto rot(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* v,
    bool fold_identity) -> Expr const*
{
    if (chart.reference.dim() != 3)
        throw std::invalid_argument(
            "rot: only the 3D cross product is defined");
    Basis const fb = physical_frame(ctx, chart);
    // rot v = ∇×v = Σ_i (1/h_i) e_i × ∂_i v.
    return del_apply(
        ctx,
        chart,
        fb,
        v,
        [&](Expr const* ei, Expr const* di)
        { return reduce_cross(ctx, fb, ei, di); },
        fold_identity);
}

auto frame_dot(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* u,
    Expr const* v) -> Expr const*
{
    return reduce_dot(ctx, physical_frame(ctx, chart), u, v);
}

auto frame_cross(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* u,
    Expr const* v) -> Expr const*
{
    if (chart.reference.dim() != 3)
        throw std::invalid_argument(
            "frame_cross: only the 3D cross product is defined");
    return reduce_cross(ctx, physical_frame(ctx, chart), u, v);
}

auto to_reference(Context& ctx, Expr const* v) -> Expr const*
{
    // Replace every registered physical-frame vector e_i in `v` with its
    // reference-frame (WCS) expansion (vibe 000071 P4).  Works via the
    // connection registry, so a vector mixing several charts' frames is fully
    // expanded; anything not a registered frame atom is left untouched.
    Expr const* out = rewrite_tree(
        ctx,
        v,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&node->node);
            if (!t || t->slots.size() != 1 || !t->slots[0].index)
                return node;
            int const bid = t->slots[0].slot.basis_id;
            if (bid == 0)
                return node;
            auto const* conn = c.connection(bid);
            if (!conn || conn->reference_expansion.empty())
                return node;
            auto const* ci = std::get_if<ConcreteIndex>(&*t->slots[0].index);
            if (!ci)
                return node;
            for (std::size_t k = 0; k < conn->values.size(); ++k)
                if (conn->values[k] == ci->value)
                    return conn->reference_expansion[k];
            return node;
        });
    out = steps::expand_products(ctx, out);
    return steps::simplify_scalars(ctx, steps::canonicalize(ctx, out));
}

auto express(Context& ctx, CoordinateChart const& target, Expr const* v)
    -> Expr const*
{
    // Re-express `v` in `target`'s physical frame (vibe 000071 P4; generalised
    // to every leg in vibe 000072 Obs 8): first bring it to the shared
    // reference (WCS) frame, then rewrite each reference direction i_a into its
    // target-frame expansion
    //     i_a = Σ_k (i_a · e_k) e_k,
    // which projects *all* legs of a tensor of any rank (not just the one an
    // outer dot would contract) onto target's symbolic e_k atoms.
    //
    // An abstract ∇ over an already-expanded basis cannot be re-expressed:
    // canon has no normal form for ∇ in a ⊗-fence (vibe 000081, Part 2). Refuse
    // with a clear pointer to the ∇-first order rather than silently
    // corrupting.
    if (steps::abstract_nabla_over_expanded_basis(ctx, v))
        throw std::invalid_argument(
            "express: ∇ is still abstract over an expanded basis — canon cannot "
            "reduce a ∇ nested in a basis ⊗-fence (it would silently drop the "
            "gradient or crash). Expand ∇ first (expand_nabla / grad / div / "
            "rot), then expand the basis.");
    Basis const tf = physical_frame(ctx, target);
    Basis const& ref = target.reference;
    Expr const* w = to_reference(ctx, v);

    // Each reference axis a's expansion in the target frame, precomputed once.
    int const n = tf.dim();
    std::vector<Expr const*> expansion(static_cast<std::size_t>(ref.dim()));
    for (int a = 0; a < ref.dim(); ++a)
    {
        Expr const* ea = nullptr;
        for (int k = 0; k < n; ++k)
        {
            // component (i_a · e_k), reduced in the reference frame.
            Expr const* ck = reduce_dot(ctx, ref, ref.basis(a), tf.basis(k));
            if (auto const* s = std::get_if<ScalarLiteral>(&ck->node);
                s && s->value.is_zero())
                continue;
            Expr const* term =
                make_tensor_product(ctx, ck, tf.direction(ctx, k));
            ea = ea ? make_sum(ctx, ea, term) : term;
        }
        expansion[static_cast<std::size_t>(a)] =
            ea ? ea : make_scalar(ctx, Rational{0});
    }

    // Replace every reference basis vector atom in w by its target expansion.
    Expr const* out = rewrite_tree(
        ctx,
        w,
        [&](Context&, Expr const* node) -> Expr const*
        {
            for (int a = 0; a < ref.dim(); ++a)
                if (structural_eq(node, ref.basis(a)))
                    return expansion[static_cast<std::size_t>(a)];
            return node;
        });
    out = steps::expand_products(ctx, out);
    // Collect like frame-dyad terms before simplifying: re-expressing a moving
    // frame leg spreads one coefficient over several trig pieces (e.g. `1/r` of
    // ∇e_r fans into four `e_θe_θ` terms), which only fold once gathered onto a
    // common coefficient and reduced (vibe 000081, I9).
    out = steps::simplify_scalars(
        ctx, steps::collect_terms(ctx, steps::canonicalize(ctx, out)));
    // vibe 000072 Obs 8: collapse a completed resolution of identity
    // Σ_k e_k⊗e_k in the target frame back to I (guarded — a no-op otherwise).
    return fold_resolution_of_identity(ctx, out, tf);
}

auto expand(Context& ctx, CoordinateChart const& chart, Expr const* v)
    -> Expr const*
{
    Basis const fb = physical_frame(ctx, chart); // ensures connection
                                                 // registered
    Expr const* out = expand_fields(ctx, fb, v);
    // Reduce to clean frame form: distribute products and the contractions that
    // now straddle ⊗ fences (the connection terms), reduce basis dots/crosses
    // to δ/ε, evaluate them, and fold.  Iterated to a fixpoint.
    Expr const* prev = nullptr;
    for (int guard = 0; out != prev && guard <= 8; ++guard)
    {
        prev = out;
        out = steps::expand_products(ctx, out);
        out = steps::distribute_contraction(ctx, out);
        out = simplify_basis_dot(ctx, out, fb);
        out = simplify_basis_cross(ctx, out, fb);
        out = steps::canonicalize(ctx, out);
        out = steps::unroll_sums(ctx, out);
        out = steps::eval_delta_concrete(ctx, out);
        out = steps::eval_eps_concrete(ctx, out);
        out = steps::fold_arithmetic(ctx, out);
        out = steps::canonicalize(ctx, out);
    }
    out = steps::simplify_scalars(ctx, out);
    return fold_resolution_of_identity(ctx, out, fb);
}

auto components(Context& ctx, CoordinateChart const& chart, Expr const* v)
    -> std::vector<Expr const*>
{
    if (auto const rk = infer_rank(v); rk && *rk >= 2)
        throw std::invalid_argument(
            "components: input has rank " + std::to_string(*rk)
            + " — components projects a vector onto the frame; use "
              "component_matrix for a rank-2 tensor");
    Basis const fb = physical_frame(ctx, chart);
    // Expand-first (vibe 000073), so an abstract vector field f surfaces as its
    // frame components [f_r, f_θ, f_z].  A no-op on an already-explicit vector
    // (its e_i carry slots, so nothing is an expandable field leaf).
    Expr const* w = expand_fields(ctx, fb, v);
    // An atomic identity tensor would leave e_i·I unreduced — expand it into
    // Σ e_k ⊗ e_k so the projection dots evaluate (vibe 000075).
    w = expand_identity_frame(ctx, fb, w);
    std::vector<Expr const*> out;
    out.reserve(static_cast<std::size_t>(fb.dim()));
    for (int i = 0; i < fb.dim(); ++i)
        out.push_back(reduce_dot(ctx, fb, w, fb.direction(ctx, i)));
    return out;
}

auto component_matrix(Context& ctx, CoordinateChart const& chart, Expr const* v)
    -> std::vector<std::vector<Expr const*>>
{
    Basis const fb = physical_frame(ctx, chart);
    // Expand-first, like components: an abstract rank-2 field T surfaces as
    // Σ T_kl e_k ⊗ e_l before projection, so the matrix entries come out as
    // the minted physical components (with symmetry folded, T_θr → T_rθ).
    Expr const* w = expand_fields(ctx, fb, v);
    // A term like (Δθ)·I needs the identity opened into Σ e_k ⊗ e_k, or the
    // projection leaves (e_i·I)·e_j unreduced (vibe 000075).
    w = expand_identity_frame(ctx, fb, w);
    std::vector<std::vector<Expr const*>> out;
    out.reserve(static_cast<std::size_t>(fb.dim()));
    for (int i = 0; i < fb.dim(); ++i)
    {
        // e_i·T once per row, then (e_i·T)·e_j per column.
        Expr const* row_vec = reduce_dot(ctx, fb, fb.direction(ctx, i), w);
        std::vector<Expr const*> row;
        row.reserve(static_cast<std::size_t>(fb.dim()));
        for (int j = 0; j < fb.dim(); ++j)
            row.push_back(reduce_dot(ctx, fb, row_vec, fb.direction(ctx, j)));
        out.push_back(std::move(row));
    }
    return out;
}

auto position(Context& ctx, CoordinateChart const& chart) -> Expr const*
{
    // The position vector in the chart's own physical frame (vibe 000072 Obs
    // 6): radius_vector is assembled in the reference (WCS) frame from the
    // embedding and is the geometry primitive the metric / scale factors derive
    // from; position projects it onto the frame's e_i (e.g. cylindrical r e_r +
    // z e_z), staying intrinsic so grad(position) folds to I without a mixed
    // frame.
    return express(ctx, chart, radius_vector(ctx, chart));
}

} // namespace tender
