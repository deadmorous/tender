#include <tender/chart.hpp>

#include <tender/context.hpp>
#include <tender/derivation.hpp>
#include <tender/rewrite.hpp>

#include <cmath>
#include <stdexcept>
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

auto physical_basis(Context& ctx, CoordinateChart const& chart) -> Basis
{
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
            return chart.reference;
    }
    return make_orthonormal_basis(
        ctx,
        chart.reference.space(),
        std::move(vectors),
        make_tensor_name("e"),
        Handedness::Right,
        BasisNaming{.value_names = std::move(value_names)});
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

auto physical_frame(Context& ctx, CoordinateChart const& chart) -> Basis
{
    int const n = static_cast<int>(chart.coords.size());

    // The chart these coordinates belong to (all coords share one chart_id).
    auto const* c0 = std::get_if<TensorObject>(&chart.coords.front()->node);
    if (!c0 || !c0->traits || !c0->traits->coordinate)
        throw std::invalid_argument(
            "physical_frame: coordinate is not a coordinate variable");
    int const chart_id = c0->traits->coordinate->chart_id;

    // Idempotent per chart: reuse the frame if already built, so the user's
    // fields and the operators share the same e_i atoms and connection.  The
    // cache is validated by the chart's geometry fingerprint (vibe 000072 Obs
    // 2): a chart_id reused for a *different* chart fails the match and falls
    // through to rebuild, rather than silently returning the stale frame.
    auto fingerprint = chart_fingerprint(chart);
    if (auto const* cf = ctx.chart_frame(chart_id);
        cf && cf->fingerprint == fingerprint)
        if (auto const* b = ctx.basis(cf->basis_id))
            return *b;

    Basis const basis = physical_basis(ctx, chart);

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
    ctx.register_chart_frame(chart_id, basis.basis_id(), std::move(fingerprint));
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
        T = steps::canonicalize(ctx, T);
        T = steps::unroll_sums(ctx, T);
        T = steps::eval_eps_concrete(ctx, T);
        T = steps::fold_arithmetic(ctx, T);
        T = steps::canonicalize(ctx, T);
    }
    return steps::simplify_scalars(ctx, T);
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
    // Reduce the field to clean dyad form on the frame first: expand any I and
    // reduce frame crosses (so R×I etc. differentiate without an atomic I or an
    // un-fenced cross tripping canonicalize), then distribute into monomials.
    T = reduce_field(ctx, fb, T);
    T = steps::canonicalize(ctx, steps::expand_products(ctx, T));
    Expr const* acc = make_scalar(ctx, Rational{0});
    for (int i = 0; i < n; ++i)
    {
        Expr const* di = steps::simplify_scalars(
            ctx, steps::partial(ctx, T, chart.coords[i]));
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
    out = steps::simplify_scalars(ctx, steps::canonicalize(ctx, out));
    // vibe 000072 Obs 8: collapse a completed resolution of identity
    // Σ_k e_k⊗e_k in the target frame back to I (guarded — a no-op otherwise).
    return fold_resolution_of_identity(ctx, out, tf);
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
