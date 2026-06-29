#include <tender/chart.hpp>

#include <tender/derivation.hpp>

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

// Bilinearly expand a dot of (sums/differences/negations of) vector terms:
// (A + B)·C → A·C + B·C, so the basis-vector dots inside reach
// simplify_basis_dot.  Leaves a leaf-vs-leaf pair as a single Dot.
auto distribute_dot(Context& ctx, Expr const* l, Expr const* r) -> Expr const*
{
    if (auto const* s = std::get_if<Sum>(&l->node))
        return make_sum(
            ctx,
            distribute_dot(ctx, s->left, r),
            distribute_dot(ctx, s->right, r));
    if (auto const* d = std::get_if<Difference>(&l->node))
        return make_difference(
            ctx,
            distribute_dot(ctx, d->left, r),
            distribute_dot(ctx, d->right, r));
    if (auto const* n = std::get_if<Negate>(&l->node))
        return make_negate(ctx, distribute_dot(ctx, n->operand, r));
    if (auto const* s = std::get_if<Sum>(&r->node))
        return make_sum(
            ctx,
            distribute_dot(ctx, l, s->left),
            distribute_dot(ctx, l, s->right));
    if (auto const* d = std::get_if<Difference>(&r->node))
        return make_difference(
            ctx,
            distribute_dot(ctx, l, d->left),
            distribute_dot(ctx, l, d->right));
    if (auto const* n = std::get_if<Negate>(&r->node))
        return make_negate(ctx, distribute_dot(ctx, l, n->operand));
    return make_dot(ctx, l, r);
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

// The invariant dot u·v of two vectors expressed in the orthonormal reference
// frame `ref`, reduced to a scalar field: distribute the dot bilinearly over
// the sums so the basis-vector dots surface, turn the concrete frame-vector
// legs into Kronecker deltas, evaluate them, and simplify.
auto reduce_scalar_dot(
    Context& ctx, Basis const& ref, Expr const* u, Expr const* v) -> Expr const*
{
    Expr const* e = distribute_dot(ctx, u, v);
    e = simplify_basis_dot(ctx, e, ref);
    e = steps::canonicalize(ctx, e);
    e = steps::eval_delta_concrete(ctx, e);
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
        out.push_back(reduce_scalar_dot(ctx, chart.reference, de, pb.basis(k)));
    return out;
}

// (1/h) ⊗ x as a scalar quotient, simplified.
auto over_h(Context& ctx, Expr const* x, Expr const* h) -> Expr const*
{
    return steps::simplify_scalars(ctx, make_scalar_div(ctx, x, h));
}

// The sign of the permutation (a, b, c) of {0,1,2}: +1 even, −1 odd, 0 on any
// repeat — the 3D Levi-Civita symbol.
auto eps3(int a, int b, int c) -> int
{
    if (a == b || b == c || a == c)
        return 0;
    return ((b - a) * (c - a) * (c - b) > 0) ? 1 : -1;
}

} // namespace

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
    return reduce_scalar_dot(ctx, chart.reference, gi, gj);
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

// ---- differential operators (vibe 000069 M6) ---------------------------
//
// In the orthonormal physical frame, ∇ = Σ_i (1/h_i) e_i ∂_{q^i}.  Applying it
// with the Leibniz rule and the connection coefficients γ^k_{ij} = (∂_j
// e_i)·e_k (M5) gives the textbook curvilinear operators; nothing is
// hand-tabulated. Vector fields are carried by their physical components v =
// Σ_i v_i e_i.

auto gradient(Context& ctx, CoordinateChart const& chart, Expr const* f)
    -> std::vector<Expr const*>
{
    int const n = static_cast<int>(chart.coords.size());
    std::vector<Expr const*> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i)
        out.push_back(over_h(
            ctx,
            steps::partial(ctx, f, chart.coords[i]),
            scale_factor(ctx, chart, i)));
    return out;
}

auto divergence(
    Context& ctx,
    CoordinateChart const& chart,
    std::vector<Expr const*> const& v) -> Expr const*
{
    int const n = static_cast<int>(chart.coords.size());
    if (static_cast<int>(v.size()) != n)
        throw std::invalid_argument("divergence: component count != dimension");
    Basis const pb = physical_basis(ctx, chart);
    std::vector<Expr const*> h(n);
    for (int i = 0; i < n; ++i)
        h[i] = scale_factor(ctx, chart, i);

    // ∇·v = Σ_i (1/h_i) e_i·∂_{q^i} v, v = Σ_j v_j e_j.  With the Leibniz rule
    // and e_i·e_k = δ_ik this is Σ_i (1/h_i) ∂_i v_i + Σ_{i,j} (1/h_i) v_j
    // γ^i_{ji} (γ^i_{ji} = (∂_{q^i} e_j)·e_i).
    Expr const* acc = make_scalar(ctx, Rational{0});
    for (int i = 0; i < n; ++i)
        acc = make_sum(
            ctx,
            acc,
            over_h(ctx, steps::partial(ctx, v[i], chart.coords[i]), h[i]));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            Expr const* g = connection_at(ctx, chart, pb, j, i)[i];
            acc = make_sum(
                ctx, acc, over_h(ctx, make_tensor_product(ctx, v[j], g), h[i]));
        }
    return steps::simplify_scalars(ctx, acc);
}

auto laplacian(Context& ctx, CoordinateChart const& chart, Expr const* f)
    -> Expr const*
{
    return divergence(ctx, chart, gradient(ctx, chart, f));
}

auto rot(
    Context& ctx,
    CoordinateChart const& chart,
    std::vector<Expr const*> const& v) -> std::vector<Expr const*>
{
    int const n = static_cast<int>(chart.coords.size());
    if (n != 3)
        throw std::invalid_argument(
            "rot: only the 3D cross product is defined");
    if (static_cast<int>(v.size()) != 3)
        throw std::invalid_argument("rot: component count != dimension");
    Basis const pb = physical_basis(ctx, chart);
    std::vector<Expr const*> h(3);
    for (int i = 0; i < 3; ++i)
        h[i] = scale_factor(ctx, chart, i);
    // gamma[a][d] = coefficients of ∂_{q^d} e_a (over k).
    std::vector<std::vector<std::vector<Expr const*>>> gamma(
        3, std::vector<std::vector<Expr const*>>(3));
    for (int a = 0; a < 3; ++a)
        for (int d = 0; d < 3; ++d)
            gamma[a][d] = connection_at(ctx, chart, pb, a, d);

    auto add = [&](Expr const* acc, int s, Expr const* term) -> Expr const*
    { return make_sum(ctx, acc, s > 0 ? term : make_negate(ctx, term)); };

    // ∇×v = Σ_i (1/h_i) e_i × ∂_{q^i} v; with e_i×e_k = ε_{ikm} e_m the m-th
    // physical component is Σ_{i,j} (1/h_i) ε_{ijm} ∂_i v_j +
    // Σ_{i,j,k} (1/h_i) ε_{ikm} v_j γ^k_{ji}.
    std::vector<Expr const*> out(3, nullptr);
    for (int m = 0; m < 3; ++m)
    {
        Expr const* acc = make_scalar(ctx, Rational{0});
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (int s = eps3(i, j, m))
                    acc =
                        add(acc,
                            s,
                            over_h(
                                ctx,
                                steps::partial(ctx, v[j], chart.coords[i]),
                                h[i]));
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    if (int s = eps3(i, k, m))
                        acc = add(
                            acc,
                            s,
                            over_h(
                                ctx,
                                make_tensor_product(ctx, v[j], gamma[j][i][k]),
                                h[i]));
        out[m] = steps::simplify_scalars(ctx, acc);
    }
    return out;
}

} // namespace tender
