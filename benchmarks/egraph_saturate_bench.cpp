// Saturation benchmark (vibe 000034 item #10).
//
// Reports the deterministic shape of saturation — passes to the fixed point,
// e-node and e-class counts — plus a rough wall-clock per run.  The counts are
// the primary signal; timing depends on the build type and is informational.
//
// Dependency-free by design (CLAUDE.md principle 7): a tiny steady_clock loop,
// no benchmark framework.

#include <tender/context.hpp>
#include <tender/egraph.hpp>
#include <tender/expr.hpp>
#include <tender/identity.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

using namespace tender;

namespace
{

auto delta_ul(Context& ctx, CountableIndex a, CountableIndex b) -> Expr const*
{
    return make_delta(
        ctx, Realm::Oblique, space_3d(), Level::Upper, Level::Lower, a, b);
}

auto delta_ll(Context& ctx, CountableIndex a, CountableIndex b) -> Expr const*
{
    return make_delta(
        ctx, Realm::Oblique, space_3d(), Level::Lower, Level::Lower, a, b);
}

// Σ_p δ^p_a δ^p_b
auto contraction(
    Context& ctx,
    CountableIndex p,
    CountableIndex a,
    CountableIndex b) -> Expr const*
{
    return make_explicit_sum(
        ctx,
        p,
        make_tensor_product(ctx, delta_ul(ctx, p, a), delta_ul(ctx, p, b)));
}

// Σ_i Σ_j ε^{ijk} ε_{ijl}
auto eps_pair(
    Context& ctx,
    CountableIndex i,
    CountableIndex j,
    CountableIndex k,
    CountableIndex l) -> Expr const*
{
    auto eps =
        [&](Level lvl, CountableIndex x, CountableIndex y, CountableIndex z)
    {
        return make_levi_civita(
            ctx,
            Realm::Oblique,
            space_3d(),
            {lvl, lvl, lvl},
            {IndexAssoc{x}, IndexAssoc{y}, IndexAssoc{z}});
    };
    return make_explicit_sum(
        ctx,
        i,
        make_explicit_sum(
            ctx,
            j,
            make_tensor_product(
                ctx, eps(Level::Upper, i, j, k), eps(Level::Lower, i, j, l))));
}

template <typename F>
auto ns_per_op(int reps, F&& f) -> double
{
    using clock = std::chrono::steady_clock;
    auto const t0 = clock::now();
    for (int i = 0; i < reps; ++i)
        f();
    auto const t1 = clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
}

struct Case final
{
    char const* name;
    Expr const* target;
    Identity rule;
};

void run(Context& ctx, Case const& c, int reps)
{
    // Deterministic shape: one cold run.
    EGraph eg{ctx};
    auto const root = eg.add(c.target);
    int const passes = eg.saturate({c.rule});
    auto const nodes = eg.node_count();
    auto const classes = eg.class_count();
    (void)eg.extract(eg.find(root));

    auto const ns = ns_per_op(
        reps,
        [&]
        {
            EGraph e{ctx};
            auto r = e.add(c.target);
            (void)e.saturate({c.rule});
            (void)e.extract(e.find(r));
        });

    std::printf(
        "%-22s passes=%d  nodes=%zu  classes=%zu  %8.1f ns/op\n",
        c.name,
        passes,
        nodes,
        classes,
        ns);
}

} // namespace

auto main() -> int
{
    Context ctx;

    CountableIndex p{ctx.alloc_index_id()}, a{ctx.alloc_index_id()},
        b{ctx.alloc_index_id()};
    CountableIndex q{ctx.alloc_index_id()}, m{ctx.alloc_index_id()},
        n{ctx.alloc_index_id()};
    Identity contraction_rule{
        "delta-contraction", contraction(ctx, p, a, b), delta_ll(ctx, a, b)};

    CountableIndex i{ctx.alloc_index_id()}, j{ctx.alloc_index_id()},
        k{ctx.alloc_index_id()}, l{ctx.alloc_index_id()};
    CountableIndex e{ctx.alloc_index_id()}, f{ctx.alloc_index_id()},
        g{ctx.alloc_index_id()}, h{ctx.alloc_index_id()};
    Identity eps_rule{
        "eps-delta-2",
        eps_pair(ctx, i, j, k, l),
        make_tensor_product(
            ctx, make_scalar(ctx, Rational{2}), delta_ul(ctx, k, l))};

    std::vector<Case> cases = {
        {"delta-contraction", contraction(ctx, q, m, n), contraction_rule},
        {"eps-delta-2", eps_pair(ctx, e, f, g, h), eps_rule},
    };

    std::printf("e-graph saturation benchmark\n");
    for (auto const& c: cases)
        run(ctx, c, 2000);
    return 0;
}
