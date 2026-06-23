// Nf canon benchmark (vibe 000058 / C10).
//
// Times the new `Expr → Nf` lowering (`canonicalize_nf`) on a representative
// expression: a small additive layer with like-term merging, a cancelling
// pair, a wedged scalar that floats out, a contraction chain, and a genuine
// sum that sinks into a Paren.  The result's term count is the deterministic
// signal; timing is informational and build-type dependent.
//
// Each iteration canonicalizes into a *fresh* Context so the arena stays
// bounded — the input `Expr` is built once and read read-only by every call.
//
// Dependency-free by design (CLAUDE.md principle 7): a steady_clock loop, no
// benchmark framework.

#include <tender/context.hpp>
#include <tender/expr.hpp>
#include <tender/name.hpp>
#include <tender/nf.hpp>
#include <tender/nf_lower.hpp>
#include <tender/rational.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>

using namespace tender;
using tender::nf::canonicalize_nf;
using Clock = std::chrono::high_resolution_clock;

namespace
{

auto now_ns() -> int64_t
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               Clock::now().time_since_epoch())
        .count();
}

auto vec(Context& ctx, char const* name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 1);
}

auto mat(Context& ctx, char const* name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 2);
}

auto scaled(Context& ctx, int n, Expr const* e) -> Expr const*
{
    return make_tensor_product(ctx, make_scalar(ctx, Rational{n}), e);
}

// (a·b)⊗C + 2·((a·b)⊗C) + A:B − A:B + D·(b+c)
//   → 3·((a·b)⊗C) + D·(b+c)   (a wedged scalar, a cancellation, a Paren)
auto sample(Context& ctx) -> Expr const*
{
    auto const* a = vec(ctx, "a");
    auto const* b = vec(ctx, "b");
    auto const* c = vec(ctx, "c");
    auto const* A = mat(ctx, "A");
    auto const* B = mat(ctx, "B");
    auto const* C = mat(ctx, "C");
    auto const* D = mat(ctx, "D");

    auto const* wedged = make_tensor_product(ctx, make_dot(ctx, a, b), C);
    auto const* ab = make_ddot(ctx, A, B);
    auto const* paren = make_dot(ctx, D, make_sum(ctx, b, c));

    auto const* e = make_sum(ctx, wedged, scaled(ctx, 2, wedged));
    e = make_sum(ctx, e, ab);
    e = make_difference(ctx, e, ab);
    return make_sum(ctx, e, paren);
}

} // namespace

int main()
{
    constexpr int N = 200'000;

    std::printf("Nf canon benchmark\n");
    std::printf("==================\n");

    Context build;
    auto const* e = sample(build);

    // Deterministic shape: term count of the canonical form.
    Context probe;
    auto const* probe_nf = canonicalize_nf(probe, e);
    std::printf("  canonical terms: %zu\n", probe_nf->terms.size());

    int64_t checksum = 0;
    int64_t const t0 = now_ns();
    for (int i = 0; i < N; ++i)
    {
        Context iter;
        auto const* out = canonicalize_nf(iter, e);
        checksum += static_cast<int64_t>(out->terms.size());
    }
    int64_t const t1 = now_ns();
    int64_t volatile sink = checksum;
    (void)sink;

    double const us_per_op = static_cast<double>(t1 - t0) / N / 1000.0;
    std::printf("  canonicalize_nf: %6.2f us/op  (%d ops)\n", us_per_op, N);
    return 0;
}
