#include <tender/rational.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>

using tender::Rational;
using Clock = std::chrono::high_resolution_clock;

static auto now_ns() -> int64_t
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               Clock::now().time_since_epoch())
        .count();
}

static void print_result(char const* label, int64_t elapsed_ns, int iterations)
{
    double const ns_per_op = static_cast<double>(elapsed_ns) / iterations;
    std::printf("  %-26s %6.1f ns/op  (%d ops)\n", label, ns_per_op, iterations);
}

int main()
{
    constexpr int N = 10'000'000;

    std::printf("Rational benchmark\n");
    std::printf("==================\n");

    // Use small prime-like moduli so inputs vary but stay well within int64.
    // Numerators in [1..97], denominators in [2..104] — all products fit
    // easily.

    // --- Addition ---
    {
        int64_t checksum = 0;
        int64_t const t0 = now_ns();
        for (int i = 0; i < N; ++i)
        {
            Rational const a(i % 97 + 1, i % 101 + 3);
            Rational const b(i % 89 + 1, i % 103 + 2);
            Rational const r = a + b;
            checksum += r.num();
        }
        int64_t const t1 = now_ns();
        int64_t volatile sink = checksum;
        (void)sink;
        print_result("Addition:", t1 - t0, N);
    }

    // --- Multiplication ---
    {
        int64_t checksum = 0;
        int64_t const t0 = now_ns();
        for (int i = 0; i < N; ++i)
        {
            Rational const a(i % 97 + 1, i % 101 + 3);
            Rational const b(i % 89 + 1, i % 103 + 2);
            Rational const r = a * b;
            checksum += r.num();
        }
        int64_t const t1 = now_ns();
        int64_t volatile sink = checksum;
        (void)sink;
        print_result("Multiplication:", t1 - t0, N);
    }

    // --- Construction + GCD normalisation ---
    {
        int64_t checksum = 0;
        int64_t const t0 = now_ns();
        for (int i = 1; i <= N; ++i)
        {
            // 2i/4i always normalises to 1/2; exercises the GCD path every
            // time.
            Rational const r(2 * i, 4 * i);
            checksum += r.num();
        }
        int64_t const t1 = now_ns();
        int64_t volatile sink = checksum;
        (void)sink;
        print_result("Construction + GCD:", t1 - t0, N);
    }

    return 0;
}
