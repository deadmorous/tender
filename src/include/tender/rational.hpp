#pragma once

#include <compare>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace tender
{

/// Exact rational number backed by int64_t numerator and denominator.
/// Always kept in lowest terms with a positive denominator.
/// Arithmetic overflow triggers std::abort() — by design (see vibes/000011).
class Rational
{
public:
    Rational() noexcept : num_(0), den_(1)
    {
    }

    // Implicit conversion from integer is intentional: lets users write
    // Rational r = 3 and mix integers in arithmetic naturally.
    // NOLINTNEXTLINE(google-explicit-constructor)
    Rational(int64_t n) noexcept : num_(n), den_(1)
    {
    }

    Rational(int64_t num, int64_t den) : num_(num), den_(den)
    {
        if (den_ == 0)
            die("Rational: zero denominator");
        normalize();
    }

    // --- Accessors ---

    [[nodiscard]] auto num() const noexcept -> int64_t
    {
        return num_;
    }
    [[nodiscard]] auto den() const noexcept -> int64_t
    {
        return den_;
    }
    [[nodiscard]] auto is_zero() const noexcept -> bool
    {
        return num_ == 0;
    }
    [[nodiscard]] auto is_integer() const noexcept -> bool
    {
        return den_ == 1;
    }

    // --- Conversions ---

    [[nodiscard]] auto to_double() const noexcept -> double
    {
        return static_cast<double>(num_) / static_cast<double>(den_);
    }

    [[nodiscard]] auto to_string() const -> std::string
    {
        if (den_ == 1)
            return std::to_string(num_);
        return std::to_string(num_) + '/' + std::to_string(den_);
    }

    // --- Arithmetic ---

    [[nodiscard]] auto operator-() const -> Rational
    {
        if (num_ == INT64_MIN)
            die("Rational: overflow (negation of INT64_MIN)");
        return Rational{Tag{}, -num_, den_};
    }

    [[nodiscard]] auto operator+(Rational const& rhs) const -> Rational
    {
        // (num_*rhs.den_ + rhs.num_*den_) / (den_*rhs.den_)
        auto const n = i128(num_) * rhs.den_ + i128(rhs.num_) * den_;
        auto const d = i128(den_) * rhs.den_;
        return Rational{to64(n, "addition"), to64(d, "addition")};
    }

    [[nodiscard]] auto operator-(Rational const& rhs) const -> Rational
    {
        return *this + (-rhs);
    }

    [[nodiscard]] auto operator*(Rational const& rhs) const -> Rational
    {
        return Rational{
            to64(i128(num_) * rhs.num_, "multiplication"),
            to64(i128(den_) * rhs.den_, "multiplication")};
    }

    [[nodiscard]] auto operator/(Rational const& rhs) const -> Rational
    {
        if (rhs.is_zero())
            die("Rational: division by zero");
        // num_/den_ ÷ rhs.num_/rhs.den_ = (num_*rhs.den_) / (den_*rhs.num_)
        return Rational{
            to64(i128(num_) * rhs.den_, "division"),
            to64(i128(den_) * rhs.num_, "division")};
    }

    auto operator+=(Rational const& rhs) -> Rational&
    {
        return *this = *this + rhs;
    }
    auto operator-=(Rational const& rhs) -> Rational&
    {
        return *this = *this - rhs;
    }
    auto operator*=(Rational const& rhs) -> Rational&
    {
        return *this = *this * rhs;
    }
    auto operator/=(Rational const& rhs) -> Rational&
    {
        return *this = *this / rhs;
    }

    // --- Comparison ---

    // Normalised form makes equality a direct field compare.
    [[nodiscard]] auto operator==(Rational const& rhs) const noexcept -> bool
    {
        return num_ == rhs.num_ && den_ == rhs.den_;
    }

    // Cross-multiply for ordering; denominators are positive so sign is safe.
    // Uses __int128 to avoid overflow in the product.
    [[nodiscard]] auto operator<=>(Rational const& rhs) const noexcept
        -> std::strong_ordering
    {
        auto const l = i128(num_) * rhs.den_;
        auto const r = i128(rhs.num_) * den_;
        if (l < r)
            return std::strong_ordering::less;
        if (l > r)
            return std::strong_ordering::greater;
        return std::strong_ordering::equal;
    }

private:
    int64_t num_;
    int64_t den_; // always > 0 after normalisation

    using i128 = __int128;

    // Tag for the already-normalised fast-path constructor.
    struct Tag
    {
    };
    Rational(Tag, int64_t num, int64_t den) noexcept : num_(num), den_(den)
    {
    }

    static auto to64(i128 v, char const* op) -> int64_t
    {
        if (v > INT64_MAX || v < INT64_MIN)
        {
            char buf[64];
            std::snprintf(
                buf, sizeof(buf), "Rational: arithmetic overflow (%s)", op);
            die(buf);
        }
        return static_cast<int64_t>(v);
    }

    static auto gcd(int64_t a, int64_t b) noexcept -> int64_t
    {
        // Euclidean algorithm; a and b must be non-negative.
        while (b != 0)
        {
            int64_t const t = b;
            b = a % b;
            a = t;
        }
        return a;
    }

    auto normalize() -> void
    {
        if (num_ == 0)
        {
            den_ = 1;
            return;
        }
        if (den_ < 0)
        {
            if (num_ == INT64_MIN || den_ == INT64_MIN)
                die("Rational: overflow in normalisation");
            num_ = -num_;
            den_ = -den_;
        }
        int64_t const g = gcd(num_ < 0 ? -num_ : num_, den_);
        num_ /= g;
        den_ /= g;
    }

    [[noreturn]] static auto die(char const* msg) noexcept -> void
    {
        std::fputs(msg, stderr);
        std::fputc('\n', stderr);
        std::abort();
    }
};

} // namespace tender
