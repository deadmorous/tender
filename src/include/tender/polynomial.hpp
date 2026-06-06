#pragma once

#include <string>
#include <utility>
#include <vector>

#include <tender/rational.hpp>

namespace tender
{

// Univariate polynomial with Rational coefficients and integer exponents.
// Stored in canonical form: terms sorted by descending exponent, zero
// coefficients omitted. The "variable" is anonymous at this level; callers
// supply the variable name when converting to LaTeX or Python strings.
class Polynomial
{
public:
    // (coefficient, non-negative exponent)
    using Term = std::pair<Rational, int>;

    // Zero polynomial.
    Polynomial() noexcept = default;

    // Construct from an unordered list of terms (duplicates are accumulated;
    // zero-coefficient terms and those with negative exponents are rejected).
    explicit Polynomial(std::vector<Term> terms);

    // --- Accessors ---

    [[nodiscard]] auto terms() const noexcept -> std::vector<Term> const&
    {
        return terms_;
    }

    // Degree of the polynomial, or -1 for the zero polynomial.
    [[nodiscard]] auto degree() const noexcept -> int;

    // Coefficient of x^e (zero if the term is absent).
    [[nodiscard]] auto coeff(int e) const -> Rational;

    // True iff all coefficients are zero.
    [[nodiscard]] auto is_zero() const noexcept -> bool
    {
        return terms_.empty();
    }

    // --- Arithmetic ---

    [[nodiscard]] auto operator+(Polynomial const& rhs) const -> Polynomial;
    [[nodiscard]] auto operator-(Polynomial const& rhs) const -> Polynomial;
    [[nodiscard]] auto operator*(Polynomial const& rhs) const -> Polynomial;
    [[nodiscard]] auto operator-() const -> Polynomial;

    auto operator+=(Polynomial const& rhs) -> Polynomial&
    {
        return *this = *this + rhs;
    }
    auto operator-=(Polynomial const& rhs) -> Polynomial&
    {
        return *this = *this - rhs;
    }
    auto operator*=(Polynomial const& rhs) -> Polynomial&
    {
        return *this = *this * rhs;
    }

    [[nodiscard]] auto operator==(Polynomial const& rhs) const noexcept -> bool;

    // --- Calculus ---

    // Formal derivative (d/dx).
    [[nodiscard]] auto diff() const -> Polynomial;

    // --- Evaluation ---

    // Evaluate at x, using exact Rational arithmetic.
    [[nodiscard]] auto eval(Rational x) const -> Rational;

    // --- Output ---

    // LaTeX string for the polynomial, e.g. "3 x^{2} - 2 x + 1".
    // `var` is the variable name (single char or \text{...} is caller's job).
    [[nodiscard]] auto to_latex(std::string const& var = "x") const
        -> std::string;

    // Python/SymPy-style string, e.g. "3*x**2 - 2*x + 1".
    [[nodiscard]] auto to_python(std::string const& var = "x") const
        -> std::string;

private:
    // terms_ is always sorted descending by exponent with no zero coefficients.
    std::vector<Term> terms_;

    // Build the canonical form from a raw term list.
    static auto canonicalize(std::vector<Term> raw) -> std::vector<Term>;
};

} // namespace tender
