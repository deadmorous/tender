#include <tender/polynomial.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace tender
{

// ===========================================================================
// Helpers
// ===========================================================================

auto Polynomial::canonicalize(std::vector<Term> raw) -> std::vector<Term>
{
    // Reject negative exponents.
    for (auto const& [c, e]: raw)
        if (e < 0)
            throw std::invalid_argument(
                "Polynomial: negative exponents are not allowed");

    // Accumulate coefficients for the same exponent.
    std::vector<Term> result;
    for (auto const& [c, e]: raw)
    {
        bool found = false;
        for (auto& [rc, re]: result)
        {
            if (re == e)
            {
                rc += c;
                found = true;
                break;
            }
        }
        if (!found)
            result.push_back({c, e});
    }

    // Remove zero-coefficient terms.
    result.erase(
        std::remove_if(
            result.begin(),
            result.end(),
            [](Term const& t) { return t.first.is_zero(); }),
        result.end());

    // Sort descending by exponent.
    std::sort(
        result.begin(),
        result.end(),
        [](Term const& a, Term const& b) { return a.second > b.second; });

    return result;
}

// ===========================================================================
// Constructor
// ===========================================================================

Polynomial::Polynomial(std::vector<Term> terms) :
  terms_(canonicalize(std::move(terms)))
{
}

// ===========================================================================
// Accessors
// ===========================================================================

auto Polynomial::degree() const noexcept -> int
{
    if (terms_.empty())
        return -1;
    return terms_.front().second;
}

auto Polynomial::coeff(int e) const -> Rational
{
    for (auto const& [c, ex]: terms_)
        if (ex == e)
            return c;
    return Rational{0};
}

// ===========================================================================
// Arithmetic
// ===========================================================================

auto Polynomial::operator+(Polynomial const& rhs) const -> Polynomial
{
    auto combined = terms_;
    combined.insert(combined.end(), rhs.terms_.begin(), rhs.terms_.end());
    return Polynomial{std::move(combined)};
}

auto Polynomial::operator-(Polynomial const& rhs) const -> Polynomial
{
    return *this + (-rhs);
}

auto Polynomial::operator*(Polynomial const& rhs) const -> Polynomial
{
    std::vector<Term> result;
    for (auto const& [lc, le]: terms_)
        for (auto const& [rc, re]: rhs.terms_)
            result.push_back({lc * rc, le + re});
    return Polynomial{std::move(result)};
}

auto Polynomial::operator-() const -> Polynomial
{
    std::vector<Term> result;
    result.reserve(terms_.size());
    for (auto const& [c, e]: terms_)
        result.push_back({-c, e});
    // Already canonical except for sign flip — canonicalize to be safe.
    return Polynomial{std::move(result)};
}

auto Polynomial::operator==(Polynomial const& rhs) const noexcept -> bool
{
    return terms_ == rhs.terms_;
}

// ===========================================================================
// Calculus
// ===========================================================================

auto Polynomial::diff() const -> Polynomial
{
    std::vector<Term> result;
    for (auto const& [c, e]: terms_)
    {
        if (e == 0)
            continue; // constant term vanishes
        result.push_back({c * Rational{e}, e - 1});
    }
    return Polynomial{std::move(result)};
}

// ===========================================================================
// Evaluation
// ===========================================================================

auto Polynomial::eval(Rational x) const -> Rational
{
    Rational result{0};
    for (auto const& [c, e]: terms_)
    {
        // Compute x^e by repeated multiplication.
        Rational power{1};
        for (int i = 0; i < e; ++i)
            power *= x;
        result += c * power;
    }
    return result;
}

// ===========================================================================
// Output helpers
// ===========================================================================

// Render a coefficient for LaTeX (absolute value with special-case ±1
// when an explicit variable follows).
static auto coeff_latex(Rational const& c, bool has_var) -> std::string
{
    Rational abs_c = c.num() < 0 ? -c : c;
    if (has_var && abs_c == Rational{1})
        return "";
    if (abs_c.den() == 1)
        return std::to_string(abs_c.num());
    return "\\frac{" + std::to_string(abs_c.num()) + "}{"
           + std::to_string(abs_c.den()) + "}";
}

static auto coeff_python(Rational const& c, bool has_var) -> std::string
{
    Rational abs_c = c.num() < 0 ? -c : c;
    if (has_var && abs_c == Rational{1})
        return "";
    return abs_c.to_string();
}

auto Polynomial::to_latex(std::string const& var) const -> std::string
{
    if (terms_.empty())
        return "0";

    std::string result;
    for (auto const& [c, e]: terms_)
    {
        bool negative = c.num() < 0;
        std::string var_part;
        if (e == 0)
            var_part = "";
        else if (e == 1)
            var_part = " " + var;
        else
            var_part = " " + var + "^{" + std::to_string(e) + "}";

        bool has_var = (e > 0);
        std::string abs_coeff = coeff_latex(c, has_var);

        std::string term_str;
        if (!abs_coeff.empty())
            term_str = abs_coeff + var_part;
        else
            term_str = var_part.substr(1); // strip leading space

        if (result.empty())
        {
            result = (negative ? "-" : "") + term_str;
        }
        else
        {
            result += (negative ? " - " : " + ") + term_str;
        }
    }
    return result;
}

auto Polynomial::to_python(std::string const& var) const -> std::string
{
    if (terms_.empty())
        return "0";

    std::string result;
    for (auto const& [c, e]: terms_)
    {
        bool negative = c.num() < 0;
        std::string var_part;
        if (e == 0)
            var_part = "";
        else if (e == 1)
            var_part = "*" + var;
        else
            var_part = "*" + var + "**" + std::to_string(e);

        bool has_var = (e > 0);
        std::string abs_coeff = coeff_python(c, has_var);

        std::string term_str;
        if (!abs_coeff.empty())
            term_str = abs_coeff + var_part;
        else
            term_str = var_part.substr(1); // strip leading '*'

        if (result.empty())
        {
            result = (negative ? "-" : "") + term_str;
        }
        else
        {
            result += (negative ? " - " : " + ") + term_str;
        }
    }
    return result;
}

} // namespace tender
