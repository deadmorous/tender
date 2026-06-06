#include <tender/expr.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace tender
{

// ===========================================================================
// Helpers — shared with Polynomial rendering but private to this TU
// ===========================================================================

// Absolute-value LaTeX for a coefficient; omits "1" when a ring-element
// factor follows (has_factor == true).
static auto coeff_latex(Rational const& c, bool has_factor) -> std::string
{
    Rational const abs_c(c.num() < 0 ? -c.num() : c.num(), c.den());
    if (has_factor && abs_c == Rational{1})
        return "";
    if (abs_c.den() == 1)
        return std::to_string(abs_c.num());
    return "\\frac{" + std::to_string(abs_c.num()) + "}{"
           + std::to_string(abs_c.den()) + "}";
}

static auto coeff_python(Rational const& c, bool has_factor) -> std::string
{
    Rational const abs_c(c.num() < 0 ? -c.num() : c.num(), c.den());
    if (has_factor && abs_c == Rational{1})
        return "";
    return abs_c.to_string();
}

// Wrap var latex/python in parens when it is a Sum (precedence guard).
static auto parenthesise_if_sum(Expr const* var, std::string s) -> std::string
{
    if (dynamic_cast<Sum const*>(var))
        return "(" + s + ")";
    return s;
}

// ===========================================================================
// ring_power — compute var^n in the ring determined by var->rank()
// ===========================================================================

static auto ring_power(ResourceList& rl, Expr* var, int n) -> Expr*
{
    if (var->rank() == 0)
        return make_pow(rl, var, Rational{n});

    // rank == 2: repeated single contraction; n >= 0 guaranteed
    if (n == 0)
        return make_identity(rl);
    Expr* result = var;
    for (int i = 1; i < n; ++i)
        result = make_contract(rl, result, var);
    return result;
}

// ===========================================================================
// PolynomialExpr — constructor
// ===========================================================================

PolynomialExpr::PolynomialExpr(Polynomial poly, Expr* var) :
  poly_(std::move(poly)), var_(var), rank_(var->rank())
{
    if (var->rank() != 0 && var->rank() != 2)
        throw std::invalid_argument(
            "PolynomialExpr: variable must be rank 0 or 2, got rank "
            + std::to_string(var->rank()));
}

// ===========================================================================
// PolynomialExpr — latex
// ===========================================================================

auto PolynomialExpr::latex() const -> std::string
{
    if (poly_.is_zero())
        return "0";

    std::string var_tex =
        parenthesise_if_sum(var_, var_->latex());

    // rank 0: identical structure to Polynomial::to_latex —
    // constant term is the bare coefficient, no unit symbol.
    if (rank_ == 0)
        return poly_.to_latex(var_tex);

    // rank 2: constant term renders as coeff·I  (A^0 = I in the ring).
    std::string const unit = "\\mathbf{I}";
    std::string result;
    for (auto const& [c, e]: poly_.terms())
    {
        bool const negative = c.num() < 0;

        std::string var_part;
        if (e == 0)
            var_part = " " + unit;
        else if (e == 1)
            var_part = " " + var_tex;
        else
            var_part = " " + var_tex + "^{" + std::to_string(e) + "}";

        std::string const abs_coeff = coeff_latex(c, true);

        std::string term_str;
        if (!abs_coeff.empty())
            term_str = abs_coeff + var_part;
        else
            term_str = var_part.substr(1); // strip leading space

        if (result.empty())
            result = (negative ? "-" : "") + term_str;
        else
            result += (negative ? " - " : " + ") + term_str;
    }
    return result;
}

// ===========================================================================
// PolynomialExpr — python
// ===========================================================================

auto PolynomialExpr::python() const -> std::string
{
    if (rank_ == 0)
    {
        std::string var_py = parenthesise_if_sum(var_, var_->python());
        return poly_.to_python(var_py);
    }

    // rank 2: no natural inline Python expression; serialise explicitly.
    std::string result = "polynomial_expr(" + var_->python() + ", [";
    bool first = true;
    for (auto const& [c, e]: poly_.terms())
    {
        if (!first)
            result += ", ";
        result += "(" + c.to_string() + ", " + std::to_string(e) + ")";
        first = false;
    }
    return result + "])";
}

// ===========================================================================
// PolynomialExpr — expand
// ===========================================================================

auto PolynomialExpr::expand(ResourceList& rl) const -> Expr*
{
    if (poly_.is_zero())
        return make_rational(rl, Rational{0});

    std::vector<Expr*> terms;
    terms.reserve(poly_.terms().size());
    for (auto const& [c, e]: poly_.terms())
        terms.push_back(make_scale(rl, c, ring_power(rl, var_, e)));
    return make_sum(rl, std::move(terms));
}

// ===========================================================================
// Factory
// ===========================================================================

auto make_polynomial_expr(
    ResourceList& rl, Polynomial poly, Expr* var) -> PolynomialExpr*
{
    return rl.make<PolynomialExpr>(std::move(poly), var);
}

} // namespace tender
