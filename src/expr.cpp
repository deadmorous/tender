#include <tender/expr.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace tender
{

// ===========================================================================
// LaTeX helpers
// ===========================================================================

static auto sym_to_latex(std::string const& sym) -> std::string
{
    if (sym.size() == 1)
        return sym;
    // If the symbol already contains LaTeX commands, pass it through as-is
    // so that e.g. \boldsymbol{\sigma} is not wrapped in \text{}.
    if (sym.find('\\') != std::string::npos)
        return sym;
    // Component notation: "a^1" → "a^{1}", "b_1" → "b_{1}".
    // Splitting at ^ / _ keeps them in math mode, avoiding pdflatex's
    // rejection of ^ inside \text{} in display math.
    auto hat = sym.find('^');
    if (hat != std::string::npos)
        return sym_to_latex(sym.substr(0, hat)) + "^{" + sym.substr(hat + 1)
               + "}";
    auto us = sym.find('_');
    if (us != std::string::npos)
        return sym_to_latex(sym.substr(0, us)) + "_{" + sym.substr(us + 1)
               + "}";
    return "\\text{" + sym + "}";
}

static auto rational_to_latex(Rational const& r) -> std::string
{
    if (r.den() == 1)
        return std::to_string(r.num());
    auto const abs_num = r.num() < 0 ? -r.num() : r.num();
    std::string frac = "\\frac{" + std::to_string(abs_num) + "}{"
                       + std::to_string(r.den()) + "}";
    return r.num() < 0 ? "-" + frac : frac;
}

// Absolute-value latex for a rational: same as rational_to_latex but with
// the sign stripped (used by Sum when emitting an explicit " - ").
// make_sum always places the RationalConst at position 0, so a negative
// RationalConst can only appear as the first term and never reaches
// latex_unsigned(); this function is defensive generality. GCOV_EXCL_START
static auto rational_to_latex_abs(Rational const& r) -> std::string
{
    Rational const abs_r(r.num() < 0 ? -r.num() : r.num(), r.den());
    return rational_to_latex(abs_r);
} // GCOV_EXCL_STOP

// ===========================================================================
// Expr::set_name
// ===========================================================================

auto Expr::set_name(std::string n) -> void
{
    if (!name_.empty() && name_ != n)
        throw std::logic_error(
            "Expr: cannot rename '" + name_ + "' to '" + n + "'");
    name_ = std::move(n);
}

// ===========================================================================
// RationalConst
// ===========================================================================

auto RationalConst::latex() const -> std::string
{
    if (has_name())
        return sym_to_latex(name());
    return rational_to_latex(value_);
}

auto RationalConst::python() const -> std::string
{
    if (has_name())
        return name();
    return "Rational(" + value_.to_string() + ")";
}

// ===========================================================================
// NamedConst
// ===========================================================================

auto NamedConst::latex() const -> std::string
{
    return sym_to_latex(has_name() ? name() : symbol_);
}

auto NamedConst::python() const -> std::string
{
    return "named_const('" + symbol_ + "')";
}

// ===========================================================================
// SymbolicVar
// ===========================================================================

auto SymbolicVar::latex() const -> std::string
{
    return sym_to_latex(has_name() ? name() : symbol_);
}

auto SymbolicVar::python() const -> std::string
{
    return "symbolic_var('" + symbol_ + "')";
}

// ===========================================================================
// Scale
// ===========================================================================

auto Scale::latex() const -> std::string
{
    if (coeff_ == Rational{-1})
        return "-" + expr_->latex();
    return rational_to_latex(coeff_) + " " + expr_->latex();
}

auto Scale::python() const -> std::string
{
    return "scale(" + coeff_.to_string() + ", " + expr_->python() + ")";
}

// ===========================================================================
// Sum — latex helpers
// ===========================================================================

// True if e has a negative leading coefficient, so Sum should emit " - |e|".
static auto has_negative_lead(Expr const* e) -> bool
{
    if (auto const* rc = dynamic_cast<RationalConst const*>(e))
        return rc->value().num() < 0; // GCOV_EXCL_LINE
    if (auto const* sc = dynamic_cast<Scale const*>(e))
        return sc->coeff().num() < 0;
    return false;
}

// LaTeX for e with its leading sign stripped (for use after an explicit " - ").
static auto latex_unsigned(Expr const* e) -> std::string
{
    if (auto const* rc = dynamic_cast<RationalConst const*>(e))
        return rational_to_latex_abs(rc->value()); // GCOV_EXCL_LINE
    if (auto const* sc = dynamic_cast<Scale const*>(e))
    {
        Rational const abs_coeff(
            sc->coeff().num() < 0 ? -sc->coeff().num() : sc->coeff().num(),
            sc->coeff().den());
        if (abs_coeff == Rational{1})
            return sc->expr()->latex();
        return rational_to_latex(abs_coeff) + " " + sc->expr()->latex();
    }
    return e->latex(); // GCOV_EXCL_LINE
}

auto Sum::latex() const -> std::string
{
    std::string result;
    for (auto const* term: terms_)
    {
        if (result.empty())
        {
            result = term->latex();
        }
        else if (has_negative_lead(term))
        {
            result += " - " + latex_unsigned(term);
        }
        else
        {
            result += " + " + term->latex();
        }
    }
    return result.empty() ? "0" : result;
}

auto Sum::python() const -> std::string
{
    std::string result = "sum([";
    for (std::size_t i = 0; i < terms_.size(); ++i)
    {
        if (i > 0)
            result += ", ";
        result += terms_[i]->python();
    }
    return result + "])";
}

// ===========================================================================
// TensorProduct
// ===========================================================================

auto TensorProduct::latex() const -> std::string
{
    if (lhs_->rank() == 0 && rhs_->rank() >= 1)
    {
        // Scalar × tensor: render without \otimes for readability.
        bool rhs_needs_parens = dynamic_cast<Sum const*>(rhs_) != nullptr;
        std::string rhs_tex =
            rhs_needs_parens ? "(" + rhs_->latex() + ")" : rhs_->latex();
        return "(" + lhs_->latex() + ") " + rhs_tex;
    }
    if (lhs_->rank() >= 1 && rhs_->rank() == 0)
    {
        // Tensor × scalar: render without \otimes for readability.
        // Atomic scalars (single-token symbols, functions, powers) need no
        // parens.
        bool rhs_atomic = dynamic_cast<RationalConst const*>(rhs_)
                          || dynamic_cast<NamedConst const*>(rhs_)
                          || dynamic_cast<SymbolicVar const*>(rhs_)
                          || dynamic_cast<NamedTensor const*>(rhs_)
                          || dynamic_cast<FunctionApply const*>(rhs_)
                          || dynamic_cast<Pow const*>(rhs_);
        std::string rhs_tex =
            rhs_atomic ? rhs_->latex() : "(" + rhs_->latex() + ")";
        return lhs_->latex() + " " + rhs_tex;
    }
    return lhs_->latex() + " \\otimes " + rhs_->latex();
}

auto TensorProduct::python() const -> std::string
{
    return "tp(" + lhs_->python() + ", " + rhs_->python() + ")";
}

// ===========================================================================
// Factory: make_rational, make_named_const, make_symbolic_var
// ===========================================================================

auto make_rational(ResourceList& rl, Rational r) -> Expr*
{
    return rl.make<RationalConst>(std::move(r));
}

auto make_named_const(ResourceList& rl, std::string sym) -> Expr*
{
    return rl.make<NamedConst>(std::move(sym));
}

auto make_symbolic_var(ResourceList& rl, std::string sym) -> Expr*
{
    return rl.make<SymbolicVar>(std::move(sym));
}

// ===========================================================================
// Factory: make_scale (forward-declared for use in make_sum)
// ===========================================================================

auto make_scale(ResourceList& rl, Rational r, Expr* e) -> Expr*
{
    // Scale(0, e) → 0
    if (r.is_zero())
        return make_rational(rl, Rational{0});

    // Scale(r, RationalConst(c)) → RationalConst(r * c)
    if (auto* rc = dynamic_cast<RationalConst*>(e))
        return make_rational(rl, r * rc->value());

    // Scale(r, Scale(s, inner)) → Scale(r*s, inner)
    if (auto* sc = dynamic_cast<Scale*>(e))
        return make_scale(rl, r * sc->coeff(), sc->expr());

    // Scale(r, Sum(...)) → Sum(Scale(r, t1), Scale(r, t2), …)
    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> scaled;
        scaled.reserve(s->terms().size());
        for (auto* t: s->terms())
            scaled.push_back(make_scale(rl, r, t));
        return make_sum(rl, std::move(scaled));
    }

    // Scale(1, e) → e  (after other cases so Scale(1, Scale(…)) is still
    // collapsed)
    if (r == Rational{1})
        return e;

    return rl.make<Scale>(std::move(r), e);
}

// ===========================================================================
// Factory: make_sum
// ===========================================================================

namespace
{

// Decomposes a flat sequence of Expr* into (base, coeff) contributions,
// flattening nested Sums and hoisting Scale coefficients.
struct Flattener
{
    ResourceList& rl;
    Rational constant{0};
    // base is never nullptr, never a Sum, never a Scale
    std::vector<std::pair<Expr*, Rational>> terms;

    void accumulate(Rational const& coeff, Expr* base)
    {
        if (coeff.is_zero())
            return; // GCOV_EXCL_LINE
        for (auto& [b, c]: terms)
        {
            if (b == base)
            {
                c += coeff;
                return;
            }
        }
        terms.push_back({base, coeff});
    }

    void add(Expr* e)
    {
        if (auto* s = dynamic_cast<Sum*>(e))
        {
            for (auto* t: s->terms())
                add(t);
        }
        else if (auto* sc = dynamic_cast<Scale*>(e))
        {
            // Scale invariant: inner is not Sum or Scale or RationalConst
            accumulate(sc->coeff(), sc->expr());
        }
        else if (auto* rc = dynamic_cast<RationalConst*>(e))
        {
            constant += rc->value();
        }
        else
        {
            accumulate(Rational{1}, e);
        }
    }
};

} // namespace

auto make_sum(ResourceList& rl, std::vector<Expr*> terms) -> Expr*
{
    Flattener f{rl};
    for (auto* t: terms)
        f.add(t);

    std::vector<Expr*> result;

    if (!f.constant.is_zero())
        result.push_back(rl.make<RationalConst>(f.constant));

    for (auto const& [base, coeff]: f.terms)
    {
        if (!coeff.is_zero())
            result.push_back(make_scale(rl, coeff, base));
    }

    if (result.empty())
        return make_rational(rl, Rational{0});
    if (result.size() == 1)
        return result[0];

    int const rank = result[0]->rank();
    for (auto const* t: result)
    {
        if (t->rank() != rank)
            throw std::invalid_argument(
                "Sum: all terms must have the same rank");
    }

    return rl.make<Sum>(std::move(result), rank);
}

// ===========================================================================
// Factory: make_tensor_product
// ===========================================================================

static auto is_zero_expr(Expr const* e) -> bool
{
    if (auto const* rc = dynamic_cast<RationalConst const*>(e))
        return rc->value().is_zero();
    return false;
}

auto make_tensor_product(ResourceList& rl, Expr* lhs, Expr* rhs) -> Expr*
{
    if (is_zero_expr(lhs) || is_zero_expr(rhs))
        return make_rational(rl, Rational{0});
    return rl.make<TensorProduct>(lhs, rhs);
}

// ===========================================================================
// named()
// ===========================================================================

auto named(std::string n, Expr* e) -> Expr*
{
    e->set_name(std::move(n));
    return e;
}

// ===========================================================================
// NamedTensor
// ===========================================================================

NamedTensor::NamedTensor(std::string symbol, int rank, SlotList slots) :
  symbol_(std::move(symbol)), rank_(rank)
{
    set_slots(std::move(slots));
}

static auto slot_decorations(SlotList const& sl) -> std::string
{
    std::string upper_part;
    std::string lower_part;
    for (auto const& s: sl)
    {
        if (s.level == SlotLevel::Upper)
            upper_part += s.display;
        else
            lower_part += s.display;
    }
    std::string result;
    if (!upper_part.empty())
        result += "^{" + upper_part + "}";
    if (!lower_part.empty())
        result += "_{" + lower_part + "}";
    return result;
}

auto NamedTensor::latex() const -> std::string
{
    std::string base = sym_to_latex(has_name() ? name() : symbol_);
    return base + slot_decorations(slots());
}

auto NamedTensor::python() const -> std::string
{
    return "tensor('" + symbol_ + "', " + std::to_string(rank_) + ")";
}

// ===========================================================================
// ExplicitSum
// ===========================================================================

ExplicitSum::ExplicitSum(Expr* body, Index* index) : body_(body), index_(index)
{
    // Find the first upper and first lower slot bound to index.
    std::size_t upper_pos = SIZE_MAX;
    std::size_t lower_pos = SIZE_MAX;
    auto const& bslots = body->slots();
    for (std::size_t i = 0; i < bslots.size(); ++i)
    {
        if (bslots[i].index != index_)
            continue;
        if (bslots[i].level == SlotLevel::Upper && upper_pos == SIZE_MAX)
            upper_pos = i;
        else if (bslots[i].level == SlotLevel::Lower && lower_pos == SIZE_MAX)
            lower_pos = i;
    }
    if (upper_pos == SIZE_MAX || lower_pos == SIZE_MAX)
        throw std::invalid_argument(
            "ExplicitSum: body must have one upper and one lower slot "
            "bound to the given index");
    rank_ = body->rank() - 2;
    SlotList remaining;
    for (std::size_t i = 0; i < bslots.size(); ++i)
        if (i != upper_pos && i != lower_pos)
            remaining.push_back(bslots[i]);
    set_slots(std::move(remaining));
}

auto ExplicitSum::latex() const -> std::string
{
    return "\\sum_{" + index_->letter() + "} " + body_->latex();
}

auto ExplicitSum::python() const -> std::string
{
    return "explicit_sum(index=" + index_->letter() + ", " + body_->python()
           + ")";
}

// ===========================================================================
// NoSum
// ===========================================================================

NoSum::NoSum(Expr* body, Index* index) : body_(body), index_(index)
{
    rank_ = body->rank();
    set_slots(body->slots());
}

auto NoSum::latex() const -> std::string
{
    return body_->latex();
}

auto NoSum::python() const -> std::string
{
    return "no_sum(index=" + index_->letter() + ", " + body_->python() + ")";
}

// ===========================================================================
// Contraction
// ===========================================================================

Contraction::Contraction(
    Expr* lhs,
    std::size_t slot_lhs,
    Expr* rhs,
    std::size_t slot_rhs,
    int rank,
    SlotList slots) :
  lhs_(lhs), rhs_(rhs), slot_lhs_(slot_lhs), slot_rhs_(slot_rhs), rank_(rank)
{
    set_slots(std::move(slots));
}

auto Contraction::latex() const -> std::string
{
    return lhs_->latex() + " " + rhs_->latex();
}

auto Contraction::python() const -> std::string
{
    return "convolve(" + lhs_->python() + ", " + std::to_string(slot_lhs_)
           + ", " + rhs_->python() + ", " + std::to_string(slot_rhs_) + ")";
}

// ===========================================================================
// Factory functions — indexed nodes (Phase 3)
// ===========================================================================

auto make_named_tensor(
    ResourceList& rl, std::string sym, int rank, SlotList slots) -> Expr*
{
    return rl.make<NamedTensor>(std::move(sym), rank, std::move(slots));
}

auto make_explicit_sum(ResourceList& rl, Expr* body, Index* index) -> Expr*
{
    return rl.make<ExplicitSum>(body, index);
}

auto make_no_sum(ResourceList& rl, Expr* body, Index* index) -> Expr*
{
    return rl.make<NoSum>(body, index);
}

auto convolve(
    ResourceList& rl, Expr* a, std::size_t slot_a, Expr* b, std::size_t slot_b)
    -> Expr*
{
    auto const& sa = a->slots();
    auto const& sb = b->slots();
    if (slot_a >= sa.size())
        throw std::invalid_argument(
            "convolve: slot_a out of range for first operand");
    if (slot_b >= sb.size())
        throw std::invalid_argument(
            "convolve: slot_b out of range for second operand");
    if (sa[slot_a].level == sb[slot_b].level)
        throw std::invalid_argument(
            "convolve: both slots have the same level (need one upper, "
            "one lower)");

    int const result_rank = a->rank() + b->rank() - 2;

    SlotList result_slots;
    for (std::size_t i = 0; i < sa.size(); ++i)
        if (i != slot_a)
            result_slots.push_back(sa[i]);
    for (std::size_t i = 0; i < sb.size(); ++i)
        if (i != slot_b)
            result_slots.push_back(sb[i]);

    return rl.make<Contraction>(
        a, slot_a, b, slot_b, result_rank, std::move(result_slots));
}

// ===========================================================================
// Phase 5 — standard scalar function nodes
// ===========================================================================

static auto function_latex_name(FunctionKind k) -> char const*
{
    switch (k)
    {
        case FunctionKind::Exp: return "\\exp";
        case FunctionKind::Log: return "\\ln";
        case FunctionKind::Sin: return "\\sin";
        case FunctionKind::Cos: return "\\cos";
        case FunctionKind::Tan: return "\\tan";
        case FunctionKind::ASin: return "\\arcsin";
        case FunctionKind::ACos: return "\\arccos";
        case FunctionKind::ATan: return "\\arctan";
        case FunctionKind::Sinh: return "\\sinh";
        case FunctionKind::Cosh: return "\\cosh";
        case FunctionKind::Tanh: return "\\tanh";
        case FunctionKind::Sqrt: return "\\sqrt";
    }
    return ""; // GCOV_EXCL_LINE
}

static auto function_python_name(FunctionKind k) -> char const*
{
    switch (k)
    {
        case FunctionKind::Exp: return "exp";
        case FunctionKind::Log: return "log";
        case FunctionKind::Sin: return "sin";
        case FunctionKind::Cos: return "cos";
        case FunctionKind::Tan: return "tan";
        case FunctionKind::ASin: return "asin";
        case FunctionKind::ACos: return "acos";
        case FunctionKind::ATan: return "atan";
        case FunctionKind::Sinh: return "sinh";
        case FunctionKind::Cosh: return "cosh";
        case FunctionKind::Tanh: return "tanh";
        case FunctionKind::Sqrt: return "sqrt";
    }
    return ""; // GCOV_EXCL_LINE
}

FunctionApply::FunctionApply(FunctionKind kind, Expr* arg) :
  kind_(kind), arg_(arg)
{
    if (arg->rank() != 0)
        throw std::invalid_argument(
            "FunctionApply: argument must be a scalar (rank 0), got rank "
            + std::to_string(arg->rank()));
}

auto FunctionApply::latex() const -> std::string
{
    std::string name = function_latex_name(kind_);
    if (kind_ == FunctionKind::Sqrt)
        return name + "{" + arg_->latex() + "}";
    return name + "(" + arg_->latex() + ")";
}

auto FunctionApply::python() const -> std::string
{
    return std::string{function_python_name(kind_)} + "(" + arg_->python()
           + ")";
}

Pow::Pow(Expr* base, Rational exp) : base_(base), exp_(std::move(exp))
{
    if (base->rank() != 0)
        throw std::invalid_argument(
            "Pow: base must be a scalar (rank 0), got rank "
            + std::to_string(base->rank()));
}

auto Pow::latex() const -> std::string
{
    return base_->latex() + "^{" + rational_to_latex(exp_) + "}";
}

auto Pow::python() const -> std::string
{
    return "pow(" + base_->python() + ", " + exp_.to_string() + ")";
}

ATan2::ATan2(Expr* y, Expr* x) : y_(y), x_(x)
{
    if (y->rank() != 0 || x->rank() != 0)
        throw std::invalid_argument(
            "ATan2: both arguments must be scalar (rank 0)");
}

auto ATan2::latex() const -> std::string
{
    return "\\operatorname{atan2}(" + y_->latex() + ", " + x_->latex() + ")";
}

auto ATan2::python() const -> std::string
{
    return "atan2(" + y_->python() + ", " + x_->python() + ")";
}

// ===========================================================================
// Phase 5 — factory functions
// ===========================================================================

auto make_function(ResourceList& rl, FunctionKind kind, Expr* arg) -> Expr*
{
    return rl.make<FunctionApply>(kind, arg);
}

auto make_exp(ResourceList& rl, Expr* arg) -> Expr*
{
    return make_function(rl, FunctionKind::Exp, arg);
}
auto make_log(ResourceList& rl, Expr* arg) -> Expr*
{
    return make_function(rl, FunctionKind::Log, arg);
}
auto make_sin(ResourceList& rl, Expr* arg) -> Expr*
{
    return make_function(rl, FunctionKind::Sin, arg);
}
auto make_cos(ResourceList& rl, Expr* arg) -> Expr*
{
    return make_function(rl, FunctionKind::Cos, arg);
}
auto make_tan(ResourceList& rl, Expr* arg) -> Expr*
{
    return make_function(rl, FunctionKind::Tan, arg);
}
auto make_asin(ResourceList& rl, Expr* arg) -> Expr*
{
    return make_function(rl, FunctionKind::ASin, arg);
}
auto make_acos(ResourceList& rl, Expr* arg) -> Expr*
{
    return make_function(rl, FunctionKind::ACos, arg);
}
auto make_atan(ResourceList& rl, Expr* arg) -> Expr*
{
    return make_function(rl, FunctionKind::ATan, arg);
}
auto make_sinh(ResourceList& rl, Expr* arg) -> Expr*
{
    return make_function(rl, FunctionKind::Sinh, arg);
}
auto make_cosh(ResourceList& rl, Expr* arg) -> Expr*
{
    return make_function(rl, FunctionKind::Cosh, arg);
}
auto make_tanh(ResourceList& rl, Expr* arg) -> Expr*
{
    return make_function(rl, FunctionKind::Tanh, arg);
}
auto make_sqrt(ResourceList& rl, Expr* arg) -> Expr*
{
    return make_function(rl, FunctionKind::Sqrt, arg);
}

auto make_pow(ResourceList& rl, Expr* base, Rational exp) -> Expr*
{
    if (base->rank() != 0)
        throw std::invalid_argument(
            "make_pow: base must be a scalar (rank 0), got rank "
            + std::to_string(base->rank()));
    if (exp.is_zero())
        return make_rational(rl, Rational{1});
    if (exp == Rational{1})
        return base;
    return rl.make<Pow>(base, std::move(exp));
}

auto make_atan2(ResourceList& rl, Expr* y, Expr* x) -> Expr*
{
    return rl.make<ATan2>(y, x);
}

auto derivative_of(ResourceList& rl, FunctionKind kind, Expr* arg) -> Expr*
{
    switch (kind)
    {
        case FunctionKind::Exp:
            // d/dx exp(x) = exp(x)
            return make_exp(rl, arg);
        case FunctionKind::Log:
            // d/dx ln(x) = x^(-1)
            return make_pow(rl, arg, Rational{-1});
        case FunctionKind::Sin:
            // d/dx sin(x) = cos(x)
            return make_cos(rl, arg);
        case FunctionKind::Cos:
            // d/dx cos(x) = -sin(x)
            return make_scale(rl, Rational{-1}, make_sin(rl, arg));
        case FunctionKind::Tan:
            // d/dx tan(x) = cos(x)^(-2) = sec^2(x)
            return make_pow(rl, make_cos(rl, arg), Rational{-2});
        case FunctionKind::ASin:
            // d/dx asin(x) = (1 - x^2)^(-1/2)
            return make_pow(
                rl,
                make_sum(
                    rl,
                    {make_rational(rl, Rational{1}),
                     make_scale(
                         rl, Rational{-1}, make_pow(rl, arg, Rational{2}))}),
                Rational{-1, 2});
        case FunctionKind::ACos:
            // d/dx acos(x) = -(1 - x^2)^(-1/2)
            return make_scale(
                rl, Rational{-1}, derivative_of(rl, FunctionKind::ASin, arg));
        case FunctionKind::ATan:
            // d/dx atan(x) = (1 + x^2)^(-1)
            return make_pow(
                rl,
                make_sum(
                    rl,
                    {make_rational(rl, Rational{1}),
                     make_pow(rl, arg, Rational{2})}),
                Rational{-1});
        case FunctionKind::Sinh:
            // d/dx sinh(x) = cosh(x)
            return make_cosh(rl, arg);
        case FunctionKind::Cosh:
            // d/dx cosh(x) = sinh(x)
            return make_sinh(rl, arg);
        case FunctionKind::Tanh:
            // d/dx tanh(x) = cosh(x)^(-2) = sech^2(x)
            return make_pow(rl, make_cosh(rl, arg), Rational{-2});
        case FunctionKind::Sqrt:
            // d/dx sqrt(x) = (1/2) * x^(-1/2)
            return make_scale(
                rl, Rational{1, 2}, make_pow(rl, arg, Rational{-1, 2}));
    }
    return nullptr; // GCOV_EXCL_LINE
}

auto derivative_of_pow(ResourceList& rl, Expr* base, Rational exp) -> Expr*
{
    // d/dbase pow(base, exp) = exp * pow(base, exp - 1)
    return make_scale(rl, exp, make_pow(rl, base, exp - Rational{1}));
}

// ===========================================================================
// Phase 4 — LaTeX helper for binary operation nodes
// ===========================================================================

// Wrap a Sum sub-expression in parentheses so binary operators render cleanly.
static auto latex_operand(Expr const* e) -> std::string
{
    if (dynamic_cast<Sum const*>(e) || dynamic_cast<CrossProduct const*>(e))
        return "(" + e->latex() + ")";
    return e->latex();
}

// True for expressions that render as a single atomic token (no space-separated
// sub-expressions), so they don't need parentheses when used as a scalar
// factor.
static auto is_atomic_scalar(Expr const* e) -> bool
{
    return dynamic_cast<RationalConst const*>(e)
           || dynamic_cast<NamedConst const*>(e)
           || dynamic_cast<SymbolicVar const*>(e)
           || dynamic_cast<NamedTensor const*>(e)
           || dynamic_cast<FunctionApply const*>(e)
           || dynamic_cast<Pow const*>(e);
}

// ===========================================================================
// IdentityTensor
// ===========================================================================

auto IdentityTensor::latex() const -> std::string
{
    return "\\mathbf{I}";
}

auto IdentityTensor::python() const -> std::string
{
    return "I";
}

// ===========================================================================
// LeviCivitaTensor
// ===========================================================================

auto LeviCivitaTensor::latex() const -> std::string
{
    return "\\boldsymbol{\\varepsilon}";
}

auto LeviCivitaTensor::python() const -> std::string
{
    return "eps";
}

// ===========================================================================
// Trace
// ===========================================================================

Trace::Trace(Expr* arg) : arg_(arg)
{
    if (arg->rank() != 2)
        throw std::invalid_argument(
            "Trace: argument must have rank 2, got rank "
            + std::to_string(arg->rank()));
}

auto Trace::latex() const -> std::string
{
    return "\\mathrm{tr}(" + arg_->latex() + ")";
}

auto Trace::python() const -> std::string
{
    return "trace(" + arg_->python() + ")";
}

// ===========================================================================
// Contract
// ===========================================================================

auto Contract::latex() const -> std::string
{
    return latex_operand(lhs_) + " \\cdot " + latex_operand(rhs_);
}

auto Contract::python() const -> std::string
{
    return "dot(" + lhs_->python() + ", " + rhs_->python() + ")";
}

// ===========================================================================
// DoubleContract
// ===========================================================================

auto DoubleContract::latex() const -> std::string
{
    return latex_operand(lhs_) + " : " + latex_operand(rhs_);
}

auto DoubleContract::python() const -> std::string
{
    return "ddot(" + lhs_->python() + ", " + rhs_->python() + ")";
}

// ===========================================================================
// DoubleContractReversed
// ===========================================================================

auto DoubleContractReversed::latex() const -> std::string
{
    return latex_operand(lhs_) + " \\cdot\\!\\cdot " + latex_operand(rhs_);
}

auto DoubleContractReversed::python() const -> std::string
{
    return "ddot_rev(" + lhs_->python() + ", " + rhs_->python() + ")";
}

// ===========================================================================
// CrossProduct
// ===========================================================================

CrossProduct::CrossProduct(Expr* lhs, Expr* rhs) :
  lhs_(lhs), rhs_(rhs), rank_(lhs->rank() + rhs->rank() - 1)
{
}

auto CrossProduct::latex() const -> std::string
{
    return latex_operand(lhs_) + " \\times " + latex_operand(rhs_);
}

auto CrossProduct::python() const -> std::string
{
    return "cross(" + lhs_->python() + ", " + rhs_->python() + ")";
}

// ===========================================================================
// Factory functions — Phase 4
// ===========================================================================

auto make_identity(ResourceList& rl) -> Expr*
{
    return rl.make<IdentityTensor>();
}

auto make_levi_civita(ResourceList& rl) -> Expr*
{
    return rl.make<LeviCivitaTensor>();
}

auto make_trace(ResourceList& rl, Expr* arg) -> Expr*
{
    return rl.make<Trace>(arg);
}

auto make_contract(ResourceList& rl, Expr* lhs, Expr* rhs) -> Expr*
{
    if (lhs->rank() < 1 || rhs->rank() < 1)
        throw std::invalid_argument(
            "make_contract: both operands must have rank ≥ 1");
    if (dynamic_cast<IdentityTensor*>(lhs))
        return rhs;
    if (dynamic_cast<IdentityTensor*>(rhs))
        return lhs;
    return rl.make<Contract>(lhs, rhs);
}

auto make_double_contract(ResourceList& rl, Expr* lhs, Expr* rhs) -> Expr*
{
    if (lhs->rank() < 2 || rhs->rank() < 2)
        throw std::invalid_argument(
            "make_double_contract: both operands must have rank ≥ 2");
    if (dynamic_cast<IdentityTensor*>(lhs) && rhs->rank() == 2)
        return make_trace(rl, rhs);
    return rl.make<DoubleContract>(lhs, rhs);
}

auto make_double_contract_reversed(ResourceList& rl, Expr* lhs, Expr* rhs)
    -> Expr*
{
    if (lhs->rank() < 2 || rhs->rank() < 2)
        throw std::invalid_argument(
            "make_double_contract_reversed: both operands must have rank ≥ 2");
    return rl.make<DoubleContractReversed>(lhs, rhs);
}

auto make_cross_product(ResourceList& rl, Expr* lhs, Expr* rhs) -> Expr*
{
    if (lhs->rank() < 1 || rhs->rank() < 1)
        throw std::invalid_argument(
            "make_cross_product: both operands must have rank ≥ 1");
    if (dynamic_cast<CrossProduct*>(lhs) || dynamic_cast<CrossProduct*>(rhs))
        throw std::invalid_argument(
            "CrossProduct: chaining is ambiguous — wrap the intermediate "
            "result with named() to make parenthesisation explicit");
    return rl.make<CrossProduct>(lhs, rhs);
}

// ===========================================================================
// Phase 6 — Parameter
// ===========================================================================

auto Parameter::python() const -> std::string
{
    return "parameter('" + symbol() + "')";
}

// ===========================================================================
// Phase 6 — Product
// ===========================================================================

Product::Product(Expr* lhs, Expr* rhs) : lhs_(lhs), rhs_(rhs)
{
    if (lhs->rank() != 0 || rhs->rank() != 0)
        throw std::invalid_argument(
            "Product: both operands must be scalar (rank 0)");
}

auto Product::latex() const -> std::string
{
    return lhs_->latex() + " \\cdot " + rhs_->latex();
}

auto Product::python() const -> std::string
{
    return "prod(" + lhs_->python() + ", " + rhs_->python() + ")";
}

// ===========================================================================
// Phase 6 — MaterialDeriv
// ===========================================================================

MaterialDeriv::MaterialDeriv(Expr* velocity, Expr* field) :
  velocity_(velocity), field_(field), rank_(field->rank())
{
    if (velocity->rank() != 1)
        throw std::invalid_argument(
            "MaterialDeriv: velocity must have rank 1, got rank "
            + std::to_string(velocity->rank()));
}

auto MaterialDeriv::latex() const -> std::string
{
    return "\\frac{\\mathrm{D}}{\\mathrm{D}t}\\left(" + field_->latex()
           + "\\right)";
}

auto MaterialDeriv::python() const -> std::string
{
    return "material_deriv(" + velocity_->python() + ", " + field_->python()
           + ")";
}

// ===========================================================================
// Phase 6 — factories: make_parameter, make_product, time_parameter
// ===========================================================================

auto make_parameter(ResourceList& rl, std::string symbol) -> Parameter*
{
    return rl.make<Parameter>(std::move(symbol));
}

auto make_product(ResourceList& rl, Expr* lhs, Expr* rhs) -> Expr*
{
    if (auto* lc = dynamic_cast<RationalConst*>(lhs))
        return make_scale(rl, lc->value(), rhs);
    if (auto* rc = dynamic_cast<RationalConst*>(rhs))
        return make_scale(rl, rc->value(), lhs);
    return rl.make<Product>(lhs, rhs);
}

static ResourceList s_builtin_rl;
static Parameter* s_time_param = s_builtin_rl.make<Parameter>("t");

auto time_parameter() -> Parameter const*
{
    return s_time_param;
}

// ===========================================================================
// Phase 6 — depends_on
// ===========================================================================

static auto depends_on_impl(std::string const& sym, Expr const* e) -> bool
{
    if (auto const* p = dynamic_cast<Parameter const*>(e))
        return p->symbol() == sym;
    if (auto const* sc = dynamic_cast<Scale const*>(e))
        return depends_on_impl(sym, sc->expr());
    if (auto const* s = dynamic_cast<Sum const*>(e))
    {
        for (auto const* t: s->terms())
            if (depends_on_impl(sym, t))
                return true;
        return false;
    }
    if (auto const* tp = dynamic_cast<TensorProduct const*>(e))
        return depends_on_impl(sym, tp->lhs())
               || depends_on_impl(sym, tp->rhs());
    if (auto const* fa = dynamic_cast<FunctionApply const*>(e))
        return depends_on_impl(sym, fa->arg());
    if (auto const* pw = dynamic_cast<Pow const*>(e))
        return depends_on_impl(sym, pw->base());
    if (auto const* a2 = dynamic_cast<ATan2 const*>(e))
        return depends_on_impl(sym, a2->y()) || depends_on_impl(sym, a2->x());
    if (auto const* pr = dynamic_cast<Product const*>(e))
        return depends_on_impl(sym, pr->lhs())
               || depends_on_impl(sym, pr->rhs());
    if (auto const* tr = dynamic_cast<Trace const*>(e))
        return depends_on_impl(sym, tr->arg());
    if (auto const* co = dynamic_cast<Contract const*>(e))
        return depends_on_impl(sym, co->lhs())
               || depends_on_impl(sym, co->rhs());
    if (auto const* dc = dynamic_cast<DoubleContract const*>(e))
        return depends_on_impl(sym, dc->lhs())
               || depends_on_impl(sym, dc->rhs());
    if (auto const* dr = dynamic_cast<DoubleContractReversed const*>(e))
        return depends_on_impl(sym, dr->lhs())
               || depends_on_impl(sym, dr->rhs());
    if (auto const* cp = dynamic_cast<CrossProduct const*>(e))
        return depends_on_impl(sym, cp->lhs())
               || depends_on_impl(sym, cp->rhs());
    if (auto const* md = dynamic_cast<MaterialDeriv const*>(e))
        return depends_on_impl(sym, md->velocity())
               || depends_on_impl(sym, md->field());
    if (auto const* pe = dynamic_cast<PolynomialExpr const*>(e))
        return depends_on_impl(sym, pe->var());
    return false;
}

auto depends_on(Parameter const* p, Expr const* e) -> bool
{
    return depends_on_impl(p->symbol(), e);
}

// ===========================================================================
// Phase 6 — deriv
// ===========================================================================

auto deriv(ResourceList& rl, Parameter const* p, Expr* e) -> Expr*
{
    if (!depends_on(p, e))
        return make_rational(rl, Rational{0});

    // Base case: the parameter itself
    if (dynamic_cast<Parameter*>(e))
        return make_rational(rl, Rational{1});

    // Scale(c, inner): d/dp = c * d(inner)/dp
    if (auto* sc = dynamic_cast<Scale*>(e))
        return make_scale(rl, sc->coeff(), deriv(rl, p, sc->expr()));

    // Sum: linearity
    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> dterms;
        dterms.reserve(s->terms().size());
        for (auto* t: s->terms())
            dterms.push_back(deriv(rl, p, t));
        return make_sum(rl, std::move(dterms));
    }

    // TensorProduct: Leibniz rule
    if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        bool ld = depends_on(p, tp->lhs());
        bool rd = depends_on(p, tp->rhs());
        if (ld && rd)
            return make_sum(
                rl,
                {make_tensor_product(rl, deriv(rl, p, tp->lhs()), tp->rhs()),
                 make_tensor_product(rl, tp->lhs(), deriv(rl, p, tp->rhs()))});
        if (ld)
            return make_tensor_product(rl, deriv(rl, p, tp->lhs()), tp->rhs());
        return make_tensor_product(rl, tp->lhs(), deriv(rl, p, tp->rhs()));
    }

    // FunctionApply: chain rule d/dp f(g) = f'(g) * dg/dp
    if (auto* fa = dynamic_cast<FunctionApply*>(e))
    {
        auto* f_prime = derivative_of(rl, fa->kind(), fa->arg());
        auto* g_prime = deriv(rl, p, fa->arg());
        return make_product(rl, f_prime, g_prime);
    }

    // Pow: chain rule d/dp base^n = n*base^(n-1) * d(base)/dp
    if (auto* pw = dynamic_cast<Pow*>(e))
    {
        auto* pw_prime = derivative_of_pow(rl, pw->base(), pw->exponent());
        auto* base_prime = deriv(rl, p, pw->base());
        return make_product(rl, pw_prime, base_prime);
    }

    // Product: Leibniz rule (scalar case)
    if (auto* pr = dynamic_cast<Product*>(e))
    {
        bool ld = depends_on(p, pr->lhs());
        bool rd = depends_on(p, pr->rhs());
        if (ld && rd)
            return make_sum(
                rl,
                {make_product(rl, deriv(rl, p, pr->lhs()), pr->rhs()),
                 make_product(rl, pr->lhs(), deriv(rl, p, pr->rhs()))});
        if (ld)
            return make_product(rl, deriv(rl, p, pr->lhs()), pr->rhs());
        return make_product(rl, pr->lhs(), deriv(rl, p, pr->rhs()));
    }

    // Trace: d/dp tr(A) = tr(dA/dp)  — valid when dA/dp has rank 2
    if (auto* tr = dynamic_cast<Trace*>(e))
        return make_trace(rl, deriv(rl, p, tr->arg()));

    // Contract: Leibniz rule  d/dp (a·b) = da/dp·b + a·db/dp
    if (auto* co = dynamic_cast<Contract*>(e))
    {
        bool ld = depends_on(p, co->lhs());
        bool rd = depends_on(p, co->rhs());
        if (ld && rd)
            return make_sum(
                rl,
                {make_contract(rl, deriv(rl, p, co->lhs()), co->rhs()),
                 make_contract(rl, co->lhs(), deriv(rl, p, co->rhs()))});
        if (ld)
            return make_contract(rl, deriv(rl, p, co->lhs()), co->rhs());
        return make_contract(rl, co->lhs(), deriv(rl, p, co->rhs()));
    }

    // DoubleContract: Leibniz rule
    if (auto* dc = dynamic_cast<DoubleContract*>(e))
    {
        bool ld = depends_on(p, dc->lhs());
        bool rd = depends_on(p, dc->rhs());
        if (ld && rd)
            return make_sum(
                rl,
                {make_double_contract(rl, deriv(rl, p, dc->lhs()), dc->rhs()),
                 make_double_contract(rl, dc->lhs(), deriv(rl, p, dc->rhs()))});
        if (ld)
            return make_double_contract(rl, deriv(rl, p, dc->lhs()), dc->rhs());
        return make_double_contract(rl, dc->lhs(), deriv(rl, p, dc->rhs()));
    }

    // DoubleContractReversed: Leibniz rule
    if (auto* dr = dynamic_cast<DoubleContractReversed*>(e))
    {
        bool ld = depends_on(p, dr->lhs());
        bool rd = depends_on(p, dr->rhs());
        if (ld && rd)
            return make_sum(
                rl,
                {make_double_contract_reversed(
                     rl, deriv(rl, p, dr->lhs()), dr->rhs()),
                 make_double_contract_reversed(
                     rl, dr->lhs(), deriv(rl, p, dr->rhs()))});
        if (ld)
            return make_double_contract_reversed(
                rl, deriv(rl, p, dr->lhs()), dr->rhs());
        return make_double_contract_reversed(
            rl, dr->lhs(), deriv(rl, p, dr->rhs()));
    }

    // CrossProduct: Leibniz rule
    if (auto* cp = dynamic_cast<CrossProduct*>(e))
    {
        bool ld = depends_on(p, cp->lhs());
        bool rd = depends_on(p, cp->rhs());
        if (ld && rd)
            return make_sum(
                rl,
                {make_cross_product(rl, deriv(rl, p, cp->lhs()), cp->rhs()),
                 make_cross_product(rl, cp->lhs(), deriv(rl, p, cp->rhs()))});
        if (ld)
            return make_cross_product(rl, deriv(rl, p, cp->lhs()), cp->rhs());
        return make_cross_product(rl, cp->lhs(), deriv(rl, p, cp->rhs()));
    }

    // PolynomialExpr: chain rule  d/dp [p(v)] = p'(v) · dv/dp
    // Only rank-0 variables are supported; rank-2 NamedTensor vars always
    // report no parameter dependency so can never reach here.
    if (auto* pe = dynamic_cast<PolynomialExpr*>(e))
    {
        auto dp = pe->poly().diff();
        if (dp.is_zero())
            return make_rational(rl, Rational{0});
        auto* dp_expr = rl.make<PolynomialExpr>(std::move(dp), pe->var());
        auto* dv = deriv(rl, p, pe->var());
        return make_product(rl, dp_expr, dv);
    }

    return make_rational(rl, Rational{0}); // GCOV_EXCL_LINE
}

// ===========================================================================
// Phase 6 — dt, ddt, make_material_deriv
// ===========================================================================

auto dt(ResourceList& rl, Expr* e) -> Expr*
{
    return deriv(rl, s_time_param, e);
}

auto ddt(ResourceList& rl, Expr* e) -> Expr*
{
    return deriv(rl, s_time_param, deriv(rl, s_time_param, e));
}

auto make_material_deriv(ResourceList& rl, Expr* velocity, Expr* field) -> Expr*
{
    return rl.make<MaterialDeriv>(velocity, field);
}

} // namespace tender
