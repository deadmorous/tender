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
static auto rational_to_latex_abs(Rational const& r) -> std::string
{
    Rational const abs_r(r.num() < 0 ? -r.num() : r.num(), r.den());
    return rational_to_latex(abs_r);
}

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
        return rc->value().num() < 0;
    if (auto const* sc = dynamic_cast<Scale const*>(e))
        return sc->coeff().num() < 0;
    return false;
}

// LaTeX for e with its leading sign stripped (for use after an explicit " - ").
static auto latex_unsigned(Expr const* e) -> std::string
{
    if (auto const* rc = dynamic_cast<RationalConst const*>(e))
        return rational_to_latex_abs(rc->value());
    if (auto const* sc = dynamic_cast<Scale const*>(e))
    {
        Rational const abs_coeff(
            sc->coeff().num() < 0 ? -sc->coeff().num() : sc->coeff().num(),
            sc->coeff().den());
        if (abs_coeff == Rational{1})
            return sc->expr()->latex();
        return rational_to_latex(abs_coeff) + " " + sc->expr()->latex();
    }
    return e->latex();
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
            return;
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

auto make_tensor_product(ResourceList& rl, Expr* lhs, Expr* rhs) -> Expr*
{
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

} // namespace tender
