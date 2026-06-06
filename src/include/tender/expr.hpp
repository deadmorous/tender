#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <mpk/mix/util/resource_list.hpp>

#include <tender/rational.hpp>

namespace tender
{

using ResourceList = mpk::mix::ResourceList;

// ===========================================================================
// Abstract base
// ===========================================================================

class Expr
{
public:
    virtual ~Expr() = default;

    [[nodiscard]] virtual auto rank() const noexcept -> int = 0;
    [[nodiscard]] virtual auto latex() const -> std::string = 0;
    [[nodiscard]] virtual auto python() const -> std::string = 0;

    [[nodiscard]] auto name() const noexcept -> std::string const&
    {
        return name_;
    }

    // Set a display name. Idempotent; throws if a different name is already
    // set.
    auto set_name(std::string n) -> void;

protected:
    [[nodiscard]] auto has_name() const noexcept -> bool
    {
        return !name_.empty();
    }

private:
    std::string name_;
};

// ===========================================================================
// Concrete nodes
// ===========================================================================

// Exact rational constant. Rank 0.
class RationalConst : public Expr
{
public:
    explicit RationalConst(Rational value) : value_(std::move(value))
    {
    }

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 0;
    }
    [[nodiscard]] auto latex() const -> std::string override;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto value() const noexcept -> Rational const&
    {
        return value_;
    }

private:
    Rational value_;
};

// Named mathematical constant (π, e, …). Rank 0.
class NamedConst : public Expr
{
public:
    explicit NamedConst(std::string symbol) : symbol_(std::move(symbol))
    {
    }

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 0;
    }
    [[nodiscard]] auto latex() const -> std::string override;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto symbol() const noexcept -> std::string const&
    {
        return symbol_;
    }

private:
    std::string symbol_;
};

// Opaque symbolic scalar variable. Rank 0.
class SymbolicVar : public Expr
{
public:
    explicit SymbolicVar(std::string symbol) : symbol_(std::move(symbol))
    {
    }

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 0;
    }
    [[nodiscard]] auto latex() const -> std::string override;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto symbol() const noexcept -> std::string const&
    {
        return symbol_;
    }

private:
    std::string symbol_;
};

// Flat sum of normalised terms (no nested Sums; no Scale wrapping a Sum).
class Sum : public Expr
{
public:
    Sum(std::vector<Expr*> terms, int rank) :
      terms_(std::move(terms)), rank_(rank)
    {
    }

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
    }
    [[nodiscard]] auto latex() const -> std::string override;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto terms() const noexcept -> std::vector<Expr*> const&
    {
        return terms_;
    }

private:
    std::vector<Expr*> terms_;
    int rank_;
};

// Rational scalar coefficient × expression.
// Invariant: coeff != 0, coeff != 1, inner is not a Scale or RationalConst.
class Scale : public Expr
{
public:
    Scale(Rational coeff, Expr* expr) : coeff_(std::move(coeff)), expr_(expr)
    {
    }

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return expr_->rank();
    }
    [[nodiscard]] auto latex() const -> std::string override;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto coeff() const noexcept -> Rational const&
    {
        return coeff_;
    }
    [[nodiscard]] auto expr() const noexcept -> Expr*
    {
        return expr_;
    }

private:
    Rational coeff_;
    Expr* expr_;
};

// Tensor product a ⊗ b. Rank = lhs.rank + rhs.rank.
class TensorProduct : public Expr
{
public:
    TensorProduct(Expr* lhs, Expr* rhs) : lhs_(lhs), rhs_(rhs)
    {
    }

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return lhs_->rank() + rhs_->rank();
    }
    [[nodiscard]] auto latex() const -> std::string override;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto lhs() const noexcept -> Expr*
    {
        return lhs_;
    }
    [[nodiscard]] auto rhs() const noexcept -> Expr*
    {
        return rhs_;
    }

private:
    Expr* lhs_;
    Expr* rhs_;
};

// ===========================================================================
// Factory functions — apply always-on simplifications at construction
// ===========================================================================

auto make_rational(ResourceList& rl, Rational r) -> Expr*;
auto make_named_const(ResourceList& rl, std::string sym) -> Expr*;
auto make_symbolic_var(ResourceList& rl, std::string sym) -> Expr*;
auto make_sum(ResourceList& rl, std::vector<Expr*> terms) -> Expr*;
auto make_scale(ResourceList& rl, Rational r, Expr* e) -> Expr*;
auto make_tensor_product(ResourceList& rl, Expr* lhs, Expr* rhs) -> Expr*;

// Set a display name on an expression and return it.
auto named(std::string n, Expr* e) -> Expr*;

} // namespace tender
