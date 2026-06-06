#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <tender/index.hpp>
#include <tender/rational.hpp>

namespace tender
{

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

    [[nodiscard]] auto slots() const noexcept -> SlotList const&
    {
        return slots_;
    }

protected:
    [[nodiscard]] auto has_name() const noexcept -> bool
    {
        return !name_.empty();
    }

    auto set_slots(SlotList s) -> void
    {
        slots_ = std::move(s);
    }

private:
    std::string name_;
    SlotList slots_;
};

// ===========================================================================
// Concrete nodes — scalars (rank 0)
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

// ===========================================================================
// Structural nodes
// ===========================================================================

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
// Indexed nodes (Phase 3)
// ===========================================================================

// Generic named tensor of any rank with explicit slot annotations.
class NamedTensor : public Expr
{
public:
    NamedTensor(std::string symbol, int rank, SlotList slots);

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
    }
    [[nodiscard]] auto latex() const -> std::string override;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto symbol() const noexcept -> std::string const&
    {
        return symbol_;
    }

private:
    std::string symbol_;
    int rank_;
};

// Annotation: force summation over `index` regardless of occurrence count.
// Rank = body.rank() − 2 (the matched upper+lower slot pair is consumed).
// Throws std::invalid_argument if body does not carry a matching upper+lower
// slot pair for the given index.
class ExplicitSum : public Expr
{
public:
    ExplicitSum(Expr* body, Index* index);

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
    }
    [[nodiscard]] auto latex() const -> std::string override;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto body() const noexcept -> Expr*
    {
        return body_;
    }
    [[nodiscard]] auto index() const noexcept -> Index*
    {
        return index_;
    }

private:
    Expr* body_;
    Index* index_;
    int rank_;
};

// Annotation: suppress automatic Einstein contraction over `index`.
// Rank = body.rank() (no contraction performed; free slots preserved).
class NoSum : public Expr
{
public:
    NoSum(Expr* body, Index* index);

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
    }
    [[nodiscard]] auto latex() const -> std::string override;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto body() const noexcept -> Expr*
    {
        return body_;
    }
    [[nodiscard]] auto index() const noexcept -> Index*
    {
        return index_;
    }

private:
    Expr* body_;
    Index* index_;
    int rank_;
};

// Result of convolve(): pairs one slot from lhs with one slot from rhs.
// Rank = lhs.rank() + rhs.rank() − 2.
class Contraction : public Expr
{
public:
    Contraction(
        Expr* lhs,
        std::size_t slot_lhs,
        Expr* rhs,
        std::size_t slot_rhs,
        int rank,
        SlotList slots);

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
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
    [[nodiscard]] auto slot_lhs() const noexcept -> std::size_t
    {
        return slot_lhs_;
    }
    [[nodiscard]] auto slot_rhs() const noexcept -> std::size_t
    {
        return slot_rhs_;
    }

private:
    Expr* lhs_;
    Expr* rhs_;
    std::size_t slot_lhs_;
    std::size_t slot_rhs_;
    int rank_;
};

// ===========================================================================
// Factory functions — scalar nodes (Phase 2)
// ===========================================================================

auto make_rational(ResourceList& rl, Rational r) -> Expr*;
auto make_named_const(ResourceList& rl, std::string sym) -> Expr*;
auto make_symbolic_var(ResourceList& rl, std::string sym) -> Expr*;
auto make_sum(ResourceList& rl, std::vector<Expr*> terms) -> Expr*;
auto make_scale(ResourceList& rl, Rational r, Expr* e) -> Expr*;
auto make_tensor_product(ResourceList& rl, Expr* lhs, Expr* rhs) -> Expr*;

// Set a display name on an expression and return it.
auto named(std::string n, Expr* e) -> Expr*;

// ===========================================================================
// Factory functions — indexed nodes (Phase 3)
// ===========================================================================

auto make_named_tensor(
    ResourceList& rl, std::string sym, int rank, SlotList slots) -> Expr*;
auto make_explicit_sum(ResourceList& rl, Expr* body, Index* index) -> Expr*;
auto make_no_sum(ResourceList& rl, Expr* body, Index* index) -> Expr*;

// Pair slot_a from a with slot_b from b. Exactly one slot must be Upper and
// the other Lower; throws std::invalid_argument on level mismatch or
// out-of-bounds slot indices.
auto convolve(
    ResourceList& rl, Expr* a, std::size_t slot_a, Expr* b, std::size_t slot_b)
    -> Expr*;

} // namespace tender
