#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <tender/derivation.hpp>
#include <tender/expr.hpp>

namespace tender
{

// ===========================================================================
// PatternConstraints
// ===========================================================================

struct PatternConstraints
{
    int required_rank = -1; // -1 = any rank
    bool symmetric = false;
    bool skew_symmetric = false;
};

// ===========================================================================
// PatternVar — typed placeholder used inside Identity patterns
// ===========================================================================

class PatternVar : public Expr
{
public:
    explicit PatternVar(std::string symbol);

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return constraints_.required_rank;
    }

    [[nodiscard]] auto latex() const -> std::string override;
    [[nodiscard]] auto python() const -> std::string override;

    [[nodiscard]] auto symbol() const noexcept -> std::string const&
    {
        return symbol_;
    }

    [[nodiscard]] auto constraints() const noexcept -> PatternConstraints const&
    {
        return constraints_;
    }

    // Fluent constraint setters — return this for chaining.
    auto constrain_rank(int r) -> PatternVar*;
    auto constrain_symmetric() -> PatternVar*;
    auto constrain_skew_symmetric() -> PatternVar*;

private:
    std::string symbol_;
    PatternConstraints constraints_;
};

// ===========================================================================
// Identity — directed rewrite rule  lhs → rhs
// ===========================================================================

class Identity
{
public:
    Identity(std::string name, Expr* lhs, Expr* rhs);

    [[nodiscard]] auto name() const noexcept -> std::string const&
    {
        return name_;
    }
    [[nodiscard]] auto lhs() const noexcept -> Expr*
    {
        return lhs_;
    }
    [[nodiscard]] auto rhs() const noexcept -> Expr*
    {
        return rhs_;
    }

    // Promote the first and last states of a derivation history to an identity.
    static auto from_derivation(
        std::string name, std::vector<State> const& history) -> Identity;

private:
    std::string name_;
    Expr* lhs_;
    Expr* rhs_;
};

// ===========================================================================
// apply_identity — manual-targeting application
// ===========================================================================

using PatternMapping = std::unordered_map<PatternVar const*, Expr*>;

// Returns a DerivationStep that substitutes mapping into id.rhs().
// Validates rank constraints; symmetry constraints are noted but not yet
// enforced (deferred until the constraint cache is introduced).
auto apply_identity(Identity const& id, PatternMapping const& mapping)
    -> DerivationStep;

// Returns the RHS expression with pattern variables substituted by mapping.
// No rank/constraint validation — caller is responsible.
auto apply_rhs(
    ResourceList& rl,
    Identity const& id,
    PatternMapping const& mapping) -> Expr*;

// ===========================================================================
// Factory
// ===========================================================================

auto make_pattern_var(ResourceList& rl, std::string symbol) -> PatternVar*;

} // namespace tender
