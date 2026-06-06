#pragma once

#include <tender/derivation.hpp>
#include <tender/expr.hpp>

#include <string>

namespace tender
{

// ===========================================================================
// Domain types — geometric regions of integration
// ===========================================================================

class Domain
{
public:
    virtual ~Domain() = default;
    virtual auto name() const noexcept -> std::string const& = 0;
    // Next lower-dimensional boundary; nullptr at the bottom (Point).
    virtual auto boundary() const -> Domain* = 0;
    // LaTeX measure symbol appended to \int (e.g. \mathrm{d}V).
    virtual auto measure_latex() const -> std::string = 0;
    virtual auto measure_python() const -> std::string = 0;
};

// Two-dimensional surface embedded in 3D. Carries an outward normal.
class SurfaceDomain : public Domain
{
public:
    SurfaceDomain(std::string name, Expr* normal);
    auto name() const noexcept -> std::string const& override
    {
        return name_;
    }
    auto boundary() const -> Domain* override
    {
        return nullptr;
    }
    auto measure_latex() const -> std::string override
    {
        return "\\mathrm{d}S";
    }
    auto measure_python() const -> std::string override
    {
        return "dS";
    }
    auto normal() const noexcept -> Expr*
    {
        return normal_;
    }

private:
    std::string name_;
    Expr* normal_;
};

// Three-dimensional region. Carries an outward normal on its surface boundary.
class VolumeDomain : public Domain
{
public:
    VolumeDomain(std::string name, Expr* outward_normal, SurfaceDomain* bdy);
    auto name() const noexcept -> std::string const& override
    {
        return name_;
    }
    auto boundary() const -> Domain* override
    {
        return boundary_;
    }
    auto surface_boundary() const noexcept -> SurfaceDomain*
    {
        return boundary_;
    }
    auto measure_latex() const -> std::string override
    {
        return "\\mathrm{d}V";
    }
    auto measure_python() const -> std::string override
    {
        return "dV";
    }
    // Outward normal on the boundary surface (rank-1 Expr).
    auto outward_normal() const noexcept -> Expr*
    {
        return normal_;
    }

private:
    std::string name_;
    Expr* normal_;
    SurfaceDomain* boundary_;
};

// Factories. make_volume_domain automatically creates the boundary
// SurfaceDomain (name = "\\partial " + name) using the same outward normal.
auto make_surface_domain(ResourceList& rl, std::string name, Expr* normal)
    -> SurfaceDomain*;
auto make_volume_domain(
    ResourceList& rl, std::string name, Expr* outward_normal) -> VolumeDomain*;

// ===========================================================================
// Gradient node — symbolic ∇ ⊗ expr; rank = arg->rank() + 1
// ===========================================================================

class Gradient : public Expr
{
public:
    explicit Gradient(Expr* arg);
    auto rank() const noexcept -> int override
    {
        return arg_->rank() + 1;
    }
    auto latex() const -> std::string override;
    auto python() const -> std::string override;
    auto arg() const noexcept -> Expr*
    {
        return arg_;
    }

private:
    Expr* arg_;
};

auto make_gradient(ResourceList& rl, Expr* arg) -> Expr*;

// ===========================================================================
// Divergence node — symbolic ∇ · expr; rank = arg->rank() - 1
// ===========================================================================

class Divergence : public Expr
{
public:
    explicit Divergence(Expr* arg);
    auto rank() const noexcept -> int override
    {
        return arg_->rank() - 1;
    }
    auto latex() const -> std::string override;
    auto python() const -> std::string override;
    auto arg() const noexcept -> Expr*
    {
        return arg_;
    }

private:
    Expr* arg_;
};

// Throws std::invalid_argument if arg->rank() < 1.
auto make_divergence(ResourceList& rl, Expr* arg) -> Expr*;

// ===========================================================================
// Rotor node — symbolic ∇ × expr; rank = arg->rank()
// ===========================================================================

class Rotor : public Expr
{
public:
    explicit Rotor(Expr* arg);
    auto rank() const noexcept -> int override
    {
        return arg_->rank();
    }
    auto latex() const -> std::string override;
    auto python() const -> std::string override;
    auto arg() const noexcept -> Expr*
    {
        return arg_;
    }

private:
    Expr* arg_;
};

// Throws std::invalid_argument if arg->rank() < 1.
auto make_rotor(ResourceList& rl, Expr* arg) -> Expr*;

// ===========================================================================
// Integral node — ∫_domain integrand d(measure); rank = integrand->rank()
// ===========================================================================

class Integral : public Expr
{
public:
    Integral(Domain* domain, Expr* integrand);
    auto rank() const noexcept -> int override
    {
        return integrand_->rank();
    }
    auto latex() const -> std::string override;
    auto python() const -> std::string override;
    auto domain() const noexcept -> Domain*
    {
        return domain_;
    }
    auto integrand() const noexcept -> Expr*
    {
        return integrand_;
    }

private:
    Domain* domain_;
    Expr* integrand_;
};

auto make_integral(ResourceList& rl, Domain* domain, Expr* integrand) -> Expr*;

// ===========================================================================
// Named derivation steps
// ===========================================================================

// Integration by parts on ∫_V A:(∇B) dV  (where A = any rank-2+ tensor,
// B = rank-1 vector, and Gradient(B) appears as rhs of DoubleContract):
//
//   ∫_V A:(∇B) dV  →  ∫_∂V (A·n)·B dS  −  ∫_V (∇·A)·B dV
//
// Tree-walks the whole expression; applies wherever Integral(domain, …)
// matches the pattern.
auto apply_integration_by_parts_step(VolumeDomain* domain) -> DerivationStep;

// Divergence theorem: ∫_V ∇·A dV → ∫_∂V A·n dS.
// Tree-walks; applies to every Integral(domain, Divergence(A)) found.
auto apply_divergence_theorem_step(VolumeDomain* domain) -> DerivationStep;

// Localization (fundamental lemma of calculus of variations):
// strips the integral wrapper from every Integral(domain, f) found,
// leaving the integrand f.  The semantic interpretation — that the integral
// vanishes for all test functions, hence the integrand vanishes pointwise —
// is supplied by the user.
auto localize_step(Domain* domain) -> DerivationStep;

} // namespace tender
