#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <tender/index.hpp>
#include <tender/polynomial.hpp>
#include <tender/rational.hpp>

namespace tender
{

class CoordSystem; // forward declaration — defined in coord_system.hpp

using IndexNameMap = std::unordered_map<int, std::string>;

struct EnrichedExpr;

// ===========================================================================
// Abstract base
// ===========================================================================

class Expr
{
public:
    virtual ~Expr() = default;

    [[nodiscard]] virtual auto rank() const noexcept -> int = 0;
    [[nodiscard]] virtual auto latex(IndexNameMap const& map) const
        -> std::string = 0;
    [[nodiscard]] virtual auto python() const -> std::string = 0;

    // Non-virtual convenience: calls enrich(this).latex()
    [[nodiscard]] auto latex() const -> std::string;

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
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto symbol() const noexcept -> std::string const&
    {
        return symbol_;
    }

private:
    std::string symbol_;
};

// Named scalar parameter (time, load parameter, …). Rank 0.
// Identity is by name: two Parameters with the same symbol are the same
// parameter for dependency-tracking and differentiation purposes.
class Parameter : public SymbolicVar
{
public:
    using SymbolicVar::SymbolicVar;
    [[nodiscard]] auto python() const -> std::string override;
    // latex() inherited from SymbolicVar
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
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto symbol() const noexcept -> std::string const&
    {
        return symbol_;
    }

    [[nodiscard]] auto is_symmetric() const noexcept -> bool
    {
        return symmetric_;
    }
    [[nodiscard]] auto is_skew_symmetric() const noexcept -> bool
    {
        return skew_symmetric_;
    }
    auto declare_symmetric() -> void
    {
        symmetric_ = true;
    }
    auto declare_skew_symmetric() -> void
    {
        skew_symmetric_ = true;
    }

private:
    std::string symbol_;
    int rank_;
    bool symmetric_ = false;
    bool skew_symmetric_ = false;
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
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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

// Display-only terminal node for Einstein indexed-sum notation.
// Renders e.g. "a^{i} b_{i}" from base symbols and separator characters.
// Not rewritten by any rule; produced by collect_repeated_sum_step.
class IndexedSum : public Expr
{
public:
    IndexedSum(
        std::string lhs_sym,
        std::string lhs_sep,
        std::string rhs_sym,
        std::string rhs_sep,
        std::string index_letter,
        int rank);

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;

    [[nodiscard]] auto lhs_sym() const noexcept -> std::string const&
    {
        return lhs_sym_;
    }
    [[nodiscard]] auto lhs_sep() const noexcept -> std::string const&
    {
        return lhs_sep_;
    }
    [[nodiscard]] auto rhs_sym() const noexcept -> std::string const&
    {
        return rhs_sym_;
    }
    [[nodiscard]] auto rhs_sep() const noexcept -> std::string const&
    {
        return rhs_sep_;
    }
    [[nodiscard]] auto index_letter() const noexcept -> std::string const&
    {
        return index_letter_;
    }

private:
    std::string lhs_sym_;
    std::string lhs_sep_;
    std::string rhs_sym_;
    std::string rhs_sep_;
    std::string index_letter_;
    int rank_;
};

// Abstract symbolic basis vector e_i or cobasis e^i.
// Used for index-notation derivations without concrete component expansion:
//   expand_in_basis_step(v, cs, abstract=True)  →  TensorProduct(AbstractComp,
//   SymBasisVec(cs, index_id, false))
// simplify_basis_dot_step recognises Contract(TP(AbstractComp, SBV),
// TP(AbstractComp, SBV)) and produces AbstractIndexedSum.
class SymBasisVec : public Expr
{
public:
    SymBasisVec(CoordSystem const& cs, int index_id, bool cobasis);

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 1;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto cs() const noexcept -> CoordSystem const&
    {
        return *cs_;
    }
    [[nodiscard]] auto index_id() const noexcept -> int
    {
        return index_id_;
    }
    [[nodiscard]] auto is_cobasis() const noexcept -> bool
    {
        return cobasis_;
    }

private:
    CoordSystem const* cs_;
    int index_id_;
    bool cobasis_;
};

auto make_sym_basis_vec(
    ResourceList& rl,
    CoordSystem const& cs,
    int index_id,
    bool cobasis) -> Expr*;

// Abstract component: rank-0 node holding a base symbol and a list of
// (index_id, is_upper) pairs. Replaces the NamedTensor("a^i", 0) hack for
// the abstract-basis derivation path.
class AbstractComp : public Expr
{
public:
    using IndexSpec = std::vector<std::pair<int, bool>>;

    AbstractComp(std::string base_sym, IndexSpec indices);

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 0;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto base_sym() const noexcept -> std::string const&
    {
        return base_sym_;
    }
    [[nodiscard]] auto indices() const noexcept -> IndexSpec const&
    {
        return indices_;
    }

private:
    std::string base_sym_;
    IndexSpec indices_;
};

auto make_abstract_comp(
    ResourceList& rl,
    std::string base_sym,
    std::vector<std::pair<int, bool>> indices) -> AbstractComp*;

// Display-only node for abstract Einstein-notation indexed sum.
// Produced by simplify_basis_dot_step when contracting two
// AbstractComp-SBV tensor products.
class AbstractIndexedSum : public Expr
{
public:
    AbstractIndexedSum(
        AbstractComp const* lhs,
        AbstractComp const* rhs,
        int index_id,
        int rank);

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto lhs() const noexcept -> AbstractComp const*
    {
        return lhs_;
    }
    [[nodiscard]] auto rhs() const noexcept -> AbstractComp const*
    {
        return rhs_;
    }
    [[nodiscard]] auto index_id() const noexcept -> int
    {
        return index_id_;
    }

private:
    AbstractComp const* lhs_;
    AbstractComp const* rhs_;
    int index_id_;
    int rank_;
};

auto make_abstract_indexed_sum(
    ResourceList& rl,
    AbstractComp const* lhs,
    AbstractComp const* rhs,
    int index_id,
    int rank) -> AbstractIndexedSum*;

// Kronecker delta δ_{lower}^{upper}.  Rank 0.
// Factory make_kronecker_delta returns Expr* so it can fold equal IDs → 1.
class KroneckerDelta : public Expr
{
public:
    KroneckerDelta(int lower_id, int upper_id) :
      lower_id_(lower_id), upper_id_(upper_id)
    {
    }

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 0;
    }
    [[nodiscard]] auto latex(IndexNameMap const&) const -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto lower_id() const noexcept -> int
    {
        return lower_id_;
    }
    [[nodiscard]] auto upper_id() const noexcept -> int
    {
        return upper_id_;
    }

private:
    int lower_id_;
    int upper_id_;
};

// Returns RationalConst(1) when lower_id == upper_id (Rule 2: trace = 1 slot).
auto make_kronecker_delta(ResourceList& rl, int lower_id, int upper_id) -> Expr*;

// Levi-Civita symbol ε with three abstract index IDs and upper/lower flags.
// Distinct from LeviCivitaTensor (the invariant rank-3 tensor object).
class LeviCivitaSymbol : public Expr
{
public:
    LeviCivitaSymbol(std::array<int, 3> ids, std::array<bool, 3> upper) :
      ids_(ids), upper_(upper)
    {
    }

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 0;
    }
    [[nodiscard]] auto latex(IndexNameMap const&) const -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto ids() const noexcept -> std::array<int, 3> const&
    {
        return ids_;
    }
    [[nodiscard]] auto upper() const noexcept -> std::array<bool, 3> const&
    {
        return upper_;
    }

private:
    std::array<int, 3> ids_;
    std::array<bool, 3> upper_;
};

auto make_levi_civita_symbol(
    ResourceList& rl,
    std::array<int, 3> ids,
    std::array<bool, 3> upper) -> LeviCivitaSymbol*;

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
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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
// Scalar product node — Phase 6 (for chain rule results)
// ===========================================================================

// Product of two rank-0 expressions. Created by make_product after rational
// simplifications are exhausted.
class Product : public Expr
{
public:
    Product(Expr* lhs, Expr* rhs);
    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 0;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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
// Standard scalar function nodes (Phase 5)
// ===========================================================================

// Which unary scalar function is applied.
enum class FunctionKind
{
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    ASin,
    ACos,
    ATan,
    Sinh,
    Cosh,
    Tanh,
    Sqrt,
};

// Application of a unary scalar function to a rank-0 expression.
// Throws std::invalid_argument if arg->rank() != 0.
class FunctionApply : public Expr
{
public:
    FunctionApply(FunctionKind kind, Expr* arg);
    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 0;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto kind() const noexcept -> FunctionKind
    {
        return kind_;
    }
    [[nodiscard]] auto arg() const noexcept -> Expr*
    {
        return arg_;
    }

private:
    FunctionKind kind_;
    Expr* arg_;
};

// Power: base^exp with a Rational exponent. Rank 0.
// Throws std::invalid_argument if base->rank() != 0.
// Simplifications are applied by the factory make_pow (not the constructor).
class Pow : public Expr
{
public:
    Pow(Expr* base, Rational exp);
    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 0;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto base() const noexcept -> Expr*
    {
        return base_;
    }
    [[nodiscard]] auto exponent() const noexcept -> Rational const&
    {
        return exp_;
    }

private:
    Expr* base_;
    Rational exp_;
};

// Two-argument arctangent atan2(y, x). Rank 0.
// Throws std::invalid_argument if y->rank() != 0 or x->rank() != 0.
class ATan2 : public Expr
{
public:
    ATan2(Expr* y, Expr* x);
    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 0;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto y() const noexcept -> Expr*
    {
        return y_;
    }
    [[nodiscard]] auto x() const noexcept -> Expr*
    {
        return x_;
    }

private:
    Expr* y_;
    Expr* x_;
};

// ===========================================================================
// Geometric operation nodes (Phase 4)
// ===========================================================================

// Identity tensor: rank 2. make_contract(I, a) → a.
class IdentityTensor : public Expr
{
public:
    IdentityTensor() noexcept = default;
    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 2;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
};

// Levi-Civita permutation tensor: rank 3.
class LeviCivitaTensor : public Expr
{
public:
    LeviCivitaTensor() noexcept = default;
    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 3;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
};

// Trace of a rank-2 expression: rank 0.
// Produced by the make_double_contract(I, A) simplification.
class Trace : public Expr
{
public:
    explicit Trace(Expr* arg);
    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return 0;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto arg() const noexcept -> Expr*
    {
        return arg_;
    }

private:
    Expr* arg_;
};

// Single contraction (·): rank = lhs.rank + rhs.rank − 2.
// Constructed only when no simplification applies (use make_contract).
class Contract : public Expr
{
public:
    Contract(Expr* lhs, Expr* rhs) :
      lhs_(lhs), rhs_(rhs), rank_(lhs->rank() + rhs->rank() - 2)
    {
    }

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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
    int rank_;
};

// Double contraction — Frobenius (A:B = AᵢⱼBᵢⱼ): rank = lhs.rank + rhs.rank
// − 4. Constructed only when no simplification applies (use
// make_double_contract).
class DoubleContract : public Expr
{
public:
    DoubleContract(Expr* lhs, Expr* rhs) :
      lhs_(lhs), rhs_(rhs), rank_(lhs->rank() + rhs->rank() - 4)
    {
    }

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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
    int rank_;
};

// Reversed double contraction (A··B = AᵢⱼBⱼᵢ): rank = lhs.rank + rhs.rank − 4.
class DoubleContractReversed : public Expr
{
public:
    DoubleContractReversed(Expr* lhs, Expr* rhs) :
      lhs_(lhs), rhs_(rhs), rank_(lhs->rank() + rhs->rank() - 4)
    {
    }

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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
    int rank_;
};

// Cross product (a×b = −a·ε·b): rank = lhs.rank + rhs.rank − 1.
// make_cross_product() rejects chaining (pass a nested CrossProduct as an
// operand); use rl.make<CrossProduct> directly only in pattern-construction
// code where nesting is intentional.
class CrossProduct : public Expr
{
public:
    CrossProduct(Expr* lhs, Expr* rhs);
    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
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
    int rank_;
};

// ===========================================================================
// Phase 6 nodes — material derivative
// ===========================================================================

// Material (total) time derivative D f/Dt = ∂f/∂t + v·∇f.
// Stored as an unevaluated named node; expansion requires grad (Phase 10).
// velocity must have rank 1; rank() == field->rank().
class MaterialDeriv : public Expr
{
public:
    MaterialDeriv(Expr* velocity, Expr* field);
    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;
    [[nodiscard]] auto velocity() const noexcept -> Expr*
    {
        return velocity_;
    }
    [[nodiscard]] auto field() const noexcept -> Expr*
    {
        return field_;
    }

private:
    Expr* velocity_;
    Expr* field_;
    int rank_;
};

// ===========================================================================
// Phase 6 — polynomial over a ring embedded in the expression tree
// ===========================================================================

// A polynomial with exact Rational coefficients evaluated at a ring element
// `var`.  Supported rings:
//   rank 0 — scalar field:  x^n  via make_pow
//   rank 2 — tensor ring:   A^n  via repeated make_contract; A^0 = I
//
// expand(rl) materialises the sum of ring-power terms as an Expr*.
// rank() == var->rank().  Throws if var->rank() is not 0 or 2.
class PolynomialExpr : public Expr
{
public:
    PolynomialExpr(Polynomial poly, Expr* var);

    [[nodiscard]] auto rank() const noexcept -> int override
    {
        return rank_;
    }
    [[nodiscard]] auto latex(IndexNameMap const& map) const
        -> std::string override;
    using Expr::latex;
    [[nodiscard]] auto python() const -> std::string override;

    [[nodiscard]] auto poly() const noexcept -> Polynomial const&
    {
        return poly_;
    }
    [[nodiscard]] auto var() const noexcept -> Expr*
    {
        return var_;
    }

    // Expand to a fully explicit Expr* sum in rl.
    [[nodiscard]] auto expand(ResourceList& rl) const -> Expr*;

private:
    Polynomial poly_;
    Expr* var_;
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
auto make_indexed_sum(
    ResourceList& rl,
    std::string lhs_sym,
    std::string lhs_sep,
    std::string rhs_sym,
    std::string rhs_sep,
    std::string index_letter,
    int rank) -> Expr*;

// Pair slot_a from a with slot_b from b. Exactly one slot must be Upper and
// the other Lower; throws std::invalid_argument on level mismatch or
// out-of-bounds slot indices.
auto convolve(
    ResourceList& rl, Expr* a, std::size_t slot_a, Expr* b, std::size_t slot_b)
    -> Expr*;

// ===========================================================================
// Factory functions — geometric operations (Phase 4)
// ===========================================================================

auto make_identity(ResourceList& rl) -> Expr*;
auto make_levi_civita(ResourceList& rl) -> Expr*;

// Trace of a rank-2 expression. Throws if arg->rank() != 2.
auto make_trace(ResourceList& rl, Expr* arg) -> Expr*;

// Single contraction (·): rank = lhs + rhs − 2. Requires both rank ≥ 1.
// Always-on simplifications: I·a → a, a·I → a, 0·a → 0, a·0 → 0.
auto make_contract(ResourceList& rl, Expr* lhs, Expr* rhs) -> Expr*;

// Double contraction (A:B = AᵢⱼBᵢⱼ): rank = lhs + rhs − 4. Requires both
// rank ≥ 2. Always-on simplification: I:A → tr(A) when A has rank 2.
auto make_double_contract(ResourceList& rl, Expr* lhs, Expr* rhs) -> Expr*;

// Reversed double contraction (A··B = AᵢⱼBⱼᵢ): rank = lhs + rhs − 4.
// Requires both rank ≥ 2.
auto make_double_contract_reversed(ResourceList& rl, Expr* lhs, Expr* rhs)
    -> Expr*;

// Cross product (a×b = −a·ε·b): rank = lhs + rhs − 1. Requires both rank ≥ 1.
// Throws if either operand is itself a CrossProduct (chaining guard).
auto make_cross_product(ResourceList& rl, Expr* lhs, Expr* rhs) -> Expr*;

// ===========================================================================
// Factory functions — standard scalar functions (Phase 5)
// ===========================================================================

// Generic factory — validates arg->rank() == 0.
auto make_function(ResourceList& rl, FunctionKind kind, Expr* arg) -> Expr*;

// Convenience wrappers.
auto make_exp(ResourceList& rl, Expr* arg) -> Expr*;
auto make_log(ResourceList& rl, Expr* arg) -> Expr*;
auto make_sin(ResourceList& rl, Expr* arg) -> Expr*;
auto make_cos(ResourceList& rl, Expr* arg) -> Expr*;
auto make_tan(ResourceList& rl, Expr* arg) -> Expr*;
auto make_asin(ResourceList& rl, Expr* arg) -> Expr*;
auto make_acos(ResourceList& rl, Expr* arg) -> Expr*;
auto make_atan(ResourceList& rl, Expr* arg) -> Expr*;
auto make_sinh(ResourceList& rl, Expr* arg) -> Expr*;
auto make_cosh(ResourceList& rl, Expr* arg) -> Expr*;
auto make_tanh(ResourceList& rl, Expr* arg) -> Expr*;
auto make_sqrt(ResourceList& rl, Expr* arg) -> Expr*;

// Power: base^exp. Simplifications: pow(x,0)→1, pow(x,1)→x.
// Throws if base->rank() != 0.
auto make_pow(ResourceList& rl, Expr* base, Rational exp) -> Expr*;

// Two-argument arctangent. Throws if y->rank() != 0 or x->rank() != 0.
auto make_atan2(ResourceList& rl, Expr* y, Expr* x) -> Expr*;

// Derivative of kind(arg) w.r.t. arg (outer derivative for chain rule).
// Returns an Expr* in the given ResourceList.
auto derivative_of(ResourceList& rl, FunctionKind kind, Expr* arg) -> Expr*;

// Derivative of pow(base, exp) w.r.t. base.
auto derivative_of_pow(ResourceList& rl, Expr* base, Rational exp) -> Expr*;

// ===========================================================================
// Factory functions — Phase 6 (parameters, dependency, differentiation)
// ===========================================================================

// Create a named scalar parameter.
auto make_parameter(ResourceList& rl, std::string symbol) -> Parameter*;

// Product of two rank-0 expressions. Simplifications: rational constant on
// either side collapses to make_scale.
auto make_product(ResourceList& rl, Expr* lhs, Expr* rhs) -> Expr*;

// Return the built-in time parameter (symbol "t").  dt() and ddt() use it.
// Users who write expressions in "t" via make_parameter(rl, "t") are
// equivalent: dependency tracking is name-based.
auto time_parameter() -> Parameter const*;

// True if e contains a Parameter with the same symbol as p anywhere in its
// sub-tree. NamedTensor nodes are assumed independent unless they are
// Parameters themselves.
auto depends_on(Parameter const* p, Expr const* e) -> bool;

// Symbolic partial derivative ∂e/∂p.  Returns a zero RationalConst(0) if e
// does not depend on p (rank information is not preserved for zero; this is
// a known Phase-6 limitation).  Applies product rule, sum rule, and chain
// rule recursively for all Phase-2 through Phase-6 node types.
auto deriv(ResourceList& rl, Parameter const* p, Expr* e) -> Expr*;

// Partial time derivative ∂e/∂t using the built-in time parameter.
auto dt(ResourceList& rl, Expr* e) -> Expr*;

// Second partial time derivative ∂²e/∂t².
auto ddt(ResourceList& rl, Expr* e) -> Expr*;

// Material (total) time derivative node D f/Dt = ∂f/∂t + v·∇f.
// velocity must have rank 1; rank() == field->rank().
auto make_material_deriv(ResourceList& rl, Expr* velocity, Expr* field) -> Expr*;

// Polynomial node. var->rank() must be 0 or 2.
auto make_polynomial_expr(ResourceList& rl, Polynomial poly, Expr* var)
    -> PolynomialExpr*;

// ===========================================================================
// substitute_index — replace one abstract index ID with another throughout
// a tree.  Affects AbstractComp, SymBasisVec, AbstractIndexedSum,
// KroneckerDelta, and LeviCivitaSymbol nodes, as well as all compound nodes.
// Returns the same pointer if no substitution was needed.
// ===========================================================================

auto substitute_index(ResourceList& rl, Expr* e, int old_id, int new_id)
    -> Expr*;

// ===========================================================================
// EnrichedExpr — pre-rendering stage for abstract-index nodes
// ===========================================================================

struct EnrichedExpr
{
    Expr const* expr;
    IndexNameMap index_names;
    [[nodiscard]] auto latex() const -> std::string; // calls
                                                     // expr->latex(index_names)
};

// DFS-walk expr to collect all abstract index IDs (from AbstractComp,
// SymBasisVec, AbstractIndexedSum nodes) in first-appearance order, then
// assign letters "i","j","k","l","m","n" and "a","b","c","d","e".
auto enrich(Expr const* e) -> EnrichedExpr;

} // namespace tender
