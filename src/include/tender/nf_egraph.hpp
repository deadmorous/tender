#pragma once

// An equality graph over the normal form `Nf` (vibe 000058 / C14d part 2).
//
// This is the Nf-native successor to the `Expr`-structural `EGraph` of vibe
// 000034.  Where that e-graph's e-nodes mirror `Expr` operator nodes (binary
// `Sum` / `TensorProduct` / `Cross` …), this one's e-nodes mirror the `Nf`
// structure: the recursive `Factor` tree (`Atom` / `Contraction` / `Cross` /
// `Paren` / `Unary` / `Div`), the multiplicative `Term`
// (`coeff · scalars · tensors · bound`), and the additive `Sum` of terms.  So
// the e-class matcher can be the flat-form `nf_match` matcher rather than a
// second, divergent structural matcher.
//
// It grows one commit at a time, beside the existing `EGraph`, mirroring the
// parallel-IR strategy of the rest of 000058.  This first commit (data core)
// introduces the e-node representation, union-find, hash-consing, and `add`
// (lowering a canonical `Nf` / `Expr` into the graph); congruence `rebuild`,
// `extract`, `ematch`, and `saturate` arrive in later commits.

#include <tender/expr.hpp>
#include <tender/nf.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace tender::nf
{

// An e-class identifier, stable for the graph's lifetime; use `find()` for the
// current representative after merges.
using EClassId = int;

class NfEGraph final
{
public:
    explicit NfEGraph(Context& ctx);
    ~NfEGraph();
    NfEGraph(NfEGraph&&) noexcept;
    auto operator=(NfEGraph&&) noexcept -> NfEGraph&;

    // Insert a canonical `Nf` and return the id of the e-class representing it
    // (its additive `Sum` node).  Equal `Nf`s land in the same class.
    [[nodiscard]] auto add(Nf const* nf) -> EClassId;

    // Convenience: canonicalize `e` to `Nf`
    // (`canonicalize_nf(canonicalize(e))`) and insert it.
    [[nodiscard]] auto add(Expr const* e) -> EClassId;

    // Union the e-classes of a and b; returns the surviving representative.
    // Call rebuild() afterwards to restore congruence before querying.
    auto merge(EClassId a, EClassId b) -> EClassId;

    // Restore the congruence invariant after one or more merges.
    void rebuild();

    // Canonical representative of the class containing id.
    [[nodiscard]] auto find(EClassId id) -> EClassId;

    // Cheapest (smallest node-count) representative `Nf` of a class.  Requires
    // a `Sum`-sort class (an `add`-returned id, or one merged with such).
    [[nodiscard]] auto extract(EClassId id) -> Nf const*;

    // Number of distinct e-classes / e-nodes (diagnostics / tests).
    [[nodiscard]] auto class_count() -> std::size_t;
    [[nodiscard]] auto node_count() const -> std::size_t;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tender::nf
