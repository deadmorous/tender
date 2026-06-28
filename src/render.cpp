#include <tender/render.hpp>

#include <mpk/mix/enum_flags.hpp>
#include <mpk/mix/util/overloads.hpp>
#include <tender/basis.hpp>      // Basis::value_name
#include <tender/context.hpp>    // Context::basis
#include <tender/derivation.hpp> // infer_rank
#include <tender/nf.hpp>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tender
{

// ---- IndexNameMap -------------------------------------------------------

void IndexNameMap::assign(CountableIndex ci, IndexName name)
{
    // Remove the old forward entry for this id, if any.
    auto fwd = id_to_name_.find(ci.id);
    if (fwd != id_to_name_.end())
        name_to_id_.erase(std::string{fwd->second.v.view()});

    // Remove the old reverse entry for this name, if any.
    auto key = std::string{name.v.view()};
    auto rev = name_to_id_.find(key);
    if (rev != name_to_id_.end())
        id_to_name_.erase(rev->second);

    id_to_name_[ci.id] = name;
    name_to_id_[key] = ci.id;
}

auto IndexNameMap::name_for(CountableIndex ci, IndexSpace const* space)
    -> IndexName
{
    auto it = id_to_name_.find(ci.id);
    if (it != id_to_name_.end())
        return it->second;

    auto& pos = next_schema_pos_[space];
    while (true)
    {
        auto name = space->dummy_name(pos++); // unbounded: subscripts past the
                                              // base schema (vibe 000064 #5)
        auto key = std::string{name.v.view()};
        if (name_to_id_.find(key) == name_to_id_.end())
        {
            id_to_name_[ci.id] = name;
            name_to_id_[key] = ci.id;
            return name;
        }
    }
}

auto IndexNameMap::lookup(CountableIndex ci) const -> std::optional<IndexName>
{
    auto it = id_to_name_.find(ci.id);
    if (it == id_to_name_.end())
        return std::nullopt;
    return it->second;
}

auto IndexNameMap::index_for(IndexName name) const
    -> std::optional<CountableIndex>
{
    auto it = name_to_id_.find(std::string{name.v.view()});
    if (it == name_to_id_.end())
        return std::nullopt;
    return CountableIndex{it->second};
}

// ---- Renderer -----------------------------------------------------------

namespace
{

// Precedence levels (higher = binds tighter).
//
// The contractions (× · : ··) sit *below* the tensor product: they are
// non-associative and have no agreed precedence with ⊗, so a contraction
// nested under any product is parenthesized, while ⊗ (juxtaposition, which
// reads tightest) needs none.
constexpr int ADD_PREC = 1;      // Sum, Difference
constexpr int CONTRACT_PREC = 2; // Dot, DDot, DDotAlt, Cross (non-associative)
constexpr int TENSOR_PREC = 3;   // TensorProduct, ScalarDiv
constexpr int UNARY_PREC = 4;    // Negate
constexpr int ATOM_PREC = 5; // TensorObject, ScalarLiteral, ExplicitSum, NoSum

// True iff every leaf in the expression tree is a ScalarLiteral (no tensors).
static bool is_scalar_expr(Expr const& e)
{
    return visit(
        mpk::mix::Overloads{
            [](ScalarLiteral const&) { return true; },
            [](TensorObject const&) { return false; },
            [](Negate const& n) { return is_scalar_expr(*n.operand); },
            [](Trace const&) { return false; },
            [](VectorInvariant const&) { return false; },
            [](Transpose const&) { return false; },
            [](Sum const& s)
            { return is_scalar_expr(*s.left) && is_scalar_expr(*s.right); },
            [](Difference const& d)
            { return is_scalar_expr(*d.left) && is_scalar_expr(*d.right); },
            [](TensorProduct const& p)
            { return is_scalar_expr(*p.left) && is_scalar_expr(*p.right); },
            [](ScalarDiv const& d)
            { return is_scalar_expr(*d.left) && is_scalar_expr(*d.right); },
            [](Dot const&) { return false; },
            [](DDot const&) { return false; },
            [](DDotAlt const&) { return false; },
            [](Cross const&) { return false; },
            [](ExplicitSum const& s) { return is_scalar_expr(*s.body); },
            [](NoSum const& s) { return is_scalar_expr(*s.body); },
        },
        e);
}

// ---- shared leaf rendering (Expr atoms == Nf atoms) --------------------

auto index_str(
    IndexNameMap& map,
    std::optional<IndexAssoc> const& assoc,
    IndexSlot const& slot,
    Context const* ctx) -> std::string
{
    if (!assoc)
        return "\\bullet";
    std::string base = std::visit(
        mpk::mix::Overloads{
            [&](CountableIndex const& ci) -> std::string
            { return std::string{map.name_for(ci, slot.space).v.view()}; },
            [&](ConcreteIndex const& ci) -> std::string
            {
                // A concrete index in a basis prints with that basis's
                // coordinate letter when it has one (vibe 000067): value 1 in a
                // cylindrical frame reads "r", not "1".  Falls back to numeric.
                if (ctx && slot.basis_id != 0)
                    if (auto const* b = ctx->basis(slot.basis_id))
                        if (auto nm = b->value_name(ci.value))
                            return std::string{nm->v.view()};
                return std::to_string(ci.value);
            },
            [](LabelIndex const& li) -> std::string
            { return std::string{li.name.v.view()}; }},
        *assoc);
    // A labelled basis marks every one of its indices (vibe 000067), so two
    // frames in one term are distinguishable: the primed index in e_{i'} ⊗ e_i.
    if (ctx && slot.basis_id != 0)
        if (auto const* b = ctx->basis(slot.basis_id))
            if (auto const& lab = b->label())
                base += *lab;
    return base;
}

// Bold for rank >= 1 or rank nullopt (treated as tensor); plain for rank == 0.
auto name_str(TensorName const& name, std::optional<int> rank) -> std::string
{
    std::string n{name.v.view()};
    bool bold = !rank.has_value() || *rank >= 1;
    if (!bold)
        return n;
    bool is_cmd = (n.size() > 1 && n[0] == '\\');
    return is_cmd ? "\\boldsymbol{" + n + "}" : "\\mathbf{" + n + "}";
}

// If `t` is a basis vector with a concrete index whose basis supplies a
// standalone direction symbol (vibe 000067, e.g. WCS i, j, k), render that
// symbol on its own — replacing the e_x form, index and all.  nullopt
// otherwise, so the caller falls back to name + slots.
auto basis_vector_override(TensorObject const& t, Context const* ctx)
    -> std::optional<std::string>
{
    if (!ctx || t.slots.size() != 1 || !t.slots[0].index)
        return std::nullopt;
    auto const& sb = t.slots[0];
    if (sb.slot.basis_id == 0)
        return std::nullopt;
    auto const* ci = std::get_if<ConcreteIndex>(&*sb.index);
    if (!ci)
        return std::nullopt;
    auto const* b = ctx->basis(sb.slot.basis_id);
    if (!b || t.name.v.view() != b->vector_symbol().v.view())
        return std::nullopt;
    if (auto sym = b->vector_symbol_for(ci->value))
        return name_str(*sym, t.rank);
    return std::nullopt;
}

// Render index slots.
//
// Pure-level (all upper or all lower): flat grouping, e.g. ^{ijk}.
//
// Mixed-level, OmitVoidIndexPlaceholders set: separate ^{...} and _{...} bands
// with no \cdot markers — appropriate for objects whose symmetry makes slot
// order unimportant (e.g. Kronecker delta).
//
// Mixed-level, default: positional interleaving — each slot contributes its
// index to its own band and \cdot to the other, making position explicit.  E.g.
// [upper-i, lower-j] → ^{i\cdot}_{\cdot j}.
auto slots_str(
    IndexNameMap& map,
    std::vector<SlotBinding> const& slots,
    mpk::mix::EnumFlags<RenderHint> hints,
    Context const* ctx) -> std::string
{
    if (slots.empty())
        return {};

    bool has_upper = false, has_lower = false;
    for (auto const& sb: slots)
    {
        if (sb.slot.level == Level::Upper)
            has_upper = true;
        else
            has_lower = true;
    }

    if (!has_upper || !has_lower
        || hints.contains(RenderHint::OmitVoidIndexPlaceholders))
    {
        // Flat grouping: collect upper indices then lower indices.
        std::string upper, lower;
        for (auto const& sb: slots)
        {
            auto s = index_str(map, sb.index, sb.slot, ctx);
            if (sb.slot.level == Level::Upper)
                upper += s;
            else
                lower += s;
        }
        std::string result;
        if (has_upper)
            result += "^{" + upper + "}";
        if (has_lower)
            result += "_{" + lower + "}";
        return result;
    }

    // Mixed positional: build both bands with \cdot placeholders.
    // "\\cdot " (with trailing space) safely terminates the LaTeX command
    // before any following letter; trimmed from band ends for clean output.
    std::string upper, lower;
    for (auto const& sb: slots)
    {
        auto s = index_str(map, sb.index, sb.slot, ctx);
        if (sb.slot.level == Level::Upper)
        {
            upper += s;
            lower += "\\cdot ";
        }
        else
        {
            upper += "\\cdot ";
            lower += s;
        }
    }
    if (!upper.empty() && upper.back() == ' ')
        upper.pop_back();
    if (!lower.empty() && lower.back() == ' ')
        lower.pop_back();
    return "^{" + upper + "}_{" + lower + "}";
}

auto rational_str(Rational const& r) -> std::string
{
    if (r.is_integer())
        return std::to_string(r.num());
    return "\\frac{" + std::to_string(r.num()) + "}{" + std::to_string(r.den())
           + "}";
}

struct Renderer
{
    IndexNameMap& map;
    Context const* ctx = nullptr; // resolves basis_id for coordinate naming

    // ---- precedence ----------------------------------------------------

    static auto prec(Expr const& e) -> int
    {
        return visit(
            mpk::mix::Overloads{
                [](TensorObject const&) { return ATOM_PREC; },
                [](ScalarLiteral const&) { return ATOM_PREC; },
                [](ExplicitSum const&) { return ATOM_PREC; },
                [](NoSum const&) { return ATOM_PREC; },
                [](Negate const&) { return UNARY_PREC; },
                // Function applications / postfix — self-delimiting, so atomic.
                [](Trace const&) { return ATOM_PREC; },
                [](VectorInvariant const&) { return ATOM_PREC; },
                [](Transpose const&) { return ATOM_PREC; },
                [](Sum const&) { return ADD_PREC; },
                [](Difference const&) { return ADD_PREC; },
                [](TensorProduct const&) { return TENSOR_PREC; },
                [](ScalarDiv const&) { return TENSOR_PREC; },
                [](Dot const&) { return CONTRACT_PREC; },
                [](DDot const&) { return CONTRACT_PREC; },
                [](DDotAlt const&) { return CONTRACT_PREC; },
                [](Cross const&) { return CONTRACT_PREC; },
            },
            e);
    }

    // Render sub-expression; wrap in parens if its precedence is below
    // min_prec.
    auto sub(Expr const& e, int min_prec) -> std::string
    {
        auto s = render(e);
        return prec(e) < min_prec ? "(" + s + ")" : s;
    }

    // A contraction that evaluates to a scalar (rank 0) — e.g. a dot product
    // a·c — reads as an atom inside a product, so (a·c)(b·d) renders without
    // the redundant parens as "a·c \, b·d".
    static auto is_scalar_contraction(Expr const& e) -> bool
    {
        bool const contraction = std::holds_alternative<Dot>(e.node)
                                 || std::holds_alternative<DDot>(e.node)
                                 || std::holds_alternative<DDotAlt>(e.node)
                                 || std::holds_alternative<Cross>(e.node);
        return contraction && infer_rank(&e) == std::optional<int>{0};
    }

    // Child of a product (⊗ or a contraction operand): like sub(e, TENSOR_PREC)
    // but a scalar-valued contraction is left unwrapped.
    auto sub_product_child(Expr const& e) -> std::string
    {
        auto s = render(e);
        if (is_scalar_contraction(e))
            return s;
        return prec(e) < TENSOR_PREC ? "(" + s + ")" : s;
    }

    // Child of a Sum: no wrapping needed — Sum rendering converts a Negate
    // right child to subtraction; a Negate left child renders cleanly as "-…".
    auto sub_sum_child(Expr const& e) -> std::string
    {
        auto s = render(e);
        return prec(e) < ADD_PREC ? "(" + s + ")" : s;
    }

    // Right child of Difference: also wrap same-precedence nodes
    // (prevents a - b + c being read as a - (b + c)), and Negate nodes.
    auto sub_diff_right(Expr const& e) -> std::string
    {
        auto s = render(e);
        bool wrap =
            prec(e) <= ADD_PREC || std::holds_alternative<Negate>(e.node);
        return wrap ? "(" + s + ")" : s;
    }

    // ---- leaves --------------------------------------------------------

    // Render a sum-annotation node.  Body is rendered first so that the
    // index name is allocated before the annotation prefix is composed.
    auto sum_str(CountableIndex idx, Expr const* body, std::string const& sym)
        -> std::string
    {
        auto body_s = render(*body);
        // Wrap a Sum/Difference body (so "Σ_i a + b" is not read as
        // "(Σ_i a) + b"), and also a negated sub-sum: "Σ_j -Σ_i …" reads like a
        // difference "Σ_j - (Σ_i …)" with an empty outer body, so render it as
        // "Σ_j (-Σ_i …)" instead (vibe 000064 #8b).
        bool wrap = prec(*body) == ADD_PREC;
        if (auto const* n = std::get_if<Negate>(&body->node))
            wrap = wrap || std::holds_alternative<ExplicitSum>(n->operand->node)
                   || std::holds_alternative<NoSum>(n->operand->node);
        if (wrap)
            body_s = "(" + body_s + ")";

        auto name = map.lookup(idx);
        auto idx_s = name ? std::string{name->v.view()} : "?";
        return sym + "_{" + idx_s + "} " + body_s;
    }

    // ---- main dispatch -------------------------------------------------

    auto render(Expr const& e) -> std::string
    {
        return visit(
            mpk::mix::Overloads{
                [&](TensorObject const& t) -> std::string
                {
                    if (auto ov = basis_vector_override(t, ctx))
                        return *ov;
                    mpk::mix::EnumFlags<RenderHint> hints;
                    if (t.traits)
                        hints = t.traits->render_hints;
                    return name_str(t.name, t.rank)
                           + slots_str(map, t.slots, hints, ctx);
                },
                [&](ScalarLiteral const& s) -> std::string
                { return rational_str(s.value); },
                [&](Negate const& n) -> std::string
                {
                    // Wrap only a sum/difference operand — those have lower
                    // precedence, so "-a + b" would be misread without parens.
                    // Products, contractions, and atoms don't need them:
                    // "-a \, b" and "-a \times b" are unambiguous.
                    return "-" + sub(*n.operand, CONTRACT_PREC);
                },
                [&](Trace const& u) -> std::string
                { return "\\operatorname{tr}(" + render(*u.operand) + ")"; },
                [&](VectorInvariant const& u) -> std::string
                {
                    // vec(A) renders as A_× (subscript cross).  A slot-less
                    // tensor object stays bare; anything else (a dyad, a sum,
                    // …) is parenthesized: (a b)_×.
                    auto const* t = std::get_if<TensorObject>(&u.operand->node);
                    bool const bare = t && t->slots.empty();
                    auto const inner = render(*u.operand);
                    return (bare ? inner : "(" + inner + ")") + "_\\times";
                },
                [&](Transpose const& u) -> std::string
                { return sub(*u.operand, ATOM_PREC) + "^{\\mathsf{T}}"; },
                [&](Sum const& s) -> std::string
                {
                    // A sum whose right addend is negated renders as
                    // subtraction: "A + (-B)" → "A - B".
                    if (auto* neg = std::get_if<Negate>(&s.right->node))
                        return sub_sum_child(*s.left) + " - "
                               + sub_diff_right(*neg->operand);
                    return sub_sum_child(*s.left) + " + "
                           + sub_sum_child(*s.right);
                },
                [&](Difference const& d) -> std::string {
                    return sub(*d.left, ADD_PREC) + " - "
                           + sub_diff_right(*d.right);
                },
                [&](TensorProduct const& p) -> std::string
                {
                    bool both_scalar =
                        is_scalar_expr(*p.left) && is_scalar_expr(*p.right);
                    std::string sep = both_scalar ? " \\cdot " : " \\, ";
                    return sub_product_child(*p.left) + sep
                           + sub_product_child(*p.right);
                },
                [&](ScalarDiv const& d) -> std::string {
                    return "\\frac{" + render(*d.left) + "}{" + render(*d.right)
                           + "}";
                },
                // The contractions are non-associative and have no settled
                // precedence with ⊗, so both operands wrap any nested
                // contraction (and any sum): a × b × c → (a × b) × c, and
                // a × (b × c) keeps its parens.
                [&](Dot const& d) -> std::string {
                    return sub(*d.left, TENSOR_PREC) + " \\cdot "
                           + sub(*d.right, TENSOR_PREC);
                },
                [&](DDot const& d) -> std::string {
                    return sub(*d.left, TENSOR_PREC) + " : "
                           + sub(*d.right, TENSOR_PREC);
                },
                [&](DDotAlt const& d) -> std::string
                {
                    return sub(*d.left, TENSOR_PREC) + " \\cdot\\!\\cdot "
                           + sub(*d.right, TENSOR_PREC);
                },
                [&](Cross const& c) -> std::string
                {
                    return sub(*c.left, TENSOR_PREC) + " \\times "
                           + sub(*c.right, TENSOR_PREC);
                },
                [&](ExplicitSum const& s) -> std::string
                { return sum_str(s.index, s.body, "\\sum"); },
                [&](NoSum const& s) -> std::string
                { return sum_str(s.index, s.body, "\\cancel{\\sum}"); },
            },
            e);
    }
};

// ---- Nf renderer (C11) -------------------------------------------------

// Renders the all-`*` normal form (nf.hpp) in the same LaTeX conventions as the
// `Expr` Renderer above, reusing the shared leaf helpers.  An `Nf` is an
// additive set of signed `coeff · scalars · tensors` terms; a `Default` bound
// index renders *implicitly* (it is just a repeated slot index — the Einstein
// form), while `Sum` / `NoSum` overrides get a `\sum` / `\cancel{\sum}` prefix.
struct NfRenderer
{
    IndexNameMap& map;
    Context const* ctx = nullptr; // resolves basis_id for coordinate naming

    static auto prec(nf::Factor const& f) -> int
    {
        return nf::visit(
            mpk::mix::Overloads{
                [](nf::Atom const&) { return ATOM_PREC; },
                [](nf::Contraction const&) { return CONTRACT_PREC; },
                [](nf::Cross const&) { return CONTRACT_PREC; },
                [](nf::Paren const&)
                { return ATOM_PREC; }, // self-parenthesized
                [](nf::Unary const&) { return ATOM_PREC; }, // self-delimiting
                [](nf::Div const&) { return ATOM_PREC; },   // \frac is atomic
            },
            f);
    }

    // Render a factor, wrapping in parens if its precedence is below min_prec.
    auto sub(nf::Factor const& f, int min_prec) -> std::string
    {
        auto s = render_factor(f);
        return prec(f) < min_prec ? "(" + s + ")" : s;
    }

    static auto cop_str(nf::COp op) -> char const*
    {
        switch (op)
        {
            case nf::COp::Dot: return " \\cdot ";
            case nf::COp::DDot: return " : ";
            case nf::COp::DDotAlt: return " \\cdot\\!\\cdot ";
        }
        return " \\, "; // unreachable
    }

    auto render_factor(nf::Factor const& f) -> std::string
    {
        return nf::visit(
            mpk::mix::Overloads{
                [&](nf::Atom const& a) -> std::string
                {
                    if (auto ov = basis_vector_override(a.obj, ctx))
                        return *ov;
                    mpk::mix::EnumFlags<RenderHint> hints;
                    if (a.obj.traits)
                        hints = a.obj.traits->render_hints;
                    return name_str(a.obj.name, a.obj.rank)
                           + slots_str(map, a.obj.slots, hints, ctx);
                },
                // Flat contraction chain: f0 op0 f1 op1 …  Each operand wraps a
                // nested contraction / cross (prec < TENSOR_PREC).
                [&](nf::Contraction const& c) -> std::string
                {
                    std::string s = sub(*c.factors[0], TENSOR_PREC);
                    for (std::size_t i = 0; i + 1 < c.factors.size(); ++i)
                        s += cop_str(c.ops[i])
                             + sub(*c.factors[i + 1], TENSOR_PREC);
                    return s;
                },
                [&](nf::Cross const& c) -> std::string
                {
                    std::string s = sub(*c.factors[0], TENSOR_PREC);
                    for (std::size_t i = 1; i < c.factors.size(); ++i)
                        s += " \\times " + sub(*c.factors[i], TENSOR_PREC);
                    return s;
                },
                [&](nf::Paren const& p) -> std::string
                { return "(" + render_nf(*p.body) + ")"; },
                [&](nf::Div const& d) -> std::string {
                    return "\\frac{" + render_nf(*d.num) + "}{"
                           + render_nf(*d.den) + "}";
                },
                [&](nf::Unary const& u) -> std::string
                {
                    switch (u.op)
                    {
                        case nf::UnaryOp::Trace:
                            return "\\operatorname{tr}("
                                   + render_factor(*u.operand) + ")";
                        case nf::UnaryOp::VectorInvariant:
                        {
                            auto const* a =
                                std::get_if<nf::Atom>(&u.operand->node);
                            bool const bare = a && a->obj.slots.empty();
                            auto const inner = render_factor(*u.operand);
                            return (bare ? inner : "(" + inner + ")")
                                   + "_\\times";
                        }
                        case nf::UnaryOp::Transpose:
                            return sub(*u.operand, ATOM_PREC)
                                   + "^{\\mathsf{T}}";
                    }
                    return "?"; // unreachable
                },
            },
            f);
    }

    // Render one term's signed magnitude.  Returns (negative?, body) so the Nf
    // layer can compose `+` / `-` joins.
    auto render_term(nf::Term const& t) -> std::pair<bool, std::string>
    {
        bool const neg = t.coeff < Rational{0};
        Rational const mag = neg ? -t.coeff : t.coeff;

        bool const has_factors = !t.scalars.empty() || !t.tensors.empty();
        bool const has_coeff = !(mag == Rational{1} && has_factors);
        std::size_t const nparts =
            (has_coeff ? 1 : 0) + t.scalars.size() + t.tensors.size();
        // A tensor-valued contraction / cross needs parens only when it is
        // juxtaposed with a sibling part (coeff or another factor); alone in a
        // term it reads cleanly (`a × b`, not `(a × b)`).  Scalars are
        // scalar-valued, so they always read as atoms (no wrap).
        bool const multi = nparts > 1;

        std::vector<std::string> parts;
        if (has_coeff)
            parts.push_back(rational_str(mag));
        for (auto const* f: t.scalars)
            parts.push_back(render_factor(*f));
        for (auto const* f: t.tensors)
            parts.push_back(multi ? sub(*f, TENSOR_PREC) : render_factor(*f));

        std::string body;
        for (std::size_t i = 0; i < parts.size(); ++i)
            body += (i ? " \\, " : "") + parts[i];

        // Explicit summation overrides prefix the term; a Default bound index
        // is implicit (carried by its repeated slot index), so it adds nothing.
        std::string prefix;
        for (auto const& b: t.bound)
        {
            if (b.mode == nf::SumMode::Default)
                continue;
            auto name = map.lookup(b.index);
            auto idx_s = name ? std::string{name->v.view()} : "?";
            char const* sym =
                b.mode == nf::SumMode::Sum ? "\\sum" : "\\cancel{\\sum}";
            prefix += std::string{sym} + "_{" + idx_s + "} ";
        }
        return {neg, prefix + body};
    }

    auto render_nf(nf::Nf const& value) -> std::string
    {
        if (value.terms.empty())
            return "0";
        std::string out;
        for (std::size_t i = 0; i < value.terms.size(); ++i)
        {
            auto [neg, body] = render_term(value.terms[i]);
            if (i == 0)
                out = (neg ? "-" : "") + body;
            else
                out += (neg ? " - " : " + ") + body;
        }
        return out;
    }
};

} // anonymous namespace

auto render_latex(Expr const& e, IndexNameMap& map, Context const* ctx)
    -> std::string
{
    return Renderer{map, ctx}.render(e);
}

auto render_nf_latex(nf::Nf const& value, IndexNameMap& map, Context const* ctx)
    -> std::string
{
    return NfRenderer{map, ctx}.render_nf(value);
}

} // namespace tender
