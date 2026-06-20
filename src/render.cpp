#include <tender/render.hpp>

#include <mpk/mix/enum_flags.hpp>
#include <mpk/mix/util/overloads.hpp>

#include <stdexcept>
#include <string>

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
        auto name = space->dummy_name(pos++); // throws std::out_of_range if
                                              // exhausted
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
constexpr int ADD_PREC = 1;   // Sum, Difference
constexpr int MUL_PREC = 2;   // TensorProduct, Dot, DDot, DDotAlt, Cross,
                              // ScalarDiv
constexpr int UNARY_PREC = 3; // Negate
constexpr int ATOM_PREC = 4;  // TensorObject, ScalarLiteral, ExplicitSum, NoSum

// True iff every leaf in the expression tree is a ScalarLiteral (no tensors).
static bool is_scalar_expr(Expr const& e)
{
    return visit(
        mpk::mix::Overloads{
            [](ScalarLiteral const&) { return true; },
            [](TensorObject const&) { return false; },
            [](Negate const& n) { return is_scalar_expr(*n.operand); },
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

struct Renderer
{
    IndexNameMap& map;

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
                [](Sum const&) { return ADD_PREC; },
                [](Difference const&) { return ADD_PREC; },
                [](TensorProduct const&) { return MUL_PREC; },
                [](ScalarDiv const&) { return MUL_PREC; },
                [](Dot const&) { return MUL_PREC; },
                [](DDot const&) { return MUL_PREC; },
                [](DDotAlt const&) { return MUL_PREC; },
                [](Cross const&) { return MUL_PREC; },
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

    // Right child of a non-associative contraction (× · : ··): wrap a
    // same-or-lower-precedence product child, since a × b × c reads as
    // (a × b) × c — so a × (b × c) needs explicit parens.  (TensorProduct is
    // associative and is rendered without these.)
    auto sub_contraction_right(Expr const& e) -> std::string
    {
        auto s = render(e);
        return prec(e) <= MUL_PREC ? "(" + s + ")" : s;
    }

    // ---- leaves --------------------------------------------------------

    auto index_str(
        std::optional<IndexAssoc> const& assoc,
        IndexSpace const* space) -> std::string
    {
        if (!assoc)
            return "\\bullet";
        return std::visit(
            mpk::mix::Overloads{
                [&](CountableIndex const& ci) -> std::string
                { return std::string{map.name_for(ci, space).v.view()}; },
                [](ConcreteIndex const& ci) -> std::string
                { return std::to_string(ci.value); },
                [](LabelIndex const& li) -> std::string
                { return std::string{li.name.v.view()}; }},
            *assoc);
    }

    // Bold for rank >= 1 or rank nullopt (treated as tensor); plain for rank ==
    // 0.
    static auto name_str(TensorName const& name, std::optional<int> rank)
        -> std::string
    {
        std::string n{name.v.view()};
        bool bold = !rank.has_value() || *rank >= 1;
        if (!bold)
            return n;
        bool is_cmd = (n.size() > 1 && n[0] == '\\');
        return is_cmd ? "\\boldsymbol{" + n + "}" : "\\mathbf{" + n + "}";
    }

    // Render index slots.
    //
    // Pure-level (all upper or all lower): flat grouping, e.g. ^{ijk}.
    //
    // Mixed-level, OmitVoidIndexPlaceholders set: separate ^{...} and _{...}
    // bands with no \cdot markers — appropriate for objects whose symmetry
    // makes slot order unimportant (e.g. Kronecker delta).
    //
    // Mixed-level, default: positional interleaving — each slot contributes
    // its index to its own band and \cdot to the other, making position
    // explicit.  E.g. [upper-i, lower-j] → ^{i\cdot}_{\cdot j}.
    auto slots_str(
        std::vector<SlotBinding> const& slots,
        mpk::mix::EnumFlags<RenderHint> hints) -> std::string
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
                auto s = index_str(sb.index, sb.slot.space);
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
            auto s = index_str(sb.index, sb.slot.space);
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

    static auto rational_str(Rational const& r) -> std::string
    {
        if (r.is_integer())
            return std::to_string(r.num());
        return "\\frac{" + std::to_string(r.num()) + "}{"
               + std::to_string(r.den()) + "}";
    }

    // Render a sum-annotation node.  Body is rendered first so that the
    // index name is allocated before the annotation prefix is composed.
    auto sum_str(CountableIndex idx, Expr const* body, std::string const& sym)
        -> std::string
    {
        auto body_s = render(*body);
        if (prec(*body) == ADD_PREC)
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
                    mpk::mix::EnumFlags<RenderHint> hints;
                    if (t.traits)
                        hints = t.traits->render_hints;
                    return name_str(t.name, t.rank) + slots_str(t.slots, hints);
                },
                [&](ScalarLiteral const& s) -> std::string
                { return rational_str(s.value); },
                [&](Negate const& n) -> std::string
                {
                    // Wrap only if the operand is a sum/difference — those
                    // have lower precedence than multiplication, so "-a + b"
                    // would be misread without parens. Products and atoms
                    // don't need parens: "-a \, b" is unambiguous.
                    return "-" + sub(*n.operand, MUL_PREC);
                },
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
                    return sub(*p.left, MUL_PREC) + sep
                           + sub(*p.right, MUL_PREC);
                },
                [&](ScalarDiv const& d) -> std::string {
                    return "\\frac{" + render(*d.left) + "}{" + render(*d.right)
                           + "}";
                },
                [&](Dot const& d) -> std::string
                {
                    return sub(*d.left, MUL_PREC) + " \\cdot "
                           + sub_contraction_right(*d.right);
                },
                [&](DDot const& d) -> std::string {
                    return sub(*d.left, MUL_PREC) + " : "
                           + sub_contraction_right(*d.right);
                },
                [&](DDotAlt const& d) -> std::string
                {
                    return sub(*d.left, MUL_PREC) + " \\cdot\\!\\cdot "
                           + sub_contraction_right(*d.right);
                },
                [&](Cross const& c) -> std::string
                {
                    return sub(*c.left, MUL_PREC) + " \\times "
                           + sub_contraction_right(*c.right);
                },
                [&](ExplicitSum const& s) -> std::string
                { return sum_str(s.index, s.body, "\\sum"); },
                [&](NoSum const& s) -> std::string
                { return sum_str(s.index, s.body, "\\cancel{\\sum}"); },
            },
            e);
    }
};

} // anonymous namespace

auto render_latex(Expr const& e, IndexNameMap& map) -> std::string
{
    return Renderer{map}.render(e);
}

} // namespace tender
