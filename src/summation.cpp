#include <tender/summation.hpp>

#include <mpk/mix/util/overloads.hpp>
#include <tender/rewrite.hpp>

#include <stdexcept>
#include <variant>

namespace tender
{

using mpk::mix::Overloads;

// ---- α-renaming --------------------------------------------------------

auto substitute_index_id(Context& ctx, Expr const* e, int old_id, int new_id)
    -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [old_id, new_id](Context& ctx, Expr const* e) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&e->node);
            if (!t)
                return e;
            auto slots = t->slots;
            bool changed = false;
            for (auto& sb: slots)
            {
                if (!sb.index)
                    continue;
                if (auto const* ci = std::get_if<CountableIndex>(&*sb.index))
                    if (ci->id == old_id)
                    {
                        sb.index = CountableIndex{new_id};
                        changed = true;
                    }
            }
            if (!changed)
                return e;
            // Copy the whole object so field_derivs (the ∂ directions of a
            // field, vibe 000073) survive index substitution — rebuilding with
            // only name/rank/traits/slots silently drops the derivative.
            TensorObject obj = *t;
            obj.slots = std::move(slots);
            return ctx.make<Expr>(std::move(obj));
        });
}

auto substitute_index_ids(
    Context& ctx, Expr const* e, std::map<int, int> const& remap) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [&remap](Context& ctx, Expr const* e) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&e->node);
            if (!t)
                return e;
            auto slots = t->slots;
            bool changed = false;
            for (auto& sb: slots)
            {
                if (!sb.index)
                    continue;
                if (auto const* ci = std::get_if<CountableIndex>(&*sb.index))
                    if (auto it = remap.find(ci->id); it != remap.end())
                    {
                        sb.index = CountableIndex{it->second};
                        changed = true;
                    }
            }
            if (!changed)
                return e;
            // Copy the whole object so field_derivs (the ∂ directions of a
            // field, vibe 000073) survive index substitution — rebuilding with
            // only name/rank/traits/slots silently drops the derivative.
            TensorObject obj = *t;
            obj.slots = std::move(slots);
            return ctx.make<Expr>(std::move(obj));
        });
}

auto bound_canon_id(int depth) -> int
{
    return -(depth + 1);
}

// ---- implicit (Einstein) summation detection ---------------------------

auto is_term(Expr const* e) -> bool
{
    return visit(
        Overloads{
            [](TensorObject const&) { return true; },
            [](ScalarLiteral const&) { return true; },
            [](Negate const& n) { return is_term(n.operand); },
            [](Trace const& u) { return is_term(u.operand); },
            [](VectorInvariant const& u) { return is_term(u.operand); },
            [](Transpose const& u) { return is_term(u.operand); },
            [](TensorProduct const& p)
            { return is_term(p.left) && is_term(p.right); },
            [](Dot const& d) { return is_term(d.left) && is_term(d.right); },
            [](DDot const& d) { return is_term(d.left) && is_term(d.right); },
            [](DDotAlt const& d)
            { return is_term(d.left) && is_term(d.right); },
            [](Cross const& c) { return is_term(c.left) && is_term(c.right); },
            [](auto const&) { return false; }, // Sum/Diff/ScalarDiv/binders
        },
        *e);
}

void collect_term_uses(
    Expr const* e,
    std::set<int> const& bound,
    std::map<int, std::vector<IndexUse>>& uses)
{
    auto bin = [&](Expr const* l, Expr const* r)
    {
        collect_term_uses(l, bound, uses);
        collect_term_uses(r, bound, uses);
    };
    visit(
        Overloads{
            [&](TensorObject const& t)
            {
                for (auto const& sb: t.slots)
                {
                    if (!sb.index)
                        continue;
                    auto const* ci = std::get_if<CountableIndex>(&*sb.index);
                    if (!ci || bound.count(ci->id))
                        continue;
                    uses[ci->id].push_back({sb.slot.level, sb.slot.realm});
                }
            },
            [&](Negate const& n) { collect_term_uses(n.operand, bound, uses); },
            [&](Trace const& u) { collect_term_uses(u.operand, bound, uses); },
            [&](VectorInvariant const& u)
            { collect_term_uses(u.operand, bound, uses); },
            [&](Transpose const& u)
            { collect_term_uses(u.operand, bound, uses); },
            [&](TensorProduct const& p) { bin(p.left, p.right); },
            [&](Dot const& d) { bin(d.left, d.right); },
            [&](DDot const& d) { bin(d.left, d.right); },
            [&](DDotAlt const& d) { bin(d.left, d.right); },
            [&](Cross const& c) { bin(c.left, c.right); },
            [&](auto const&) {}, // ScalarLiteral / scope boundaries: opaque
        },
        *e);
}

auto contracted_ids(Expr const* term, std::set<int> const& bound)
    -> std::vector<int>
{
    std::map<int, std::vector<IndexUse>> uses;
    collect_term_uses(term, bound, uses);

    std::vector<int> result;
    for (auto const& [id, us]: uses)
    {
        std::size_t const n = us.size();
        switch (us.front().realm)
        {
            case Realm::Oblique:
                if (n == 1)
                    break;
                if (n == 2 && us[0].level != us[1].level)
                {
                    result.push_back(id);
                    break;
                }
                throw std::invalid_argument(
                    "implicit summation: an Oblique index must contract exactly "
                    "one upper with one lower slot; a same-level pair or three "
                    "or more occurrences requires an explicit ExplicitSum/NoSum");
            case Realm::Orthonormal:
                if (n == 1)
                    break;
                if (n == 2)
                {
                    result.push_back(id);
                    break;
                }
                throw std::invalid_argument(
                    "implicit summation: an Orthonormal index occurs three or "
                    "more times; this requires an explicit ExplicitSum/NoSum");
            case Realm::Collection:
            case Realm::Label: break; // never auto-contract
        }
    }
    return result;
}

} // namespace tender
