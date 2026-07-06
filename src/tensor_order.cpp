#include <tender/tensor_order.hpp>

#include <tender/expr.hpp> // TensorObject

#include <mpk/mix/util/overloads.hpp>

#include <variant>

namespace tender
{

using mpk::mix::Overloads;

auto name_view_cmp(std::string_view a, std::string_view b) -> int
{
    return a.compare(b);
}

auto space_cmp(IndexSpace const* a, IndexSpace const* b) -> int
{
    if (a == b)
        return 0;
    auto va = a->values(), vb = b->values();
    if (va.size() != vb.size())
        return va.size() < vb.size() ? -1 : 1;
    for (std::size_t i = 0; i < va.size(); ++i)
        if (va[i] != vb[i])
            return va[i] < vb[i] ? -1 : 1;
    // Same value set but distinct instances: fall back to pointer for a stable
    // within-run order.
    return a < b ? -1 : 1;
}

auto index_assoc_cmp(
    std::optional<IndexAssoc> const& a,
    std::optional<IndexAssoc> const& b) -> int
{
    if (!a && !b)
        return 0;
    if (!a)
        return -1;
    if (!b)
        return 1;
    if (a->index() != b->index())
        return a->index() < b->index() ? -1 : 1;
    return std::visit(
        Overloads{
            [&](CountableIndex const& ca) -> int
            {
                auto id = std::get<CountableIndex>(*b).id;
                return ca.id == id ? 0 : (ca.id < id ? -1 : 1);
            },
            [&](ConcreteIndex const& ca) -> int
            {
                auto v = std::get<ConcreteIndex>(*b).value;
                return ca.value == v ? 0 : (ca.value < v ? -1 : 1);
            },
            [&](LabelIndex const& la) -> int
            {
                return name_view_cmp(
                    la.name.v.view(), std::get<LabelIndex>(*b).name.v.view());
            }},
        *a);
}

auto tensor_object_cmp(TensorObject const& a, TensorObject const& b) -> int
{
    if (int c = name_view_cmp(a.name.v.view(), b.name.v.view()))
        return c;
    if (a.rank != b.rank)
        return a.rank < b.rank ? -1 : 1;
    if (a.slots.size() != b.slots.size())
        return a.slots.size() < b.slots.size() ? -1 : 1;
    for (std::size_t i = 0; i < a.slots.size(); ++i)
    {
        auto const& sa = a.slots[i];
        auto const& sb = b.slots[i];
        if (sa.slot.level != sb.slot.level)
            return sa.slot.level < sb.slot.level ? -1 : 1;
        if (sa.slot.realm != sb.slot.realm)
            return sa.slot.realm < sb.slot.realm ? -1 : 1;
        if (int c = space_cmp(sa.slot.space, sb.slot.space))
            return c;
        if (sa.slot.basis_id != sb.slot.basis_id)
            return sa.slot.basis_id < sb.slot.basis_id ? -1 : 1;
        if (int c = index_assoc_cmp(sa.index, sb.index))
            return c;
    }
    // Applied-derivative marks are part of identity (vibe 000077 step D): ∂_x T
    // ≠ T, and the sorted multi-index makes ∂_x∂_y T = ∂_y∂_x T.
    if (a.deriv_marks.size() != b.deriv_marks.size())
        return a.deriv_marks.size() < b.deriv_marks.size() ? -1 : 1;
    for (std::size_t i = 0; i < a.deriv_marks.size(); ++i)
    {
        auto const& da = a.deriv_marks[i];
        auto const& db = b.deriv_marks[i];
        if (da.wrt.chart_id != db.wrt.chart_id)
            return da.wrt.chart_id < db.wrt.chart_id ? -1 : 1;
        if (da.wrt.slot != db.wrt.slot)
            return da.wrt.slot < db.wrt.slot ? -1 : 1;
        if (da.link != db.link)
            return da.link < db.link ? -1 : 1;
    }
    return 0;
}

} // namespace tender
