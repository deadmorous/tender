#include <tender/tensor_symmetry.hpp>

#include <tender/permutation_spec.hpp>
#include <tender/tensor_order.hpp> // space_cmp, index_assoc_cmp

#include <cstddef>

namespace tender
{

auto slot_seq_cmp(
    std::vector<SlotBinding> const& a, std::vector<SlotBinding> const& b) -> int
{
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].slot.level != b[i].slot.level)
            return a[i].slot.level < b[i].slot.level ? -1 : 1;
        if (a[i].slot.realm != b[i].slot.realm)
            return a[i].slot.realm < b[i].slot.realm ? -1 : 1;
        if (int c = space_cmp(a[i].slot.space, b[i].slot.space))
            return c;
        if (int c = index_assoc_cmp(a[i].index, b[i].index))
            return c;
    }
    return 0;
}

namespace
{

// Apply a permutation (image[i] = destination of slot i) to a slot sequence.
auto permute_slots(
    std::vector<SlotBinding> const& slots,
    PermutationView const& p) -> std::vector<SlotBinding>
{
    std::vector<SlotBinding> out(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i)
        out[p[i]] = slots[i];
    return out;
}

} // namespace

auto canon_symmetry_slots(TensorObject const& t)
    -> std::pair<std::vector<SlotBinding>, int>
{
    if (!t.traits)
        return {t.slots, +1};
    auto const& sym = t.traits->symmetry.generators;
    auto const& asym = t.traits->antisymmetry.generators;
    if (sym.empty() && asym.empty())
        return {t.slots, +1};

    struct Reached final
    {
        std::vector<SlotBinding> slots;
        int sign;
    };
    std::vector<Reached> seen;
    seen.push_back({t.slots, +1});

    auto find_seen = [&](std::vector<SlotBinding> const& s) -> Reached*
    {
        for (auto& r: seen)
            if (slot_seq_cmp(r.slots, s) == 0)
                return &r;
        return nullptr;
    };

    // Breadth-first closure of the slot sequence under the generators.  The
    // group is finite, so the forward closure (no explicit inverses) reaches
    // every element.  Orbits are tiny (≤ |group|), so linear search suffices.
    for (std::size_t head = 0; head < seen.size(); ++head)
    {
        Reached const cur = seen[head]; // copy: seen may reallocate
        auto step = [&](PermutationView const& p, int gsign) -> bool
        {
            // Defensive: every generator we install matches the slot count.
            if (p.size() != cur.slots.size()) // GCOV_EXCL_LINE
                return false;                 // GCOV_EXCL_LINE
            auto next = permute_slots(cur.slots, p);
            int const nsign = cur.sign * gsign;
            if (auto* r = find_seen(next))
                return r->sign != nsign; // conflict ⇒ identically zero
            seen.push_back({std::move(next), nsign});
            return false;
        };
        // A symmetry step alone never sign-conflicts (all +1); a conflict
        // through the symmetry loop needs a tensor with *both* symmetry and
        // antisymmetry generators, which none we build has yet.
        for (auto const& p: sym)
            if (step(p, +1))    // GCOV_EXCL_LINE
                return {{}, 0}; // GCOV_EXCL_LINE
        for (auto const& p: asym)
            if (step(p, -1))
                return {{}, 0};
    }

    Reached const* best = &seen.front();
    for (auto const& r: seen)
        if (slot_seq_cmp(r.slots, best->slots) < 0)
            best = &r;
    return {best->slots, best->sign};
}

} // namespace tender
