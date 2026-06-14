#pragma once

#include <array>
#include <bit>
#include <cstdint>

namespace tender
{

// A fixed-rank index permutation stored as its image array.
// image[i] = the slot position that position i maps to (values 0..N-1).
template <std::size_t N>
struct Permutation final
{
    std::array<uint8_t, N> image;
    auto operator==(Permutation const&) const -> bool = default;
};

// Owning snapshot of one unpacked permutation.  Returned by the
// PermutationSpec iterator.  Not template-parameterised on rank; the
// actual rank is stored in sz.  Capacity matches PermutationSpec::MaxRank.
//
// Owning (rather than a span) is necessary because the iterator unpacks
// bit-packed storage into its own cached_ field, and PermutationView must
// remain valid after the iterator is copied or advanced.
struct PermutationView final
{
    static constexpr std::size_t Capacity = 25;
    std::array<uint8_t, Capacity> image = {};
    uint8_t sz = 0;

    auto operator[](std::size_t i) const -> uint8_t
    {
        return image[i];
    }
    auto size() const -> std::size_t
    {
        return sz;
    }
    auto rank() const -> std::size_t
    {
        return sz;
    }
};

// Compact inline storage for a sequence of same-rank permutation generators.
// Each position is packed into ceil(log2(rank)) bits; a full permutation of
// rank N occupies ceil(N * ceil(log2(N)) / 8) bytes.
//
// Example bit widths:
//   rank 2-4  → 2 bits/position → 1 byte/permutation
//   rank 5-8  → 3 bits/position → 3 bytes/permutation
//   rank 9-16 → 4 bits/position → 8 bytes/permutation
//
// MaxBytes = 16 accommodates up to MaxRank = 25 with at least one generator.
// Provides a random-access const iterator yielding PermutationView.
class PermutationSpec final
{
public:
    static constexpr std::size_t MaxBytes = 16;
    static constexpr std::size_t MaxRank = 25;

    // Bits needed to represent any position 0..rank-1.
    static constexpr auto bits_per_pos_for(uint8_t rank) -> uint8_t
    {
        if (rank <= 1)
            return 1;
        return static_cast<uint8_t>(
            std::bit_width(static_cast<unsigned>(rank - 1u)));
    }
    // Bytes needed to store one packed permutation of the given rank.
    static constexpr auto bytes_per_perm_for(uint8_t rank) -> uint8_t
    {
        return static_cast<uint8_t>(
            (static_cast<unsigned>(bits_per_pos_for(rank)) * rank + 7u) / 8u);
    }

    // Random-access const iterator over the permutation sequence.
    // Each dereference unpacks the bit-packed bytes at the current position
    // into the iterator's cached_ field and returns a reference to it.
    // The reference is valid until the next dereference on the same object.
    class const_iterator final
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = PermutationView;
        using difference_type = std::ptrdiff_t;
        using pointer = PermutationView const*;
        using reference = PermutationView const&;

        const_iterator() = default;
        const_iterator(
            uint8_t const* ptr, uint8_t rank, uint8_t bpp, uint8_t Bpp) :
          ptr_(ptr), rank_(rank), bpp_(bpp), Bpp_(Bpp)
        {
        }

        auto operator*() const -> PermutationView const&
        {
            unpack();
            return cached_;
        }
        auto operator->() const -> PermutationView const*
        {
            return &**this;
        }
        // Returns PermutationView by value — safe even for temporary iterators
        // because PermutationView owns its unpacked bytes.
        auto operator[](difference_type n) const -> PermutationView
        {
            return *(*this + n);
        }

        auto operator++() -> const_iterator&
        {
            ptr_ += Bpp_;
            return *this;
        }
        auto operator++(int) -> const_iterator
        {
            auto t = *this;
            ++*this;
            return t;
        }
        auto operator--() -> const_iterator&
        {
            ptr_ -= Bpp_;
            return *this;
        }
        auto operator--(int) -> const_iterator
        {
            auto t = *this;
            --*this;
            return t;
        }
        auto operator+=(difference_type n) -> const_iterator&
        {
            ptr_ += n * Bpp_;
            return *this;
        }
        auto operator-=(difference_type n) -> const_iterator&
        {
            ptr_ -= n * Bpp_;
            return *this;
        }

        friend auto operator+(const_iterator it, difference_type n)
            -> const_iterator
        {
            it += n;
            return it;
        }
        friend auto operator+(difference_type n, const_iterator it)
            -> const_iterator
        {
            it += n;
            return it;
        }
        friend auto operator-(const_iterator it, difference_type n)
            -> const_iterator
        {
            it -= n;
            return it;
        }
        friend auto operator-(const_iterator a, const_iterator b)
            -> difference_type
        {
            if (a.Bpp_ == 0)
                return 0;
            return (a.ptr_ - b.ptr_) / a.Bpp_;
        }

        auto operator==(const_iterator const& o) const -> bool
        {
            return ptr_ == o.ptr_;
        }
        auto operator<=>(const_iterator const& o) const
        {
            return ptr_ <=> o.ptr_;
        }

    private:
        void unpack() const
        {
            for (uint8_t i = 0; i < rank_; ++i)
            {
                uint32_t bit_off = i * bpp_;
                uint32_t byte_idx = bit_off / 8;
                uint32_t shift = bit_off % 8;
                uint32_t raw = ptr_[byte_idx];
                if (shift + bpp_ > 8)
                    raw |= static_cast<uint32_t>(ptr_[byte_idx + 1]) << 8;
                cached_.image[i] =
                    static_cast<uint8_t>((raw >> shift) & ((1u << bpp_) - 1u));
            }
            cached_.sz = rank_;
        }

        uint8_t const* ptr_ = nullptr;
        uint8_t rank_ = 0;
        uint8_t bpp_ = 0; // bits per position
        uint8_t Bpp_ = 0; // bytes per permutation (stride)
        mutable PermutationView cached_;
    };

    // Default: empty spec (zero generators, rank 0).
    PermutationSpec() = default;

    // Construct from a same_level_only flag and one or more same-rank
    // permutations.  All arguments after same_level_only must be
    // Permutation<N> for the same N, deduced from the first permutation.
    template <std::size_t N, std::same_as<Permutation<N>>... Rest>
    explicit PermutationSpec(
        bool same_level_only, Permutation<N> p0, Rest... ps) :
      rank_(static_cast<uint8_t>(N)),
      num_gens_(static_cast<uint8_t>(1 + sizeof...(ps))),
      bits_per_pos_(bits_per_pos_for(static_cast<uint8_t>(N))),
      bytes_per_perm_(bytes_per_perm_for(static_cast<uint8_t>(N))),
      same_level_only_(same_level_only)
    {
        static_assert(N > 0, "permutation rank must be at least 1");
        static_assert(N <= MaxRank, "rank exceeds PermutationSpec::MaxRank");
        static_assert(
            bytes_per_perm_for(static_cast<uint8_t>(N)) * (1 + sizeof...(ps))
                <= MaxBytes,
            "too many generators: exceeds PermutationSpec::MaxBytes");
        uint8_t offset = 0;
        auto pack_one = [&](auto const& p)
        {
            for (uint8_t i = 0; i < N; ++i)
                pack_pos(data_.data() + offset, i, bits_per_pos_, p.image[i]);
            offset += bytes_per_perm_;
        };
        pack_one(p0);
        (pack_one(ps), ...);
    }

    auto begin() const -> const_iterator
    {
        return {data_.data(), rank_, bits_per_pos_, bytes_per_perm_};
    }
    auto end() const -> const_iterator
    {
        return {
            data_.data() + bytes_per_perm_ * num_gens_,
            rank_,
            bits_per_pos_,
            bytes_per_perm_};
    }
    auto size() const -> std::size_t
    {
        return num_gens_;
    }
    auto empty() const -> bool
    {
        return num_gens_ == 0;
    }
    auto rank() const -> uint8_t
    {
        return rank_;
    }
    auto same_level_only() const -> bool
    {
        return same_level_only_;
    }

    auto operator==(PermutationSpec const&) const -> bool = default;

private:
    // Pack position value into bit field at pos_idx * bpp bits inside base[].
    static void pack_pos(
        uint8_t* base, uint8_t pos_idx, uint8_t bpp, uint8_t value)
    {
        uint32_t bit_off = pos_idx * bpp;
        uint32_t byte_idx = bit_off / 8;
        uint32_t shift = bit_off % 8;
        uint32_t mask = ((1u << bpp) - 1u) << shift;
        base[byte_idx] = static_cast<uint8_t>(
            (base[byte_idx] & ~mask) | ((value << shift) & mask));
        if (shift + bpp > 8)
        {
            uint32_t overflow = (shift + bpp) - 8;
            uint32_t omask = (1u << overflow) - 1u;
            base[byte_idx + 1] = static_cast<uint8_t>(
                (base[byte_idx + 1] & ~omask) | (value >> (bpp - overflow)));
        }
    }

    uint8_t rank_ = 0;
    uint8_t num_gens_ = 0;
    uint8_t bits_per_pos_ = 0;
    uint8_t bytes_per_perm_ = 0;
    bool same_level_only_ = true;
    std::array<uint8_t, MaxBytes> data_ = {};
};

} // namespace tender
