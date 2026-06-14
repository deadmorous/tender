#pragma once

#include <array>
#include <cstdint>
#include <iterator>
#include <span>

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

// Non-owning view of one permutation inside PermutationSpec storage.
// The span points directly into the spec's internal byte array; the view
// is valid only while the owning PermutationSpec is alive and unmoved.
struct PermutationView final
{
    std::span<uint8_t const> image;
    auto operator[](std::size_t i) const -> uint8_t
    {
        return image[i];
    }
    auto size() const -> std::size_t
    {
        return image.size();
    }
    auto rank() const -> std::size_t
    {
        return image.size();
    }
};

// Compact inline storage for a sequence of same-rank permutation generators.
// Permutations are packed as contiguous byte arrays (one byte per position).
// MaxBytes / rank generators fit; a static_assert fires at construction if
// that limit is exceeded.
//
// Provides a random-access const iterator yielding PermutationView.
class PermutationSpec final
{
public:
    static constexpr std::size_t MaxBytes = 16;

    // Random-access const iterator over the permutation sequence.
    // Each dereference materialises a PermutationView into the iterator's
    // private cached_ field and returns a reference to it.  References are
    // valid until the next dereference on the same iterator object.
    class const_iterator final
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = PermutationView;
        using difference_type = std::ptrdiff_t;
        using pointer = PermutationView const*;
        using reference = PermutationView const&;

        const_iterator() = default;
        const_iterator(uint8_t const* ptr, uint8_t rank) :
          ptr_(ptr), rank_(rank)
        {
        }

        auto operator*() const -> PermutationView const&
        {
            cached_ = PermutationView{std::span<uint8_t const>(ptr_, rank_)};
            return cached_;
        }
        auto operator->() const -> PermutationView const*
        {
            return &**this;
        }
        auto operator[](difference_type n) const -> PermutationView
        {
            return *(*this + n);
        }

        auto operator++() -> const_iterator&
        {
            ptr_ += rank_;
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
            ptr_ -= rank_;
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
            ptr_ += n * rank_;
            return *this;
        }
        auto operator-=(difference_type n) -> const_iterator&
        {
            ptr_ -= n * rank_;
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
            if (a.rank_ == 0)
                return 0;
            return (a.ptr_ - b.ptr_) / a.rank_;
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
        uint8_t const* ptr_ = nullptr;
        uint8_t rank_ = 0;
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
      same_level_only_(same_level_only)
    {
        static_assert(N > 0, "permutation rank must be at least 1");
        static_assert(
            N * (1 + sizeof...(ps)) <= MaxBytes,
            "too many generators: exceeds PermutationSpec::MaxBytes");
        uint8_t offset = 0;
        auto copy_one = [&](auto const& p)
        {
            for (std::size_t i = 0; i < N; ++i)
                data_[offset++] = p.image[i];
        };
        copy_one(p0);
        (copy_one(ps), ...);
    }

    auto begin() const -> const_iterator
    {
        return {data_.data(), rank_};
    }
    auto end() const -> const_iterator
    {
        return {data_.data() + rank_ * num_gens_, rank_};
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
    uint8_t rank_ = 0;
    uint8_t num_gens_ = 0;
    bool same_level_only_ = true;
    std::array<uint8_t, MaxBytes> data_ = {};
};

} // namespace tender
