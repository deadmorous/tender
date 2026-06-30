#pragma once

#include <mpk/mix/strong.hpp>
#include <mpk/mix/types/fixed_length_string.hpp>

#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tender
{

// ---- underlying storage ------------------------------------------------

using NameStr = mpk::mix::FixedLengthString<16>;

// ---- TensorName --------------------------------------------------------
//
// Name of a TensorObject: must be a single ASCII letter or a LaTeX command
// (backslash followed by one or more ASCII letters), e.g. "A", "\sigma".
// Multi-letter words are rejected.

MPKMIX_STRONG_STRING_VIEW(TensorNameView);
MPKMIX_STRONG_TYPE(
    TensorName, NameStr, mpk::mix::StrongStringFeatures<TensorNameView>);

// ---- IndexName ---------------------------------------------------------
//
// Name of an index: same as TensorName, plus multi-letter ASCII words for
// descriptive labels, e.g. "i", "\mu", "vol", "surf".

MPKMIX_STRONG_STRING_VIEW(IndexNameView);
MPKMIX_STRONG_TYPE(
    IndexName, NameStr, mpk::mix::StrongStringFeatures<IndexNameView>);

// ---- validation helpers ------------------------------------------------

namespace detail
{

inline auto is_latex_command(std::string_view s) noexcept -> bool
{
    if (s.size() < 2 || s[0] != '\\')
        return false;
    for (auto it = s.begin() + 1; it != s.end(); ++it)
        if (!std::isalpha(static_cast<unsigned char>(*it)))
            return false;
    return true;
}

inline auto is_alpha_word(std::string_view s) noexcept -> bool
{
    if (s.empty())
        return false;
    for (char c: s)
        if (!std::isalpha(static_cast<unsigned char>(c)))
            return false;
    return true;
}

} // namespace detail

// ---- factory functions -------------------------------------------------

[[nodiscard]] inline auto make_tensor_name(std::string_view s) -> TensorName
{
    bool valid =
        (s.size() == 1 && std::isalpha(static_cast<unsigned char>(s[0])))
        || detail::is_latex_command(s);
    if (!valid)
        throw std::invalid_argument(
            "TensorName must be a single ASCII letter (e.g. \"x\") or a LaTeX "
            "command (e.g. \"\\phi\", \"\\sigma\"); a plain multi-letter name "
            "like \""
            + std::string{s}
            + "\" is not allowed — for a Greek letter use its LaTeX command "
              "(e.g. \"\\"
            + std::string{s} + "\").");
    return TensorName{NameStr{s}};
}

[[nodiscard]] inline auto make_index_name(std::string_view s) -> IndexName
{
    bool valid = detail::is_alpha_word(s) || detail::is_latex_command(s);
    if (!valid)
        throw std::invalid_argument(
            "IndexName must be a single letter, a LaTeX command, "
            "or a multi-letter ASCII word");
    return IndexName{NameStr{s}};
}

} // namespace tender
