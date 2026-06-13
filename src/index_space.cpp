#include <tender/index_space.hpp>

#include <stdexcept>

namespace tender
{

IndexSpace::IndexSpace(std::vector<int> values, std::vector<IndexName> schema) :
  values_{std::move(values)}, schema_{std::move(schema)}
{
    if (values_.empty())
        throw std::invalid_argument("IndexSpace: value set must not be empty");
    if (schema_.empty())
        throw std::invalid_argument(
            "IndexSpace: naming schema must not be empty");
}

auto IndexSpace::dummy_name(int n) const -> IndexName
{
    if (n < 0 || n >= static_cast<int>(schema_.size()))
        throw std::out_of_range("IndexSpace::dummy_name: index out of range");
    return schema_[n];
}

// ---- Well-known index spaces -------------------------------------------

auto space_3d() -> IndexSpace const*
{
    static IndexSpace const instance{
        {1, 2, 3},
        {make_index_name("i"),
         make_index_name("j"),
         make_index_name("k"),
         make_index_name("l"),
         make_index_name("m"),
         make_index_name("n"),
         make_index_name("p"),
         make_index_name("q"),
         make_index_name("r"),
         make_index_name("s"),
         make_index_name("t"),
         make_index_name("u"),
         make_index_name("v"),
         make_index_name("w"),
         make_index_name("x"),
         make_index_name("y"),
         make_index_name("z")}};
    return &instance;
}

auto space_2d() -> IndexSpace const*
{
    static IndexSpace const instance{
        {1, 2},
        {make_index_name("\\alpha"),
         make_index_name("\\beta"),
         make_index_name("\\gamma"),
         make_index_name("\\delta"),
         make_index_name("\\epsilon"),
         make_index_name("\\zeta"),
         make_index_name("\\eta"),
         make_index_name("\\theta")}};
    return &instance;
}

auto space_4d() -> IndexSpace const*
{
    static IndexSpace const instance{
        {0, 1, 2, 3},
        {make_index_name("\\mu"),
         make_index_name("\\nu"),
         make_index_name("\\rho"),
         make_index_name("\\sigma"),
         make_index_name("\\lambda"),
         make_index_name("\\kappa"),
         make_index_name("\\tau"),
         make_index_name("\\upsilon"),
         make_index_name("\\phi"),
         make_index_name("\\chi"),
         make_index_name("\\psi"),
         make_index_name("\\omega")}};
    return &instance;
}

} // namespace tender
