#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tender/context.hpp>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/index.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>
#include <tender/rational.hpp>
#include <tender/render.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace tender;

// Module-level default context — lives for the duration of the process.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static Context g_default_ctx;

// Python wrapper for Expr. Holds ctx_keep (a Python object reference) so
// that the owning Context cannot be garbage-collected while this Expr lives.
// For expressions created in g_default_ctx, ctx_keep is nb::none() —
// no extra reference is needed because g_default_ctx is a static.
struct PyExpr
{
    nb::object ctx_keep;
    Context* ctx;
    Expr const* expr;
};

// Given an optional Python ctx argument, return (Context*, keep-alive ref).
// None  → g_default_ctx, nb::none()
// obj   → cast to Context*, the object itself
static auto resolve_ctx(nb::object const& arg) -> std::pair<Context*, nb::object>
{
    if (arg.is_none())
        return {&g_default_ctx, nb::none()};
    return {nb::cast<Context*>(arg), arg};
}

// Make a new PyExpr in the same context as an existing one.
static auto derive(PyExpr const& from, Expr const* e) -> PyExpr
{
    return PyExpr{from.ctx_keep, from.ctx, e};
}

// Convert a Python value to IndexAssoc: CountableIndex | int → Concrete | str →
// Label
static auto to_index_assoc(nb::object const& v) -> IndexAssoc
{
    if (nb::isinstance<CountableIndex>(v))
        return nb::cast<CountableIndex>(v);
    if (nb::isinstance<nb::int_>(v))
        return ConcreteIndex{nb::cast<int>(v)};
    if (nb::isinstance<nb::str>(v))
        return LabelIndex{make_index_name(nb::cast<std::string>(v))};
    throw nb::type_error("index must be CountableIndex, int, or str");
}

// Convert int or Rational Python value to a scalar Expr const*.
static auto py_to_scalar(nb::object const& v, Context& ctx) -> Expr const*
{
    if (nb::isinstance<nb::int_>(v))
        return make_scalar(ctx, Rational{nb::cast<int64_t>(v)});
    if (nb::isinstance<Rational>(v))
        return make_scalar(ctx, nb::cast<Rational>(v));
    throw nb::type_error("scalar must be int or Rational");
}

NB_MODULE(_core, m)
{
    m.doc() = "tender — tensor algebra core (C++ bindings)";

    // ------------------------------------------------------------------ //
    // Rational
    // ------------------------------------------------------------------ //
    nb::class_<Rational>(m, "Rational")
        .def(nb::init<int64_t>(), "n"_a)
        .def(nb::init<int64_t, int64_t>(), "num"_a, "den"_a)
        .def("__add__", [](Rational a, Rational b) { return a + b; })
        .def("__radd__", [](Rational a, int64_t b) { return Rational{b} + a; })
        .def("__sub__", [](Rational a, Rational b) { return a - b; })
        .def("__rsub__", [](Rational a, int64_t b) { return Rational{b} - a; })
        .def("__mul__", [](Rational a, Rational b) { return a * b; })
        .def("__rmul__", [](Rational a, int64_t b) { return Rational{b} * a; })
        .def("__truediv__", [](Rational a, Rational b) { return a / b; })
        .def("__neg__", [](Rational a) { return -a; })
        .def("__eq__", [](Rational a, Rational b) { return a == b; })
        .def("__repr__", [](Rational r) { return r.to_string(); })
        .def_prop_ro("num", &Rational::num)
        .def_prop_ro("den", &Rational::den)
        .def_prop_ro("is_zero", &Rational::is_zero)
        .def_prop_ro("is_integer", &Rational::is_integer)
        .def("to_double", &Rational::to_double);

    // ------------------------------------------------------------------ //
    // Realm, Level
    // ------------------------------------------------------------------ //
    nb::enum_<Realm>(m, "Realm")
        .value("Oblique", Realm::Oblique)
        .value("Orthonormal", Realm::Orthonormal)
        .value("Collection", Realm::Collection)
        .value("Label", Realm::Label);

    nb::enum_<Level>(m, "Level")
        .value("Upper", Level::Upper)
        .value("Lower", Level::Lower);

    // ------------------------------------------------------------------ //
    // CountableIndex
    // ------------------------------------------------------------------ //
    nb::class_<CountableIndex>(m, "CountableIndex")
        .def_ro("id", &CountableIndex::id)
        .def(
            "__repr__",
            [](CountableIndex const& ci)
            { return "<Index id=" + std::to_string(ci.id) + ">"; })
        .def(
            "__eq__",
            [](CountableIndex const& a, CountableIndex const& b)
            { return a.id == b.id; })
        .def(
            "__hash__",
            [](CountableIndex const& ci) { return std::hash<int>{}(ci.id); });

    // ------------------------------------------------------------------ //
    // IndexSpace  (opaque; identity by pointer)
    // ------------------------------------------------------------------ //
    nb::class_<IndexSpace>(m, "IndexSpace")
        .def("__repr__", [](IndexSpace const&) { return "<IndexSpace>"; });

    // ------------------------------------------------------------------ //
    // Context
    // ------------------------------------------------------------------ //
    nb::class_<Context>(m, "Context")
        .def(
            "__init__",
            [](Context* self)
            { new (self) Context(g_default_ctx.new_context()); },
            "Create a context that shares the global index-id factory.")
        .def(
            "alloc_index",
            [](Context& self) -> CountableIndex
            { return CountableIndex{self.alloc_index_id()}; },
            "Allocate a fresh dummy index id.")
        .def(
            "new_context",
            [](Context& self) -> Context { return self.new_context(); },
            "Create a child context sharing the index-id factory.");

    // ------------------------------------------------------------------ //
    // IndexNameMap
    // ------------------------------------------------------------------ //
    nb::class_<IndexNameMap>(m, "IndexNameMap")
        .def(nb::init<>())
        .def(
            "assign",
            [](IndexNameMap& map, CountableIndex ci, std::string const& name)
            { map.assign(ci, make_index_name(name)); },
            "ci"_a,
            "name"_a,
            "Explicitly bind index ci to the display name.")
        .def(
            "name_for",
            [](IndexNameMap& map,
               CountableIndex ci,
               IndexSpace const* space) -> std::string
            { return std::string{map.name_for(ci, space).v.view()}; },
            "ci"_a,
            "space"_a,
            "Return display name for ci, allocating from space schema if needed.")
        .def(
            "lookup",
            [](IndexNameMap const& map,
               CountableIndex ci) -> std::optional<std::string>
            {
                auto r = map.lookup(ci);
                if (!r)
                    return std::nullopt;
                return std::string{r->v.view()};
            },
            "ci"_a,
            "Forward lookup without allocation.")
        .def(
            "index_for",
            [](IndexNameMap const& map,
               std::string const& name) -> std::optional<CountableIndex>
            { return map.index_for(make_index_name(name)); },
            "name"_a,
            "Reverse lookup: index assigned this name, if any.");

    // ------------------------------------------------------------------ //
    // Expr  (PyExpr wrapper)
    // ------------------------------------------------------------------ //
    nb::class_<PyExpr>(m, "Expr")
        // arithmetic
        .def(
            "__add__",
            [](PyExpr const& a, PyExpr const& b) -> PyExpr
            { return derive(a, make_sum(*a.ctx, a.expr, b.expr)); })
        .def(
            "__add__",
            [](PyExpr const& a, nb::object const& b) -> PyExpr
            {
                auto s = py_to_scalar(b, *a.ctx);
                return derive(a, make_sum(*a.ctx, a.expr, s));
            })
        .def(
            "__radd__",
            [](PyExpr const& a, nb::object const& b) -> PyExpr
            {
                auto s = py_to_scalar(b, *a.ctx);
                return derive(a, make_sum(*a.ctx, s, a.expr));
            })
        .def(
            "__sub__",
            [](PyExpr const& a, PyExpr const& b) -> PyExpr
            { return derive(a, make_difference(*a.ctx, a.expr, b.expr)); })
        .def(
            "__sub__",
            [](PyExpr const& a, nb::object const& b) -> PyExpr
            {
                auto s = py_to_scalar(b, *a.ctx);
                return derive(a, make_difference(*a.ctx, a.expr, s));
            })
        .def(
            "__rsub__",
            [](PyExpr const& a, nb::object const& b) -> PyExpr
            {
                auto s = py_to_scalar(b, *a.ctx);
                return derive(a, make_difference(*a.ctx, s, a.expr));
            })
        .def(
            "__neg__",
            [](PyExpr const& a) -> PyExpr
            { return derive(a, make_negate(*a.ctx, a.expr)); })
        // * = tensor product; scalar on either side is ok
        .def(
            "__mul__",
            [](PyExpr const& a, PyExpr const& b) -> PyExpr
            { return derive(a, make_tensor_product(*a.ctx, a.expr, b.expr)); })
        .def(
            "__mul__",
            [](PyExpr const& a, nb::object const& b) -> PyExpr
            {
                auto s = py_to_scalar(b, *a.ctx);
                return derive(a, make_tensor_product(*a.ctx, a.expr, s));
            })
        .def(
            "__rmul__",
            [](PyExpr const& a, nb::object const& b) -> PyExpr
            {
                auto s = py_to_scalar(b, *a.ctx);
                return derive(a, make_tensor_product(*a.ctx, s, a.expr));
            })
        // @ = dot (·)
        .def(
            "__matmul__",
            [](PyExpr const& a, PyExpr const& b) -> PyExpr
            { return derive(a, make_dot(*a.ctx, a.expr, b.expr)); })
        // % = cross (×)
        .def(
            "__mod__",
            [](PyExpr const& a, PyExpr const& b) -> PyExpr
            { return derive(a, make_cross(*a.ctx, a.expr, b.expr)); })
        // / = scalar division
        .def(
            "__truediv__",
            [](PyExpr const& a, PyExpr const& b) -> PyExpr
            { return derive(a, make_scalar_div(*a.ctx, a.expr, b.expr)); })
        .def(
            "__truediv__",
            [](PyExpr const& a, nb::object const& b) -> PyExpr
            {
                auto s = py_to_scalar(b, *a.ctx);
                return derive(a, make_scalar_div(*a.ctx, a.expr, s));
            })
        // double-contraction (no natural operator)
        .def(
            "ddot",
            [](PyExpr const& a, PyExpr const& b) -> PyExpr
            { return derive(a, make_ddot(*a.ctx, a.expr, b.expr)); },
            "other"_a,
            "Double contraction (:).")
        .def(
            "ddot_alt",
            [](PyExpr const& a, PyExpr const& b) -> PyExpr
            { return derive(a, make_ddot_alt(*a.ctx, a.expr, b.expr)); },
            "other"_a,
            "Alternate double contraction (·̣·).")
        // rendering
        .def(
            "_repr_latex_",
            [](PyExpr const& e) -> std::string
            {
                IndexNameMap map;
                return "$" + render_latex(*e.expr, map) + "$";
            },
            "Return LaTeX string for Jupyter rich display.")
        .def(
            "latex",
            [](PyExpr const& e, IndexNameMap* map_ptr) -> std::string
            {
                if (map_ptr)
                    return render_latex(*e.expr, *map_ptr);
                IndexNameMap fresh;
                return render_latex(*e.expr, fresh);
            },
            "map"_a = nb::none(),
            "Render to a LaTeX math string (no surrounding $).")
        .def(
            "__repr__",
            [](PyExpr const& e) -> std::string
            {
                IndexNameMap map;
                return render_latex(*e.expr, map);
            });

    // ------------------------------------------------------------------ //
    // Module-level factory functions
    // All accept an optional ctx= keyword (default: module-level
    // g_default_ctx).
    // ------------------------------------------------------------------ //

    m.def(
        "tensor",
        [](std::string const& name,
           std::optional<int> rank,
           nb::object ctx_arg) -> PyExpr
        {
            auto [ctx, keep] = resolve_ctx(ctx_arg);
            return PyExpr{
                keep,
                ctx,
                make_tensor_object(*ctx, make_tensor_name(name), {}, rank)};
        },
        "name"_a,
        "rank"_a = nb::none(),
        "ctx"_a = nb::none(),
        "Create a named tensor object.");

    m.def(
        "scalar",
        [](nb::object const& val, nb::object ctx_arg) -> PyExpr
        {
            auto [ctx, keep] = resolve_ctx(ctx_arg);
            return PyExpr{keep, ctx, py_to_scalar(val, *ctx)};
        },
        "value"_a,
        "ctx"_a = nb::none(),
        "Create a scalar literal (int or Rational).");

    m.def(
        "identity",
        [](nb::object ctx_arg) -> PyExpr
        {
            auto [ctx, keep] = resolve_ctx(ctx_arg);
            return PyExpr{keep, ctx, make_identity(*ctx)};
        },
        "ctx"_a = nb::none(),
        "Create the identity tensor.");

    m.def(
        "delta",
        [](Realm realm,
           IndexSpace const* space,
           Level level0,
           Level level1,
           nb::object const& idx0,
           nb::object const& idx1,
           nb::object ctx_arg) -> PyExpr
        {
            auto [ctx, keep] = resolve_ctx(ctx_arg);
            return PyExpr{
                keep,
                ctx,
                make_delta(
                    *ctx,
                    realm,
                    space,
                    level0,
                    level1,
                    to_index_assoc(idx0),
                    to_index_assoc(idx1))};
        },
        "realm"_a,
        "space"_a,
        "level0"_a,
        "level1"_a,
        "idx0"_a,
        "idx1"_a,
        "ctx"_a = nb::none(),
        "Create a Kronecker delta.");

    m.def(
        "levi_civita",
        [](Realm realm,
           IndexSpace const* space,
           std::vector<Level> levels,
           std::vector<nb::object> indices,
           nb::object ctx_arg) -> PyExpr
        {
            auto [ctx, keep] = resolve_ctx(ctx_arg);
            std::vector<IndexAssoc> assocs;
            assocs.reserve(indices.size());
            for (auto const& idx: indices)
                assocs.push_back(to_index_assoc(idx));
            return PyExpr{
                keep,
                ctx,
                make_levi_civita(
                    *ctx, realm, space, std::move(levels), std::move(assocs))};
        },
        "realm"_a,
        "space"_a,
        "levels"_a,
        "indices"_a,
        "ctx"_a = nb::none(),
        "Create a Levi-Civita tensor.");

    m.def(
        "explicit_sum",
        [](CountableIndex idx, PyExpr const& body, nb::object ctx_arg) -> PyExpr
        {
            Context* ctx =
                ctx_arg.is_none() ? body.ctx : nb::cast<Context*>(ctx_arg);
            nb::object keep = ctx_arg.is_none() ? body.ctx_keep : ctx_arg;
            return PyExpr{keep, ctx, make_explicit_sum(*ctx, idx, body.expr)};
        },
        "index"_a,
        "body"_a,
        "ctx"_a = nb::none(),
        "Annotate body with an explicit summation over index.");

    m.def(
        "no_sum",
        [](CountableIndex idx, PyExpr const& body, nb::object ctx_arg) -> PyExpr
        {
            Context* ctx =
                ctx_arg.is_none() ? body.ctx : nb::cast<Context*>(ctx_arg);
            nb::object keep = ctx_arg.is_none() ? body.ctx_keep : ctx_arg;
            return PyExpr{keep, ctx, make_no_sum(*ctx, idx, body.expr)};
        },
        "index"_a,
        "body"_a,
        "ctx"_a = nb::none(),
        "Annotate body to suppress implicit summation over index.");

    m.def(
        "alloc_index",
        [](nb::object ctx_arg) -> CountableIndex
        {
            auto [ctx, keep] = resolve_ctx(ctx_arg);
            (void)keep;
            return CountableIndex{ctx->alloc_index_id()};
        },
        "ctx"_a = nb::none(),
        "Allocate a fresh dummy index id from the context.");

    m.def(
        "render_latex",
        [](PyExpr const& e, IndexNameMap* map_ptr) -> std::string
        {
            if (map_ptr)
                return render_latex(*e.expr, *map_ptr);
            IndexNameMap fresh;
            return render_latex(*e.expr, fresh);
        },
        "expr"_a,
        "map"_a = nb::none(),
        "Render expr to a LaTeX math string (no surrounding delimiters).");

    // ------------------------------------------------------------------ //
    // Predefined index spaces (process-lifetime singletons)
    // ------------------------------------------------------------------ //
    m.def(
        "_space_3d",
        &tender::space_3d,
        nb::rv_policy::reference,
        "3D spatial index space: values {1,2,3}.");
    m.def(
        "_space_2d",
        &tender::space_2d,
        nb::rv_policy::reference,
        "2D spatial index space: values {1,2}.");
    m.def(
        "_space_4d",
        &tender::space_4d,
        nb::rv_policy::reference,
        "4D spacetime index space: values {0,1,2,3}.");

    // ------------------------------------------------------------------ //
    // derivation submodule
    // ------------------------------------------------------------------ //
    // Steps are exposed as _name(PyExpr) -> PyExpr so that the Python
    // tender.derivation module can wrap them with clean public names.
    nb::module_ md =
        m.def_submodule("derivation", "Derivation step functions.");

    md.def(
        "_unroll_sums",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::unroll_sums(*e.ctx, e.expr)); },
        "expr"_a,
        "Expand each ExplicitSum with a concrete index space into a Sum tree.");

    md.def(
        "_eval_delta_concrete",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::eval_delta_concrete(*e.ctx, e.expr)); },
        "expr"_a,
        "Evaluate delta(a,b) with concrete indices to 1 or 0.");

    md.def(
        "_fold_arithmetic",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::fold_arithmetic(*e.ctx, e.expr)); },
        "expr"_a,
        "Constant-fold arithmetic operations on scalar literals.");

    md.def(
        "_expand_eps",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::expand_eps(*e.ctx, e.expr)); },
        "expr"_a,
        "Expand rank-3 Levi-Civita symbol to 6-term Kronecker-delta expansion.");

    md.def(
        "_fold_sums",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::fold_sums(*e.ctx, e.expr)); },
        "expr"_a,
        "Fold a concrete N-term Sum cycle into an ExplicitSum over a fresh index.");

    md.def(
        "_contract_delta",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::contract_delta(*e.ctx, e.expr)); },
        "expr"_a,
        "Contract ExplicitSum{m, δ^m_a · δ^m_b} into δ_{ab}.");
}
