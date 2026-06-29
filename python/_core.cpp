#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tender/basis.hpp>
#include <tender/chart.hpp>
#include <tender/context.hpp>
#include <tender/coord_system.hpp>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/identity.hpp>
#include <tender/index.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>
#include <tender/nf_egraph.hpp>
#include <tender/nf_lower.hpp> // nf::raise
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

// Python wrapper for Basis. Like PyExpr, holds a keep-alive reference to the
// owning Context: the Basis stores Expr const* pointers (its vectors) that
// live in that context, and its emission/step methods allocate there too.
struct PyBasis
{
    nb::object ctx_keep;
    Context* ctx;
    Basis basis;
};

// Python wrapper for CoordinateChart (vibe 000069 M4).  Like PyBasis it keeps
// the owning Context alive: the chart's reference basis, coordinate atoms and
// embedding all live there, and the geometry it derives allocates there too.
struct PyChart
{
    nb::object ctx_keep;
    Context* ctx;
    CoordinateChart chart;
};

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

// Assemble a BasisNaming from Python lists/strings (vibe 000067): value_names
// and vector_symbols are per-direction display strings; label is a free-form
// LaTeX suffix appended to the basis's indices.
static auto make_basis_naming(
    std::vector<std::string> const& value_names,
    std::vector<std::string> const& vector_symbols,
    std::optional<std::string> const& label) -> BasisNaming
{
    BasisNaming n;
    for (auto const& v: value_names)
        n.value_names.push_back(make_index_name(v));
    for (auto const& s: vector_symbols)
        n.vector_symbols.push_back(make_tensor_name(s));
    n.label = label;
    return n;
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
        // ** = power (scalar field, vibe 000069)
        .def(
            "__pow__",
            [](PyExpr const& a, PyExpr const& b) -> PyExpr
            { return derive(a, make_pow(*a.ctx, a.expr, b.expr)); })
        .def(
            "__pow__",
            [](PyExpr const& a, nb::object const& b) -> PyExpr
            {
                auto s = py_to_scalar(b, *a.ctx);
                return derive(a, make_pow(*a.ctx, a.expr, s));
            })
        // // = alternate double contraction (··); ':' is not a Python operator,
        // so ddot keeps the method form only.
        .def(
            "__floordiv__",
            [](PyExpr const& a, PyExpr const& b) -> PyExpr
            { return derive(a, make_ddot_alt(*a.ctx, a.expr, b.expr)); })
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
            "Alternate double contraction (··); also the // operator.")
        // rank-2 invariant operations
        .def(
            "tr",
            [](PyExpr const& a) -> PyExpr
            { return derive(a, make_trace(*a.ctx, a.expr)); },
            "Trace tr(A) (a scalar; tr(a⊗b) = a·b).")
        .def(
            "vec",
            [](PyExpr const& a) -> PyExpr
            { return derive(a, make_vector_invariant(*a.ctx, a.expr)); },
            "Vector invariant vec(A) (a vector; vec(a⊗b) = a×b).")
        .def(
            "transpose",
            [](PyExpr const& a) -> PyExpr
            { return derive(a, make_transpose(*a.ctx, a.expr)); },
            "Transpose A^T (transpose(a⊗b) = b⊗a).")
        // rendering
        .def(
            "_repr_latex_",
            [](PyExpr const& e) -> std::string
            {
                IndexNameMap map;
                return "$" + render_latex(*e.expr, map, e.ctx) + "$";
            },
            "Return LaTeX string for Jupyter rich display.")
        .def(
            "latex",
            [](PyExpr const& e, IndexNameMap* map_ptr) -> std::string
            {
                if (map_ptr)
                    return render_latex(*e.expr, *map_ptr, e.ctx);
                IndexNameMap fresh;
                return render_latex(*e.expr, fresh, e.ctx);
            },
            "map"_a = nb::none(),
            "Render to a LaTeX math string (no surrounding $).")
        .def_prop_ro(
            "rank",
            [](PyExpr const& e) -> std::optional<int>
            { return infer_rank(e.expr); },
            "The invariant rank, inferred through the operators (TensorProduct "
            "adds ranks, Dot removes 2, DDot 4, Cross 1, ...). None when a leaf "
            "rank is undeclared or a contraction would be ill-formed.")
        .def(
            "__repr__",
            [](PyExpr const& e) -> std::string
            {
                IndexNameMap map;
                return render_latex(*e.expr, map, e.ctx);
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

    // ---- scalar fields (vibe 000069 M1) -------------------------------- //

    m.def(
        "coordinate",
        [](std::string const& name,
           int chart_id,
           int slot,
           bool nonneg,
           nb::object ctx_arg) -> PyExpr
        {
            auto [ctx, keep] = resolve_ctx(ctx_arg);
            return PyExpr{
                keep,
                ctx,
                make_coordinate(
                    *ctx, make_tensor_name(name), chart_id, slot, nonneg)};
        },
        "name"_a,
        "chart_id"_a = 0,
        "slot"_a = 0,
        "nonneg"_a = false,
        "ctx"_a = nb::none(),
        "Create a chart coordinate variable (rank-0 scalar field).  chart_id 0 "
        "leaves it unbound to a chart; nonneg marks it ≥ 0 (enables √(x²)→x).");

    auto bind_scalar_fn =
        [&m](char const* py_name, ScalarFnKind kind, char const* doc)
    {
        m.def(
            py_name,
            [kind](PyExpr const& a) -> PyExpr
            { return derive(a, make_scalar_fn(*a.ctx, kind, a.expr)); },
            "x"_a,
            doc);
    };
    bind_scalar_fn("sin", ScalarFnKind::Sin, "Sine of a scalar field.");
    bind_scalar_fn("cos", ScalarFnKind::Cos, "Cosine of a scalar field.");
    bind_scalar_fn("tan", ScalarFnKind::Tan, "Tangent of a scalar field.");
    bind_scalar_fn("exp", ScalarFnKind::Exp, "Exponential of a scalar field.");
    bind_scalar_fn("log", ScalarFnKind::Log, "Natural log of a scalar field.");
    bind_scalar_fn("sqrt", ScalarFnKind::Sqrt, "Square root of a scalar field.");

    m.def(
        "tr",
        [](PyExpr const& a) -> PyExpr
        { return derive(a, make_trace(*a.ctx, a.expr)); },
        "A"_a,
        "Trace tr(A) (a scalar; tr(a⊗b) = a·b).");
    m.def(
        "vec",
        [](PyExpr const& a) -> PyExpr
        { return derive(a, make_vector_invariant(*a.ctx, a.expr)); },
        "A"_a,
        "Vector invariant vec(A) (a vector; vec(a⊗b) = a×b).");
    m.def(
        "transpose",
        [](PyExpr const& a) -> PyExpr
        { return derive(a, make_transpose(*a.ctx, a.expr)); },
        "A"_a,
        "Transpose A^T (transpose(a⊗b) = b⊗a).");

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
                return render_latex(*e.expr, *map_ptr, e.ctx);
            IndexNameMap fresh;
            return render_latex(*e.expr, fresh, e.ctx);
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
        "_eval_eps_concrete",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::eval_eps_concrete(*e.ctx, e.expr)); },
        "expr"_a,
        "Evaluate a Levi-Civita symbol with concrete indices to its sign "
        "(+1/-1) or 0 on a repeat.");

    md.def(
        "_fold_arithmetic",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::fold_arithmetic(*e.ctx, e.expr)); },
        "expr"_a,
        "Constant-fold arithmetic operations on scalar literals.");

    md.def(
        "_expand_products",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::expand_products(*e.ctx, e.expr)); },
        "expr"_a,
        "Distribute TensorProduct over Sum/Difference (expand brackets).");

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

    md.def(
        "_contract_identity",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::contract_identity(*e.ctx, e.expr)); },
        "expr"_a,
        "Contract the identity tensor in a dot product: I·x -> x, x·I -> x.");

    md.def(
        "_distribute_contraction",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::distribute_contraction(*e.ctx, e.expr)); },
        "expr"_a,
        "Distribute a contraction (· or ×) over the adjacent leg of a tensor "
        "product: op(L, A⊗B) -> op(L,A)⊗B, op(A⊗B, R) -> A⊗op(B,R).");

    md.def(
        "_expand_double_dot",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::expand_double_dot(*e.ctx, e.expr)); },
        "expr"_a,
        "Expand a double contraction of dyads: (a⊗b):(c⊗d) -> (a·c)(b·d), "
        "(a⊗b)··(c⊗d) -> (a·d)(b·c); distributes over sums and binders.");

    md.def(
        "_expand_dyad_ops",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::expand_dyad_ops(*e.ctx, e.expr)); },
        "expr"_a,
        "Expand tr/vec/transpose on dyads: tr(a⊗b)->a·b, vec(a⊗b)->a×b, "
        "transpose(a⊗b)->b⊗a.");

    md.def(
        "_contract_eps_pair",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::contract_eps_pair(*e.ctx, e.expr)); },
        "expr"_a,
        "Contract Σ ε ε of two Levi-Civita symbols into the generalized "
        "Kronecker delta (e.g. Σ_i ε^{ijk} ε_{iml} → δ^j_m δ^k_l − δ^j_l δ^k_m).");

    md.def(
        "_unroll_sums_for",
        [](PyExpr const& e,
           std::vector<CountableIndex> const& indices) -> PyExpr
        { return derive(e, steps::unroll_sums_for(*e.ctx, e.expr, indices)); },
        "expr"_a,
        "indices"_a,
        "Unroll only ExplicitSum nodes whose index is in `indices`.");

    md.def(
        "_has_explicit_sum_for",
        [](PyExpr const& e, std::vector<CountableIndex> const& indices) -> bool
        { return steps::has_explicit_sum_for(e.expr, indices); },
        "expr"_a,
        "indices"_a,
        "Return True if `expr` contains an ExplicitSum for any index in `indices`.");

    md.def(
        "_fold_equal_addends",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::fold_equal_addends(*e.ctx, e.expr)); },
        "expr"_a,
        "Self-preparing fold: canonicalize, group identical addends "
        "(X + X → 2X, n*X + X → (n+1)*X, X − X → 0 even up to dummy-index "
        "renaming), then restore implicit-sum form.");

    md.def(
        "_fold_equal_addends_structural",
        [](PyExpr const& e) -> PyExpr {
            return derive(
                e, steps::fold_equal_addends_structural(*e.ctx, e.expr));
        },
        "expr"_a,
        "Bare structural fold: merge addends written identically only; does NOT "
        "rename dummies or normalize factor/sign order.");

    md.def(
        "_canonicalize",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::canonicalize(*e.ctx, e.expr)); },
        "expr"_a,
        "Rewrite into algebraic normal form (sorted, signed-coefficient, "
        "like terms combined; no distribution).");

    md.def(
        "_partial",
        [](PyExpr const& e, PyExpr const& coord) -> PyExpr
        { return derive(e, steps::partial(*e.ctx, e.expr, coord.expr)); },
        "expr"_a,
        "coord"_a,
        "Partial derivative ∂/∂coord of expr with respect to a coordinate "
        "variable (chain/product/quotient rules; constants → 0).");

    md.def(
        "_simplify_scalars",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::simplify_scalars(*e.ctx, e.expr)); },
        "expr"_a,
        "Targeted scalar-field simplifier: cos²+sin²→1, x⁰→1, x¹→x, and "
        "√(x²)→x for x≥0 (vibe 000069 M3).");

    md.def(
        "_implicitize",
        [](PyExpr const& e) -> PyExpr
        { return derive(e, steps::implicitize(*e.ctx, e.expr)); },
        "expr"_a,
        "Inverse of the implicit-sum convention: drop ExplicitSum binders whose "
        "index is repeated within a single multiplicative term, leaving the "
        "summation implicit (Einstein convention).");

    md.def(
        "_apply_identity",
        [](PyExpr const& target,
           PyExpr const& lhs,
           PyExpr const& rhs,
           std::string name) -> PyExpr
        {
            Identity id{std::move(name), lhs.expr, rhs.expr};
            return derive(target, apply_identity(*target.ctx, target.expr, id));
        },
        "expr"_a,
        "lhs"_a,
        "rhs"_a,
        "name"_a,
        "Apply identity lhs=rhs to the first matching subtree of expr; the "
        "result is canonical (== canonicalize(expr) if nothing matched).");

    md.def(
        "_saturate",
        [](PyExpr const& expr,
           std::vector<PyExpr> const& lhss,
           std::vector<PyExpr> const& rhss,
           int max_iterations) -> PyExpr
        {
            Context& ctx = *expr.ctx;
            nf::NfEGraph eg{ctx};
            auto const root = eg.add(expr.expr);
            std::vector<Identity> rules;
            rules.reserve(lhss.size());
            for (std::size_t i = 0; i < lhss.size(); ++i)
                rules.push_back(Identity{"", lhss[i].expr, rhss[i].expr});
            eg.saturate(rules, max_iterations);
            // Raise the cheapest extracted `Nf` back to the user-facing
            // implicit form (the same final shape `apply_identity` returns).
            auto const* nf = eg.extract(eg.find(root));
            auto const* out = steps::implicitize(
                ctx, steps::canonicalize(ctx, nf::raise(ctx, *nf)));
            return derive(expr, out);
        },
        "expr"_a,
        "lhss"_a,
        "rhss"_a,
        "max_iterations"_a,
        "Equality-saturate expr under the rules (lhs=rhs pairs) and return the "
        "cheapest extracted expression. All exprs must share one Context.");

    // Equality predicates (not steps).
    m.def(
        "_structural_eq",
        [](PyExpr const& a, PyExpr const& b) -> bool
        { return structural_eq(a.expr, b.expr); },
        "a"_a,
        "b"_a,
        "Deep structural equality of two expression trees.");

    m.def(
        "_algebraic_eq",
        [](PyExpr const& a, PyExpr const& b) -> bool
        { return algebraic_eq(*a.ctx, a.expr, b.expr); },
        "a"_a,
        "b"_a,
        "Algebraic equality in theory T0: structural_eq of the canonical forms.");

    // ------------------------------------------------------------------ //
    // basis submodule — the invariant/coordinate bridge (vibe 000049)
    // ------------------------------------------------------------------ //
    nb::module_ mb =
        m.def_submodule("basis", "Vector bases and coordinate systems.");

    nb::enum_<Variance>(mb, "Variance")
        .value("Covariant", Variance::Covariant)
        .value("Contravariant", Variance::Contravariant);

    nb::enum_<Handedness>(mb, "Handedness")
        .value("Right", Handedness::Right)
        .value("Left", Handedness::Left);

    nb::class_<PyBasis>(mb, "Basis")
        .def_prop_ro("realm", [](PyBasis const& b) { return b.basis.realm(); })
        .def_prop_ro(
            "space",
            [](PyBasis const& b) { return b.basis.space(); },
            nb::rv_policy::reference)
        .def_prop_ro("dim", [](PyBasis const& b) { return b.basis.dim(); })
        .def_prop_ro(
            "basis_id",
            [](PyBasis const& b) { return b.basis.basis_id(); },
            "The slot tag stamped on every index this basis emits (vibe "
            "000067); 0 means unregistered.")
        .def_prop_ro(
            "is_orthonormal",
            [](PyBasis const& b) { return b.basis.is_orthonormal(); })
        .def_prop_ro(
            "vector_symbol",
            [](PyBasis const& b) -> std::string
            { return std::string{b.basis.vector_symbol().v.view()}; })
        .def(
            "basis",
            [](PyBasis const& b, int i) -> PyExpr
            { return PyExpr{b.ctx_keep, b.ctx, b.basis.basis(i)}; },
            "i"_a,
            "The i-th covariant basis vector e_i (0-based).")
        .def(
            "cobasis",
            [](PyBasis const& b, int i) -> PyExpr
            { return PyExpr{b.ctx_keep, b.ctx, b.basis.cobasis(i)}; },
            "i"_a,
            "The i-th contravariant cobasis vector e^i (0-based).")
        .def(
            "covariant_vector",
            [](PyBasis const& b, CountableIndex idx) -> PyExpr {
                return PyExpr{
                    b.ctx_keep, b.ctx, b.basis.covariant_vector(*b.ctx, idx)};
            },
            "index"_a,
            "Symbolic covariant basis vector e_i carrying the given index.")
        .def(
            "contravariant_vector",
            [](PyBasis const& b, CountableIndex idx) -> PyExpr
            {
                return PyExpr{
                    b.ctx_keep,
                    b.ctx,
                    b.basis.contravariant_vector(*b.ctx, idx)};
            },
            "index"_a,
            "Symbolic contravariant basis vector e^i carrying the given index.")
        .def(
            "__repr__",
            [](PyBasis const& b) -> std::string
            {
                return "<Basis dim=" + std::to_string(b.basis.dim())
                       + " symbol='"
                       + std::string{b.basis.vector_symbol().v.view()} + "'>";
            });

    mb.def(
        "make_orthonormal_basis",
        [](std::vector<PyExpr> const& vectors,
           IndexSpace const* space,
           std::string const& symbol,
           Handedness handedness,
           std::vector<std::string> const& value_names,
           std::vector<std::string> const& vector_symbols,
           std::optional<std::string> const& label) -> PyBasis
        {
            // The basis vectors' Expr pointers live in their own context; that
            // is the context the basis (and its emissions) must use.
            Context* ctx =
                vectors.empty() ? &g_default_ctx : vectors.front().ctx;
            nb::object keep =
                vectors.empty() ? nb::none() : vectors.front().ctx_keep;
            std::vector<Expr const*> vs;
            vs.reserve(vectors.size());
            for (auto const& v: vectors)
                vs.push_back(v.expr);
            return PyBasis{
                keep,
                ctx,
                make_orthonormal_basis(
                    *ctx,
                    space,
                    std::move(vs),
                    make_tensor_name(symbol),
                    handedness,
                    make_basis_naming(value_names, vector_symbols, label))};
        },
        "vectors"_a,
        "space"_a,
        "symbol"_a = "e",
        "handedness"_a = Handedness::Right,
        "value_names"_a = std::vector<std::string>{},
        "vector_symbols"_a = std::vector<std::string>{},
        "label"_a = nb::none(),
        "Build an orthonormal basis from rank-1 vectors (cobasis = basis); "
        "handedness fixes √g = ±1.  value_names/vector_symbols/label control "
        "how concrete indices and vectors print (vibe 000067).");

    mb.def(
        "make_oblique_basis",
        [](std::vector<PyExpr> const& vectors,
           IndexSpace const* space,
           std::string const& symbol,
           std::vector<std::string> const& value_names,
           std::vector<std::string> const& vector_symbols,
           std::optional<std::string> const& label) -> PyBasis
        {
            Context* ctx =
                vectors.empty() ? &g_default_ctx : vectors.front().ctx;
            nb::object keep =
                vectors.empty() ? nb::none() : vectors.front().ctx_keep;
            std::vector<Expr const*> vs;
            vs.reserve(vectors.size());
            for (auto const& v: vectors)
                vs.push_back(v.expr);
            return PyBasis{
                keep,
                ctx,
                make_oblique_basis(
                    *ctx,
                    space,
                    std::move(vs),
                    make_tensor_name(symbol),
                    make_basis_naming(value_names, vector_symbols, label))};
        },
        "vectors"_a,
        "space"_a,
        "symbol"_a = "e",
        "value_names"_a = std::vector<std::string>{},
        "vector_symbols"_a = std::vector<std::string>{},
        "label"_a = nb::none(),
        "Build a 3D oblique basis from its covariant vectors; the contravariant "
        "cobasis is derived via the reciprocal (cross-product) formula.  "
        "value_names/vector_symbols/label control rendering (vibe 000067).");

    auto bind_cs =
        [&mb](char const* name, Basis (*fn)(Context&), char const* doc)
    {
        mb.def(
            name,
            [fn](nb::object ctx_arg) -> PyBasis
            {
                auto [ctx, keep] = resolve_ctx(ctx_arg);
                return PyBasis{keep, ctx, fn(*ctx)};
            },
            "ctx"_a = nb::none(),
            doc);
    };
    bind_cs("wcs", &wcs, "World Cartesian System: orthonormal frame i, j, k.");
    bind_cs(
        "cylindrical",
        &cylindrical,
        "Cylindrical (r, theta, z): orthonormal frame r, \\theta, z.");
    bind_cs(
        "spherical",
        &spherical,
        "Spherical (r, theta, phi): orthonormal frame r, \\theta, \\phi.");
    bind_cs(
        "polar_2d",
        &polar_2d,
        "2D polar (r, theta): orthonormal frame r, \\theta.");

    mb.def(
        "expand_in_basis",
        [](PyExpr const& e, PyBasis const& b, Variance v) -> PyExpr
        { return derive(e, expand_in_basis(*e.ctx, e.expr, b.basis, v)); },
        "expr"_a,
        "basis"_a,
        "variance"_a,
        "Expand each generic invariant tensor into its coordinate form in the "
        "basis (A -> A^{i...} (e_i ...)).");

    mb.def(
        "expand_in_basis",
        [](PyExpr const& e,
           PyBasis const& b,
           std::vector<Variance> variances) -> PyExpr
        {
            return derive(
                e,
                expand_in_basis(*e.ctx, e.expr, b.basis, std::move(variances)));
        },
        "expr"_a,
        "basis"_a,
        "variances"_a,
        "Expand with a per-slot variance list (one Variance per slot; a single "
        "entry broadcasts to every slot, otherwise the count must equal the "
        "tensor rank). Enables mixed coordinates like A^i_j.");

    mb.def(
        "simplify_basis_dot",
        [](PyExpr const& e, PyBasis const& b) -> PyExpr
        { return derive(e, simplify_basis_dot(*e.ctx, e.expr, b.basis)); },
        "expr"_a,
        "basis"_a,
        "Turn each dot of two basis vectors into a Kronecker delta "
        "((s e_i)·(t e_j) -> s t δ_{ij}).");

    mb.def(
        "simplify_basis_cross",
        [](PyExpr const& e, PyBasis const& b) -> PyExpr
        { return derive(e, simplify_basis_cross(*e.ctx, e.expr, b.basis)); },
        "expr"_a,
        "basis"_a,
        "Expand the cross of two covariant basis vectors via Levi-Civita: "
        "e_i × e_j -> √g ε_{ijk} e^k (orthonormal: ε_{ijk} e_k).");

    mb.def(
        "reassemble",
        [](PyExpr const& e, PyBasis const& b) -> PyExpr
        { return derive(e, reassemble(*e.ctx, e.expr, b.basis)); },
        "expr"_a,
        "basis"_a,
        "Fold a coordinate expansion back to its invariant (inverse of "
        "expand_in_basis); a no-op on anything that is not such an expansion.");

    mb.def(
        "reassemble_completeness",
        [](PyExpr const& e, PyBasis const& b) -> PyExpr
        { return derive(e, reassemble_completeness(*e.ctx, e.expr, b.basis)); },
        "expr"_a,
        "basis"_a,
        "Fold the resolution of identity Σ_i e_i⊗e^i = I where it is partially "
        "contracted: Σ_i (X·e_i) e_i -> X (and Σ_i (scalars) e_i⊗e_i -> "
        "(scalars) I).  Complements reassemble; a no-op otherwise.");

    // ------------------------------------------------------------------ //
    // chart submodule — coordinate mapping → curvilinear geometry (M4)
    // ------------------------------------------------------------------ //
    nb::module_ mc = m.def_submodule(
        "chart", "Coordinate charts and derived curvilinear geometry.");

    nb::class_<PyChart>(mc, "CoordinateChart")
        .def(
            "__init__",
            [](PyChart* self,
               PyBasis const& reference,
               std::vector<PyExpr> const& coords,
               std::vector<PyExpr> const& embedding)
            {
                std::vector<Expr const*> cs;
                cs.reserve(coords.size());
                for (auto const& c: coords)
                    cs.push_back(c.expr);
                std::vector<Expr const*> emb;
                emb.reserve(embedding.size());
                for (auto const& f: embedding)
                    emb.push_back(f.expr);
                new (self) PyChart{
                    reference.ctx_keep,
                    reference.ctx,
                    CoordinateChart{
                        reference.basis, std::move(cs), std::move(emb)}};
            },
            "reference"_a,
            "coords"_a,
            "embedding"_a,
            "A coordinate chart: an orthonormal reference Basis, the coordinate "
            "variables q^i (coordinate() atoms), and the Cartesian components "
            "x^a = f^a(q) of the position vector (one per reference direction).")
        .def(
            "radius_vector",
            [](PyChart const& c) -> PyExpr {
                return PyExpr{c.ctx_keep, c.ctx, radius_vector(*c.ctx, c.chart)};
            },
            "The position vector R = Σ_a f^a(q) u_a in the reference frame.")
        .def(
            "tangent_vector",
            [](PyChart const& c, int i) -> PyExpr {
                return PyExpr{
                    c.ctx_keep, c.ctx, tangent_vector(*c.ctx, c.chart, i)};
            },
            "i"_a,
            "The holonomic tangent basis vector g_i = ∂R/∂q^i.")
        .def(
            "metric_component",
            [](PyChart const& c, int i, int j) -> PyExpr {
                return PyExpr{
                    c.ctx_keep, c.ctx, metric_component(*c.ctx, c.chart, i, j)};
            },
            "i"_a,
            "j"_a,
            "The metric component g_ij = g_i·g_j, simplified to a scalar field.")
        .def(
            "scale_factor",
            [](PyChart const& c, int i) -> PyExpr {
                return PyExpr{
                    c.ctx_keep, c.ctx, scale_factor(*c.ctx, c.chart, i)};
            },
            "i"_a,
            "The scale factor h_i = √(g_ii), the positive root by convention.")
        .def(
            "physical_basis",
            [](PyChart const& c) -> PyBasis {
                return PyBasis{
                    c.ctx_keep, c.ctx, physical_basis(*c.ctx, c.chart)};
            },
            "The derived physical orthonormal frame e_i = g_i/h_i as a Basis.")
        .def(
            "basis_derivative",
            [](PyChart const& c, int i, int j) -> PyExpr {
                return PyExpr{
                    c.ctx_keep, c.ctx, basis_derivative(*c.ctx, c.chart, i, j)};
            },
            "i"_a,
            "j"_a,
            "The derivative ∂_{q^j} e_i of the i-th physical basis vector, as a "
            "vector in the reference frame (∂_φ e_r = e_φ, ∂_φ e_φ = −e_r).")
        .def(
            "connection_coefficients",
            [](PyChart const& c, int i, int j) -> std::vector<PyExpr>
            {
                auto cs = connection_coefficients(*c.ctx, c.chart, i, j);
                std::vector<PyExpr> out;
                out.reserve(cs.size());
                for (auto const* e: cs)
                    out.push_back(PyExpr{c.ctx_keep, c.ctx, e});
                return out;
            },
            "i"_a,
            "j"_a,
            "The physical-basis connection (rotation) coefficients γ^k_{ij} "
            "re-expressing ∂_{q^j} e_i = Σ_k γ^k_{ij} e_k in the local frame "
            "(one scalar per direction k).")
        .def(
            "gradient",
            [](PyChart const& c, PyExpr const& f) -> std::vector<PyExpr>
            {
                auto gs = gradient(*c.ctx, c.chart, f.expr);
                std::vector<PyExpr> out;
                out.reserve(gs.size());
                for (auto const* e: gs)
                    out.push_back(PyExpr{c.ctx_keep, c.ctx, e});
                return out;
            },
            "f"_a,
            "∇f = Σ_i (1/h_i)(∂_{q^i} f) e_i, as physical components.")
        .def(
            "divergence",
            [](PyChart const& c, std::vector<PyExpr> const& v) -> PyExpr
            {
                std::vector<Expr const*> vs;
                vs.reserve(v.size());
                for (auto const& e: v)
                    vs.push_back(e.expr);
                return PyExpr{
                    c.ctx_keep, c.ctx, divergence(*c.ctx, c.chart, vs)};
            },
            "v"_a,
            "∇·v from the physical components v = Σ_i v_i e_i, as a scalar.")
        .def(
            "laplacian",
            [](PyChart const& c, PyExpr const& f) -> PyExpr {
                return PyExpr{
                    c.ctx_keep, c.ctx, laplacian(*c.ctx, c.chart, f.expr)};
            },
            "f"_a,
            "Δf = div(grad f), as a scalar.")
        .def(
            "rot",
            [](PyChart const& c,
               std::vector<PyExpr> const& v) -> std::vector<PyExpr>
            {
                std::vector<Expr const*> vs;
                vs.reserve(v.size());
                for (auto const& e: v)
                    vs.push_back(e.expr);
                auto rs = rot(*c.ctx, c.chart, vs);
                std::vector<PyExpr> out;
                out.reserve(rs.size());
                for (auto const* e: rs)
                    out.push_back(PyExpr{c.ctx_keep, c.ctx, e});
                return out;
            },
            "v"_a,
            "∇×v from the physical components v, as physical components (3D).");
}
