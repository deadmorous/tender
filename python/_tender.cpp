// Python bindings for tender — Phase 12.
//
// All C++ objects returned to Python are owned by the module-level g_rl arena.
// Python holds non-owning references; the arena outlives the interpreter.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tender/tender.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace tender;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static ResourceList g_rl;

// Symbolic differential operator — nabla (∇).
struct Nabla
{
};

NB_MODULE(_tender, m)
{
    m.doc() = "tender — tensor algebra library for computational mechanics";

    // =======================================================================
    // Rational
    // =======================================================================
    nb::class_<Rational>(m, "Rational")
        .def(nb::init<int64_t>(), "n"_a)
        .def(nb::init<int64_t, int64_t>(), "num"_a, "den"_a)
        .def("__add__", [](Rational a, Rational b) { return a + b; })
        .def("__sub__", [](Rational a, Rational b) { return a - b; })
        .def("__mul__", [](Rational a, Rational b) { return a * b; })
        .def("__truediv__", [](Rational a, Rational b) { return a / b; })
        .def("__neg__", [](Rational a) { return -a; })
        .def("__eq__", [](Rational a, Rational b) { return a == b; })
        .def("__repr__", [](Rational r) -> std::string {
            if (r.den() == 1)
                return std::to_string(r.num());
            return std::to_string(r.num()) + "/" + std::to_string(r.den());
        })
        .def("to_double", &Rational::to_double)
        .def_prop_ro("num", &Rational::num)
        .def_prop_ro("den", &Rational::den)
        .def_prop_ro("is_zero", &Rational::is_zero)
        .def_prop_ro("is_integer", &Rational::is_integer);

    // =======================================================================
    // Expr base class
    // =======================================================================
    nb::class_<Expr>(m, "Expr")
        .def_prop_ro("rank", &Expr::rank)
        .def("latex", &Expr::latex)
        .def("python", &Expr::python)
        .def("_repr_latex_", [](Expr const* e) {
            return "$$" + e->latex() + "$$";
        })
        .def("__repr__", [](Expr const* e) { return e->latex(); })
        .def_prop_rw(
            "name",
            [](Expr const* e) -> std::string const& { return e->name(); },
            [](Expr* e, std::string n) { e->set_name(std::move(n)); })
        // Arithmetic operators
        .def("__add__", [](Expr* a, Expr* b) -> Expr* {
            return make_sum(g_rl, {a, b});
        }, nb::rv_policy::reference)
        .def("__radd__", [](Expr* b, Expr* a) -> Expr* {
            return make_sum(g_rl, {a, b});
        }, nb::rv_policy::reference)
        .def("__sub__", [](Expr* a, Expr* b) -> Expr* {
            return make_sum(g_rl, {a, make_scale(g_rl, Rational{-1}, b)});
        }, nb::rv_policy::reference)
        .def("__rsub__", [](Expr* b, Expr* a) -> Expr* {
            return make_sum(g_rl, {a, make_scale(g_rl, Rational{-1}, b)});
        }, nb::rv_policy::reference)
        .def("__neg__", [](Expr* a) -> Expr* {
            return make_scale(g_rl, Rational{-1}, a);
        }, nb::rv_policy::reference)
        .def("__mul__", [](Expr* a, Expr* b) -> Expr* {
            return make_tensor_product(g_rl, a, b);
        }, nb::rv_policy::reference)
        .def("__mul__", [](Expr* a, int n) -> Expr* {
            return make_scale(g_rl, Rational{n}, a);
        }, nb::rv_policy::reference)
        .def("__mul__", [](Expr* a, Rational r) -> Expr* {
            return make_scale(g_rl, r, a);
        }, nb::rv_policy::reference)
        .def("__rmul__", [](Expr* b, int n) -> Expr* {
            return make_scale(g_rl, Rational{n}, b);
        }, nb::rv_policy::reference)
        .def("__rmul__", [](Expr* b, Rational r) -> Expr* {
            return make_scale(g_rl, r, b);
        }, nb::rv_policy::reference)
        .def("__truediv__", [](Expr* a, int n) -> Expr* {
            if (n == 0)
                throw std::invalid_argument("division by zero");
            return make_scale(g_rl, Rational{1, n}, a);
        }, nb::rv_policy::reference)
        // @ — single contraction
        .def("__matmul__", [](Expr* a, Expr* b) -> Expr* {
            return make_contract(g_rl, a, b);
        }, nb::rv_policy::reference)
        // // — double contraction (Frobenius)
        .def("__floordiv__", [](Expr* a, Expr* b) -> Expr* {
            return make_double_contract(g_rl, a, b);
        }, nb::rv_policy::reference)
        // ** — double contraction reversed OR scalar power
        .def("__pow__", [](Expr* a, Expr* b) -> Expr* {
            return make_double_contract_reversed(g_rl, a, b);
        }, nb::rv_policy::reference)
        .def("__pow__", [](Expr* a, int n) -> Expr* {
            return make_pow(g_rl, a, Rational{n});
        }, nb::rv_policy::reference)
        // % — cross product (chaining raises)
        .def("__mod__", [](Expr* a, Expr* b) -> Expr* {
            if (dynamic_cast<CrossProduct*>(a))
                throw std::invalid_argument(
                    "% (cross product) is not associative — "
                    "parenthesise explicitly or use cross(a, cross(b, c))");
            return make_cross_product(g_rl, a, b);
        }, nb::rv_policy::reference);

    // =======================================================================
    // Concrete Expr subclasses
    // =======================================================================
    nb::class_<RationalConst, Expr>(m, "RationalConst")
        .def_prop_ro("value", &RationalConst::value);

    nb::class_<NamedConst, Expr>(m, "NamedConst")
        .def_prop_ro("symbol", &NamedConst::symbol);

    nb::class_<SymbolicVar, Expr>(m, "SymbolicVar")
        .def_prop_ro("symbol", &SymbolicVar::symbol);

    nb::class_<Parameter, SymbolicVar>(m, "Parameter")
        .def_prop_ro("symbol", &Parameter::symbol);

    nb::class_<Sum, Expr>(m, "Sum").def_prop_ro("terms", [](Sum const* s) {
        return std::vector<Expr*>(s->terms().begin(), s->terms().end());
    });

    nb::class_<Scale, Expr>(m, "Scale")
        .def_prop_ro("coeff", &Scale::coeff)
        .def_prop_ro("expr", &Scale::expr, nb::rv_policy::reference);

    nb::class_<TensorProduct, Expr>(m, "TensorProduct")
        .def_prop_ro("lhs", &TensorProduct::lhs, nb::rv_policy::reference)
        .def_prop_ro("rhs", &TensorProduct::rhs, nb::rv_policy::reference);

    nb::class_<IdentityTensor, Expr>(m, "IdentityTensor");
    nb::class_<LeviCivitaTensor, Expr>(m, "LeviCivitaTensor");

    nb::class_<Trace, Expr>(m, "Trace")
        .def_prop_ro("arg", &Trace::arg, nb::rv_policy::reference);

    nb::class_<Contract, Expr>(m, "Contract")
        .def_prop_ro("lhs", &Contract::lhs, nb::rv_policy::reference)
        .def_prop_ro("rhs", &Contract::rhs, nb::rv_policy::reference);

    nb::class_<DoubleContract, Expr>(m, "DoubleContract")
        .def_prop_ro("lhs", &DoubleContract::lhs, nb::rv_policy::reference)
        .def_prop_ro("rhs", &DoubleContract::rhs, nb::rv_policy::reference);

    nb::class_<DoubleContractReversed, Expr>(m, "DoubleContractReversed")
        .def_prop_ro(
            "lhs", &DoubleContractReversed::lhs, nb::rv_policy::reference)
        .def_prop_ro(
            "rhs", &DoubleContractReversed::rhs, nb::rv_policy::reference);

    nb::class_<CrossProduct, Expr>(m, "CrossProduct")
        .def_prop_ro("lhs", &CrossProduct::lhs, nb::rv_policy::reference)
        .def_prop_ro("rhs", &CrossProduct::rhs, nb::rv_policy::reference);

    nb::class_<NamedTensor, Expr>(m, "NamedTensor")
        .def_prop_ro("symbol", &NamedTensor::symbol);

    nb::class_<Product, Expr>(m, "Product")
        .def_prop_ro("lhs", &Product::lhs, nb::rv_policy::reference)
        .def_prop_ro("rhs", &Product::rhs, nb::rv_policy::reference);

    nb::class_<FunctionApply, Expr>(m, "FunctionApply")
        .def_prop_ro("arg", &FunctionApply::arg, nb::rv_policy::reference);

    nb::class_<Pow, Expr>(m, "Pow")
        .def_prop_ro("base", &Pow::base, nb::rv_policy::reference)
        .def_prop_ro("exponent", &Pow::exponent);

    nb::class_<MaterialDeriv, Expr>(m, "MaterialDeriv")
        .def_prop_ro(
            "velocity", &MaterialDeriv::velocity, nb::rv_policy::reference)
        .def_prop_ro("field", &MaterialDeriv::field, nb::rv_policy::reference);

    nb::class_<PatternVar, Expr>(m, "PatternVar")
        .def_prop_ro("symbol", &PatternVar::symbol)
        .def("constrain_rank", &PatternVar::constrain_rank, nb::rv_policy::reference)
        .def("constrain_symmetric", &PatternVar::constrain_symmetric, nb::rv_policy::reference)
        .def("constrain_skew_symmetric", &PatternVar::constrain_skew_symmetric, nb::rv_policy::reference);

    // Integral-layer expression nodes
    nb::class_<Gradient, Expr>(m, "Gradient")
        .def_prop_ro("arg", &Gradient::arg, nb::rv_policy::reference);

    nb::class_<Divergence, Expr>(m, "Divergence")
        .def_prop_ro("arg", &Divergence::arg, nb::rv_policy::reference);

    nb::class_<Rotor, Expr>(m, "Rotor")
        .def_prop_ro("arg", &Rotor::arg, nb::rv_policy::reference);

    nb::class_<Integral, Expr>(m, "Integral")
        .def_prop_ro("integrand", &Integral::integrand, nb::rv_policy::reference)
        .def_prop_ro("domain", &Integral::domain, nb::rv_policy::reference);

    // =======================================================================
    // Domain types
    // =======================================================================
    nb::class_<Domain>(m, "Domain")
        .def_prop_ro(
            "name", [](Domain const* d) -> std::string const& {
                return d->name();
            })
        .def("measure_latex", [](Domain const* d) {
            return d->measure_latex();
        });

    nb::class_<SurfaceDomain, Domain>(m, "SurfaceDomain")
        .def_prop_ro(
            "normal", &SurfaceDomain::normal, nb::rv_policy::reference);

    nb::class_<VolumeDomain, Domain>(m, "VolumeDomain")
        .def_prop_ro(
            "outward_normal",
            &VolumeDomain::outward_normal,
            nb::rv_policy::reference)
        .def_prop_ro(
            "surface_boundary",
            &VolumeDomain::surface_boundary,
            nb::rv_policy::reference);

    // =======================================================================
    // CoordSystem
    // =======================================================================
    nb::class_<CoordSystem>(m, "CoordSystem")
        .def_prop_ro("dim", &CoordSystem::dim)
        .def_prop_ro("is_orthonormal", &CoordSystem::is_orthonormal)
        .def("basis", [](CoordSystem const* cs, int i) -> Expr* {
            return cs->basis(i);
        }, nb::rv_policy::reference, "i"_a)
        .def("cobasis", [](CoordSystem const* cs, int i) -> Expr* {
            return cs->cobasis(i);
        }, nb::rv_policy::reference, "i"_a)
        .def("coord", [](CoordSystem const* cs, int i) -> Parameter* {
            return cs->coord(i);
        }, nb::rv_policy::reference, "i"_a)
        .def("metric", [](CoordSystem const* cs, int i, int j) -> Expr* {
            return cs->metric(i, j);
        }, nb::rv_policy::reference, "i"_a, "j"_a)
        .def_prop_ro("basis_vectors", [](CoordSystem const* cs) {
            std::vector<Expr*> v;
            for (int i = 0; i < cs->dim(); ++i)
                v.push_back(cs->basis(i));
            return v;
        })
        .def_prop_ro("i", [](CoordSystem const* cs) -> Expr* {
            return cs->basis(0);
        }, nb::rv_policy::reference)
        .def_prop_ro("j", [](CoordSystem const* cs) -> Expr* {
            return cs->basis(1);
        }, nb::rv_policy::reference)
        .def_prop_ro("k", [](CoordSystem const* cs) -> Expr* {
            return cs->basis(2);
        }, nb::rv_policy::reference);

    // =======================================================================
    // State, DerivationStep, Derivation
    // =======================================================================
    nb::class_<State>(m, "State")
        .def(nb::init<Expr*>(), "expr"_a)
        .def_prop_ro("expr", &State::expr, nb::rv_policy::reference)
        .def_prop_ro(
            "label",
            [](State const& s) -> std::string const& { return s.label(); })
        .def("latex", &State::latex)
        .def("_repr_latex_", [](State const& s) {
            return "$$" + s.expr()->latex() + "$$";
        })
        .def("__repr__", [](State const& s) { return s.latex(); });

    nb::class_<DerivationStep>(m, "DerivationStep")
        .def_prop_ro(
            "name",
            [](DerivationStep const& s) -> std::string const& {
                return s.name();
            })
        .def("apply", [](DerivationStep const& step, State const& s) {
            return step.apply(g_rl, s);
        }, "state"_a);

    nb::class_<Derivation>(m, "Derivation")
        .def(nb::init<std::vector<DerivationStep>>(), "steps"_a)
        .def("apply", [](Derivation const& d, State const& initial) {
            return d.apply(g_rl, initial);
        }, "initial"_a)
        .def("__add__", [](Derivation const& a, Derivation const& b) {
            return a + b;
        });

    // =======================================================================
    // Identity
    // =======================================================================
    nb::class_<Identity>(m, "Identity")
        .def(nb::init<std::string, Expr*, Expr*>(), "name"_a, "lhs"_a, "rhs"_a)
        .def_prop_ro(
            "name",
            [](Identity const& id) -> std::string const& { return id.name(); })
        .def_prop_ro("lhs", &Identity::lhs, nb::rv_policy::reference)
        .def_prop_ro("rhs", &Identity::rhs, nb::rv_policy::reference)
        .def_static(
            "from_derivation",
            [](std::string name, std::vector<State> const& history) {
                return Identity::from_derivation(std::move(name), history);
            },
            "name"_a,
            "history"_a);

    // =======================================================================
    // Nabla object  (∇)
    // =======================================================================
    nb::class_<Nabla>(m, "Nabla")
        .def(nb::init<>())
        .def("__matmul__", [](Nabla const&, Expr* v) -> Expr* {
            return make_divergence(g_rl, v);
        }, nb::rv_policy::reference, "v"_a)
        .def("__mul__", [](Nabla const&, Expr* v) -> Expr* {
            return make_gradient(g_rl, v);
        }, nb::rv_policy::reference, "v"_a)
        .def("__mod__", [](Nabla const&, Expr* v) -> Expr* {
            return make_rotor(g_rl, v);
        }, nb::rv_policy::reference, "v"_a)
        .def("__repr__", [](Nabla const&) { return "nabla"; });

    // =======================================================================
    // Free factory functions
    // =======================================================================

    // --- named tensors ---
    m.def("tensor", [](std::string name, int rank) -> Expr* {
        return make_named_tensor(g_rl, std::move(name), rank, {});
    }, nb::rv_policy::reference, "name"_a, "rank"_a = 0);

    m.def("scalar", [](std::string name) -> Expr* {
        return make_named_tensor(g_rl, std::move(name), 0, {});
    }, nb::rv_policy::reference, "name"_a);

    m.def("parameter", [](std::string name) -> Parameter* {
        return make_parameter(g_rl, std::move(name));
    }, nb::rv_policy::reference, "name"_a);

    m.def("named", [](std::string n, Expr* e) -> Expr* {
        return named(std::move(n), e);
    }, nb::rv_policy::reference, "name"_a, "expr"_a);

    m.def("rational", [](int64_t n) -> Expr* {
        return make_rational(g_rl, Rational{n});
    }, nb::rv_policy::reference, "n"_a);

    m.def("rational", [](int64_t num, int64_t den) -> Expr* {
        return make_rational(g_rl, Rational{num, den});
    }, nb::rv_policy::reference, "num"_a, "den"_a);

    m.def("make_pattern_var", [](std::string sym) -> PatternVar* {
        return make_pattern_var(g_rl, std::move(sym));
    }, nb::rv_policy::reference, "symbol"_a);

    // --- algebraic operations ---
    m.def("tp", [](Expr* a, Expr* b) -> Expr* {
        return make_tensor_product(g_rl, a, b);
    }, nb::rv_policy::reference, "a"_a, "b"_a);

    m.def("dot", [](Expr* a, Expr* b) -> Expr* {
        return make_contract(g_rl, a, b);
    }, nb::rv_policy::reference, "a"_a, "b"_a);

    m.def("ddot", [](Expr* a, Expr* b) -> Expr* {
        return make_double_contract(g_rl, a, b);
    }, nb::rv_policy::reference, "a"_a, "b"_a);

    m.def("ddot2", [](Expr* a, Expr* b) -> Expr* {
        return make_double_contract_reversed(g_rl, a, b);
    }, nb::rv_policy::reference, "a"_a, "b"_a);

    m.def("cross", [](Expr* a, Expr* b) -> Expr* {
        return make_cross_product(g_rl, a, b);
    }, nb::rv_policy::reference, "a"_a, "b"_a);

    m.def("trace", [](Expr* a) -> Expr* {
        return make_trace(g_rl, a);
    }, nb::rv_policy::reference, "a"_a);

    // --- scalar functions ---
    m.def("exp",  [](Expr* a) -> Expr* { return make_exp(g_rl, a); }, nb::rv_policy::reference, "a"_a);
    m.def("log",  [](Expr* a) -> Expr* { return make_log(g_rl, a); }, nb::rv_policy::reference, "a"_a);
    m.def("sin",  [](Expr* a) -> Expr* { return make_sin(g_rl, a); }, nb::rv_policy::reference, "a"_a);
    m.def("cos",  [](Expr* a) -> Expr* { return make_cos(g_rl, a); }, nb::rv_policy::reference, "a"_a);
    m.def("tan",  [](Expr* a) -> Expr* { return make_tan(g_rl, a); }, nb::rv_policy::reference, "a"_a);
    m.def("asin", [](Expr* a) -> Expr* { return make_asin(g_rl, a); }, nb::rv_policy::reference, "a"_a);
    m.def("acos", [](Expr* a) -> Expr* { return make_acos(g_rl, a); }, nb::rv_policy::reference, "a"_a);
    m.def("atan", [](Expr* a) -> Expr* { return make_atan(g_rl, a); }, nb::rv_policy::reference, "a"_a);
    m.def("atan2", [](Expr* y, Expr* x) -> Expr* {
        return make_atan2(g_rl, y, x);
    }, nb::rv_policy::reference, "y"_a, "x"_a);
    m.def("sinh", [](Expr* a) -> Expr* { return make_sinh(g_rl, a); }, nb::rv_policy::reference, "a"_a);
    m.def("cosh", [](Expr* a) -> Expr* { return make_cosh(g_rl, a); }, nb::rv_policy::reference, "a"_a);
    m.def("tanh", [](Expr* a) -> Expr* { return make_tanh(g_rl, a); }, nb::rv_policy::reference, "a"_a);
    m.def("sqrt", [](Expr* a) -> Expr* { return make_sqrt(g_rl, a); }, nb::rv_policy::reference, "a"_a);
    m.def("pow",  [](Expr* base, int n) -> Expr* {
        return make_pow(g_rl, base, Rational{n});
    }, nb::rv_policy::reference, "base"_a, "n"_a);

    // --- differentiation ---
    m.def("deriv", [](Parameter const* p, Expr* e) -> Expr* {
        return deriv(g_rl, p, e);
    }, nb::rv_policy::reference, "param"_a, "expr"_a);

    m.def("dt", [](Expr* e) -> Expr* {
        return dt(g_rl, e);
    }, nb::rv_policy::reference, "expr"_a);

    m.def("ddt", [](Expr* e) -> Expr* {
        return ddt(g_rl, e);
    }, nb::rv_policy::reference, "expr"_a);

    // --- coord system ---
    m.def("grad", [](Expr* f, CoordSystem const& cs) -> Expr* {
        return grad(g_rl, f, cs);
    }, nb::rv_policy::reference, "f"_a, "cs"_a);

    m.def("make_direct_basis_cs",
        [](Expr* e1, Expr* e2, Expr* e3) {
            return make_direct_basis_cs(e1, e2, e3);
        }, "e1"_a, "e2"_a, "e3"_a);

    // --- differential operators (symbolic, not coord-system-aware) ---
    m.def("gradient", [](Expr* a) -> Expr* {
        return make_gradient(g_rl, a);
    }, nb::rv_policy::reference, "a"_a);

    m.def("divergence", [](Expr* a) -> Expr* {
        return make_divergence(g_rl, a);
    }, nb::rv_policy::reference, "a"_a);

    m.def("rot", [](Expr* a) -> Expr* {
        return make_rotor(g_rl, a);
    }, nb::rv_policy::reference, "a"_a);

    // --- integral / domain ---
    m.def("make_surface_domain",
        [](std::string name, Expr* normal) -> SurfaceDomain* {
            return make_surface_domain(g_rl, std::move(name), normal);
        }, nb::rv_policy::reference, "name"_a, "normal"_a);

    m.def("make_volume_domain",
        [](std::string name, Expr* outward_normal) -> VolumeDomain* {
            return make_volume_domain(g_rl, std::move(name), outward_normal);
        }, nb::rv_policy::reference, "name"_a, "outward_normal"_a);

    m.def("integral", [](Domain* domain, Expr* integrand) -> Expr* {
        return make_integral(g_rl, domain, integrand);
    }, nb::rv_policy::reference, "domain"_a, "integrand"_a);

    // --- show ---
    m.def("show", [](std::vector<State> const& history) {
        return show(history);
    }, "history"_a);

    m.def("show_final", [](std::vector<State> const& history) {
        return show_final(history);
    }, "history"_a);

    // --- built-in step factories ---
    m.def("simplify_identity_step", &simplify_identity_step);
    m.def("expand_step", &expand_step);
    m.def("expand_poly_step", &expand_poly_step);

    m.def("substitute_step", [](Expr* what, Expr* with_what) {
        return substitute_step(what, with_what);
    }, "what"_a, "with_what"_a);

    m.def("diff_step", [](Parameter const* p) {
        return diff_step(p);
    }, "param"_a);

    m.def("apply_integration_by_parts_step", [](VolumeDomain* domain) {
        return apply_integration_by_parts_step(domain);
    }, "domain"_a);

    m.def("apply_divergence_theorem_step", [](VolumeDomain* domain) {
        return apply_divergence_theorem_step(domain);
    }, "domain"_a);

    m.def("localize_step", [](Domain* domain) {
        return localize_step(domain);
    }, "domain"_a);

    m.def("apply_identity",
        [](Identity const& id, nb::dict mapping) {
            PatternMapping pm;
            for (auto [k, v] : mapping)
                pm[nb::cast<PatternVar*>(k)] = nb::cast<Expr*>(v);
            return apply_identity(id, pm);
        }, "identity"_a, "mapping"_a);

    // =======================================================================
    // Singleton getters — called once from __init__.py to create module attrs
    // =======================================================================
    m.def("_identity_singleton", []() -> Expr* {
        static Expr* e = make_identity(g_rl);
        return e;
    }, nb::rv_policy::reference);

    m.def("_levi_civita_singleton", []() -> Expr* {
        static Expr* e = make_levi_civita(g_rl);
        return e;
    }, nb::rv_policy::reference);

    m.def("_time_param_singleton", []() -> Parameter* {
        return const_cast<Parameter*>(time_parameter());
    }, nb::rv_policy::reference);

    m.def("_wcs_singleton", []() -> CoordSystem* {
        return const_cast<CoordSystem*>(&tender::wcs());
    }, nb::rv_policy::reference);

    m.def("_cylindrical_cs_singleton", []() -> CoordSystem* {
        return const_cast<CoordSystem*>(&tender::cylindrical_cs());
    }, nb::rv_policy::reference);

    m.def("_spherical_cs_singleton", []() -> CoordSystem* {
        return const_cast<CoordSystem*>(&tender::spherical_cs());
    }, nb::rv_policy::reference);
}
