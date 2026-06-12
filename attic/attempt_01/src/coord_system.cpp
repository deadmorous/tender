#include <tender/coord_system.hpp>

#include <array>
#include <stdexcept>
#include <vector>

namespace tender
{

// ===========================================================================
// WCS — World Cartesian System
// ===========================================================================

class WCSImpl : public CoordSystem
{
public:
    WCSImpl()
    {
        coords_[0] = make_parameter(rl_, "x");
        coords_[1] = make_parameter(rl_, "y");
        coords_[2] = make_parameter(rl_, "z");

        auto mk_bv = [this](char const* sym, char const* name) -> Expr*
        {
            auto* e = make_named_tensor(rl_, sym, 1, {});
            e->set_name(name);
            return e;
        };
        basis_[0] = mk_bv("i", "\\mathbf{i}");
        basis_[1] = mk_bv("j", "\\mathbf{j}");
        basis_[2] = mk_bv("k", "\\mathbf{k}");

        cobasis_ = basis_; // orthonormal: g^i = g_i

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                metric_[i][j] = make_rational(rl_, Rational{i == j ? 1 : 0});
    }

    auto dim() const noexcept -> int override
    {
        return 3;
    }
    auto coord(int i) const -> Parameter* override
    {
        return coords_[i];
    }
    auto basis(int i) const -> Expr* override
    {
        return basis_[i];
    }
    auto cobasis(int i) const -> Expr* override
    {
        return cobasis_[i];
    }
    auto metric(int i, int j) const -> Expr* override
    {
        return metric_[i][j];
    }
    auto is_orthonormal() const noexcept -> bool override
    {
        return true;
    }

private:
    ResourceList rl_;
    std::array<Parameter*, 3> coords_{};
    std::array<Expr*, 3> basis_{};
    std::array<Expr*, 3> cobasis_{};
    std::array<std::array<Expr*, 3>, 3> metric_{};
};

auto wcs() -> CoordSystem const&
{
    static WCSImpl instance;
    return instance;
}

// ===========================================================================
// DirectBasisCS
// ===========================================================================

class DirectBasisCSImpl : public CoordSystem
{
public:
    DirectBasisCSImpl(Expr* e1, Expr* e2, Expr* e3)
    {
        basis_[0] = e1;
        basis_[1] = e2;
        basis_[2] = e3;

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                metric_[i][j] = make_contract(rl_, basis_[i], basis_[j]);

        // Volume V = e1 · (e2 × e3)
        auto* e2xe3 = make_cross_product(rl_, e2, e3);
        auto* e3xe1 = make_cross_product(rl_, e3, e1);
        auto* e1xe2 = make_cross_product(rl_, e1, e2);
        auto* vol = make_contract(rl_, e1, e2xe3);
        auto* inv_vol = make_pow(rl_, vol, Rational{-1});

        // Cobasis: g^1 = (e2×e3)/V, g^2 = (e3×e1)/V, g^3 = (e1×e2)/V
        cobasis_[0] = make_tensor_product(rl_, inv_vol, e2xe3);
        cobasis_[1] = make_tensor_product(rl_, inv_vol, e3xe1);
        cobasis_[2] = make_tensor_product(rl_, inv_vol, e1xe2);
    }

    auto dim() const noexcept -> int override
    {
        return 3;
    }
    auto coord(int) const -> Parameter* override
    {
        return nullptr;
    }
    auto basis(int i) const -> Expr* override
    {
        return basis_[i];
    }
    auto cobasis(int i) const -> Expr* override
    {
        return cobasis_[i];
    }
    auto metric(int i, int j) const -> Expr* override
    {
        return metric_[i][j];
    }
    auto is_orthonormal() const noexcept -> bool override
    {
        return false;
    }

private:
    ResourceList rl_;
    std::array<Expr*, 3> basis_{};
    std::array<Expr*, 3> cobasis_{};
    std::array<std::array<Expr*, 3>, 3> metric_{};
};

auto make_direct_basis_cs(Expr* e1, Expr* e2, Expr* e3)
    -> std::unique_ptr<CoordSystem>
{
    return std::make_unique<DirectBasisCSImpl>(e1, e2, e3);
}

// ===========================================================================
// CylindricalCS
// ===========================================================================

class CylindricalCSImpl : public CoordSystem
{
public:
    CylindricalCSImpl()
    {
        auto* r = make_parameter(rl_, "r");
        auto* theta = make_parameter(rl_, "theta");
        auto* z = make_parameter(rl_, "z");
        coords_[0] = r;
        coords_[1] = theta;
        coords_[2] = z;

        auto mk_bv = [this](char const* sym, char const* name) -> Expr*
        {
            auto* e = make_named_tensor(rl_, sym, 1, {});
            e->set_name(name);
            return e;
        };
        auto* e_r = mk_bv("e_r", "\\mathbf{e}_r");
        auto* e_theta = mk_bv("e_theta", "\\mathbf{e}_\\theta");
        auto* e_z = mk_bv("e_z", "\\mathbf{e}_z");

        // Covariant basis: g_r = e_r, g_theta = r e_theta, g_z = e_z
        basis_[0] = e_r;
        basis_[1] = make_tensor_product(rl_, r, e_theta);
        basis_[2] = e_z;

        // Metric: diag(1, r^2, 1)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                metric_[i][j] = make_rational(rl_, Rational{0});
        metric_[0][0] = make_rational(rl_, Rational{1});
        metric_[1][1] = make_pow(rl_, r, Rational{2});
        metric_[2][2] = make_rational(rl_, Rational{1});

        // Cobasis: g^r = e_r, g^theta = (1/r) e_theta, g^z = e_z
        auto* inv_r = make_pow(rl_, r, Rational{-1});
        cobasis_[0] = e_r;
        cobasis_[1] = make_tensor_product(rl_, inv_r, e_theta);
        cobasis_[2] = e_z;
    }

    auto dim() const noexcept -> int override
    {
        return 3;
    }
    auto coord(int i) const -> Parameter* override
    {
        return coords_[i];
    }
    auto basis(int i) const -> Expr* override
    {
        return basis_[i];
    }
    auto cobasis(int i) const -> Expr* override
    {
        return cobasis_[i];
    }
    auto metric(int i, int j) const -> Expr* override
    {
        return metric_[i][j];
    }
    auto is_orthonormal() const noexcept -> bool override
    {
        return false;
    }

private:
    ResourceList rl_;
    std::array<Parameter*, 3> coords_{};
    std::array<Expr*, 3> basis_{};
    std::array<Expr*, 3> cobasis_{};
    std::array<std::array<Expr*, 3>, 3> metric_{};
};

auto cylindrical_cs() -> CoordSystem const&
{
    static CylindricalCSImpl instance;
    return instance;
}

// ===========================================================================
// SphericalCS
// ===========================================================================

class SphericalCSImpl : public CoordSystem
{
public:
    SphericalCSImpl()
    {
        auto* r = make_parameter(rl_, "r");
        auto* theta = make_parameter(rl_, "theta");
        auto* phi = make_parameter(rl_, "phi");
        coords_[0] = r;
        coords_[1] = theta;
        coords_[2] = phi;

        auto mk_bv = [this](char const* sym, char const* name) -> Expr*
        {
            auto* e = make_named_tensor(rl_, sym, 1, {});
            e->set_name(name);
            return e;
        };
        auto* e_r = mk_bv("e_r", "\\mathbf{e}_r");
        auto* e_theta = mk_bv("e_th", "\\mathbf{e}_\\theta");
        auto* e_phi = mk_bv("e_phi", "\\mathbf{e}_\\phi");

        // Covariant basis: g_r = e_r, g_theta = r e_theta,
        //                  g_phi = r sin(theta) e_phi
        auto* sin_theta = make_sin(rl_, theta);
        basis_[0] = e_r;
        basis_[1] = make_tensor_product(rl_, r, e_theta);
        basis_[2] =
            make_tensor_product(rl_, make_product(rl_, r, sin_theta), e_phi);

        // Metric: diag(1, r^2, r^2 sin^2(theta))
        auto* r_sq = make_pow(rl_, r, Rational{2});
        auto* s_sq = make_pow(rl_, sin_theta, Rational{2});
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                metric_[i][j] = make_rational(rl_, Rational{0});
        metric_[0][0] = make_rational(rl_, Rational{1});
        metric_[1][1] = r_sq;
        metric_[2][2] = make_product(rl_, r_sq, s_sq);

        // Cobasis: g^r = e_r, g^theta = (1/r) e_theta,
        //          g^phi = (1/(r sin(theta))) e_phi
        auto* inv_r = make_pow(rl_, r, Rational{-1});
        auto* inv_rs = make_product(
            rl_,
            make_pow(rl_, r, Rational{-1}),
            make_pow(rl_, sin_theta, Rational{-1}));
        cobasis_[0] = e_r;
        cobasis_[1] = make_tensor_product(rl_, inv_r, e_theta);
        cobasis_[2] = make_tensor_product(rl_, inv_rs, e_phi);
    }

    auto dim() const noexcept -> int override
    {
        return 3;
    }
    auto coord(int i) const -> Parameter* override
    {
        return coords_[i];
    }
    auto basis(int i) const -> Expr* override
    {
        return basis_[i];
    }
    auto cobasis(int i) const -> Expr* override
    {
        return cobasis_[i];
    }
    auto metric(int i, int j) const -> Expr* override
    {
        return metric_[i][j];
    }
    auto is_orthonormal() const noexcept -> bool override
    {
        return false;
    }

private:
    ResourceList rl_;
    std::array<Parameter*, 3> coords_{};
    std::array<Expr*, 3> basis_{};
    std::array<Expr*, 3> cobasis_{};
    std::array<std::array<Expr*, 3>, 3> metric_{};
};

auto spherical_cs() -> CoordSystem const&
{
    static SphericalCSImpl instance;
    return instance;
}

// ===========================================================================
// grad
// ===========================================================================

auto grad(ResourceList& rl, Expr* f, CoordSystem const& cs) -> Expr*
{
    std::vector<Expr*> terms;
    terms.reserve(cs.dim());
    for (int i = 0; i < cs.dim(); ++i)
    {
        Parameter* q = cs.coord(i);
        if (!q)
            throw std::invalid_argument(
                "grad: coordinate system has no explicit coordinates");
        auto* df_dq = deriv(rl, q, f);
        auto* g_i = cs.cobasis(i);
        terms.push_back(make_tensor_product(rl, df_dq, g_i));
    }
    return make_sum(rl, std::move(terms));
}

} // namespace tender
