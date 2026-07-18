#ifndef KINETIC_COMPOSITION_CPU_INTERPOLATOR_HPP
#define KINETIC_COMPOSITION_CPU_INTERPOLATOR_HPP

#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include "evaluator_iface.h"

/**
 * @brief Nested-kinetics composition wrapper ("nested multilinear interpolator").
 *
 * Wraps an inner operator-set interpolator whose KIN block slots carry SMOOTH FIELDS
 * instead of the sharp kinetic-rate operators:
 *
 *   slot kin_op_start + m              : SRb_m = min(SR_m, SR_threshold)   (m = 0..n_min-1)
 *   slot kin_op_start + n_min + m      : A_m   = sum_j k_arr_j(T) * act_j^n_j
 *   slot kin_op_start + 2*n_min        : rho_t = 1 / vol_sum  (total molar density, kmol/m3)
 *   remaining KIN slots                : unused (zeroed on output)
 *
 * After the inner interpolation, the kinetic rate of each mineral m is composed
 * analytically at the query state x:
 *
 *   rate_m(x) = c_m * z_m * rho_t(x) * A_m(x) * k_aff(SRb_m(x)),
 *   k_aff(s)  = (1 - s^p_m)^q_m           (p_m = q_m = 1: k_aff = 1 - s, exact linear)
 *
 * where z_m = x[mineral_axis_m] is the EXACT state value of the mineral's solid
 * composition axis (rate == 0 at z_m == 0 by construction), and c_m collapses the
 * surface-area/unit constants: c_m = -s_init * 1000 * 86.4 (kmol/day/m3 units).
 * The identity sat_m * rho_s,m = z_m * rho_t (with rho_t = 1/vol_sum) makes this
 * composition EXACT w.r.t. the direct kinetic-rate evaluation for shared-affinity
 * mechanisms (all mechanisms of a mineral sharing p, q).
 *
 * The engine-facing KIN operators are then
 *
 *   KIN_i = sum_m stoich[m, i] * rate_m,           i = 0..ne-1,
 *
 * with the exact product-rule gradient assembled from the inner interpolant's field
 * gradients — the returned derivative is the exact analytic gradient of the returned
 * value (same value/derivative consistency class as plain multilinear), the SR = 1
 * sign flip is located continuously (grid-free) and is shared by all ne equations of
 * a mineral (exact stoichiometric consistency across element sources).
 *
 * Continuity: C-infinity inside each inner-grid cell (product of multilinears with a
 * linear state factor), C0 across faces — identical continuity class to multilinear.
 * Compact support and manifold-restricted adaptive sampling of the inner interpolator
 * are untouched. No 2^d appears anywhere in this wrapper: plain runtime loops, valid
 * for any n_dims (no template instantiation growth toward d = 12-16).
 *
 * Ownership: the inner interpolator is NOT owned (Python keeps it alive; the pybind
 * binding adds keep_alive). Cache registration, hypercube caps and timers stay
 * attached to the inner interpolator, which this class only forwards to.
 */
class kinetic_composition_cpu_interpolator : public operator_set_gradient_evaluator_cpu
{
public:
   kinetic_composition_cpu_interpolator(operator_set_gradient_evaluator_iface *inner_,
                                        int n_dims_, int n_ops_, int kin_op_start_, int ne_,
                                        const std::vector<int> &mineral_axes_,
                                        const std::vector<double> &c_coeffs_,
                                        const std::vector<double> &p_aff_,
                                        const std::vector<double> &q_aff_,
                                        const std::vector<double> &stoich_flat_)
       : inner(inner_), n_dims(n_dims_), n_ops(n_ops_), kin_op(kin_op_start_), ne(ne_),
         mineral_axes(mineral_axes_), c_coeffs(c_coeffs_), p_aff(p_aff_), q_aff(q_aff_),
         stoich(stoich_flat_)
   {
      n_min = static_cast<int>(mineral_axes.size());
      if (2 * n_min + 1 > ne)
         throw std::invalid_argument("kinetic_composition: 2*n_minerals+1 field slots exceed the KIN block size ne");
      if (static_cast<int>(c_coeffs.size()) != n_min || static_cast<int>(p_aff.size()) != n_min ||
          static_cast<int>(q_aff.size()) != n_min)
         throw std::invalid_argument("kinetic_composition: c/p/q size mismatch with mineral_axes");
      if (static_cast<int>(stoich.size()) != n_min * ne)
         throw std::invalid_argument("kinetic_composition: stoich size != n_minerals * ne");
      if (kin_op < 0 || kin_op + ne > n_ops)
         throw std::invalid_argument("kinetic_composition: KIN block out of operator range");
      for (int a : mineral_axes)
         if (a < 0 || a >= n_dims)
            throw std::invalid_argument("kinetic_composition: mineral axis out of state range");
   }

   int get_n_ops() const override { return n_ops; }
   int init() override { return 0; } // inner is initialized by its own creation path

   // Diagnostics forwarded so the engine's statistics printout keeps working.
   int get_axis_n_points(int axis) const override { return inner->get_axis_n_points(axis); }
   double get_axis_min(int axis) const override { return inner->get_axis_min(axis); }
   double get_axis_max(int axis) const override { return inner->get_axis_max(axis); }
   uint64_t get_n_interpolations() const override { return inner->get_n_interpolations(); }
   uint64_t get_n_points_used() const override { return inner->get_n_points_used(); }
   uint64_t get_n_points_total() const override { return inner->get_n_points_total(); }

   /// Single-point evaluation (values only; incidental path).
   int evaluate(const std::vector<double> &state, std::vector<double> &values) override
   {
      int r = inner->evaluate(state, values);
      if (r != 0)
         return r;
      compose_block(state.data(), &values[0], nullptr, nullptr);
      return 0;
   }

   /// Engine hot path: inner interpolation, then analytic composition of the KIN block.
   int evaluate_with_derivatives(const std::vector<double> &states, const std::vector<int> &states_idxs,
                                 std::vector<double> &values, std::vector<double> &derivatives) override
   {
      int r = inner->evaluate_with_derivatives(states, states_idxs, values, derivatives);
      if (r != 0)
         return r;
      const std::size_t np = states_idxs.size();
      for (std::size_t p = 0; p < np; ++p)
      {
         const std::size_t i = static_cast<std::size_t>(states_idxs[p]);
         compose_block(&states[i * n_dims],
                       &values[i * n_ops],
                       &derivatives[i * static_cast<std::size_t>(n_ops) * n_dims],
                       /*have_derivs=*/&derivatives);
      }
      return 0;
   }

private:
   /**
    * Compose the KIN block for one block/state. vals points at values[i*n_ops];
    * dervs points at derivatives[(i*n_ops)*n_dims] (row stride n_dims) or nullptr
    * for value-only composition.
    */
   void compose_block(const double *state, double *vals, double *dervs, void *have_derivs)
   {
      // scratch (small, on stack via alloca-free fixed bound would need templates; use members)
      rate.assign(n_min, 0.0);
      if (have_derivs)
         drate.assign(static_cast<std::size_t>(n_min) * n_dims, 0.0);

      const double rho_t = vals[kin_op + 2 * n_min];
      const double *drho_t = have_derivs ? &dervs[(kin_op + 2 * n_min) * n_dims] : nullptr;

      for (int m = 0; m < n_min; ++m)
      {
         const double z_m = state[mineral_axes[m]];
         const double SRb = vals[kin_op + m];
         const double A_m = vals[kin_op + n_min + m];
         const double c_m = c_coeffs[m];

         double kaff, dkaff; // affinity factor and its derivative w.r.t. SRb
         if (p_aff[m] == 1.0 && q_aff[m] == 1.0)
         {
            kaff = 1.0 - SRb;
            dkaff = -1.0;
         }
         else
         {
            const double pm = p_aff[m], qm = q_aff[m];
            const double srp = std::pow(std::max(SRb, 0.0), pm);
            const double base = 1.0 - srp;
            kaff = (base >= 0.0) ? std::pow(base, qm) : -std::pow(-base, qm);
            // d/dSRb (1 - SRb^p)^q = -q p SRb^{p-1} (1 - SRb^p)^{q-1}; bounded for q >= 1
            const double basep = (base >= 0.0) ? std::pow(base, qm - 1.0) : std::pow(-base, qm - 1.0);
            dkaff = -qm * pm * std::pow(std::max(SRb, 1e-300), pm - 1.0) * basep;
         }

         rate[m] = c_m * z_m * rho_t * A_m * kaff;

         if (have_derivs)
         {
            const double *dSRb = &dervs[(kin_op + m) * n_dims];
            const double *dA = &dervs[(kin_op + n_min + m) * n_dims];
            double *dr = &drate[static_cast<std::size_t>(m) * n_dims];
            for (int k = 0; k < n_dims; ++k)
            {
               double t = drho_t[k] * A_m * kaff + rho_t * dA[k] * kaff + rho_t * A_m * dkaff * dSRb[k];
               dr[k] = c_m * z_m * t;
            }
            // exact d z_m / d x_k = delta(k == axis_m)
            dr[mineral_axes[m]] += c_m * rho_t * A_m * kaff;
         }
      }

      // scatter KIN_i = sum_m stoich[m, i] * rate_m over the whole KIN block
      for (int i = 0; i < ne; ++i)
      {
         double v = 0.0;
         for (int m = 0; m < n_min; ++m)
            v += stoich[static_cast<std::size_t>(m) * ne + i] * rate[m];
         vals[kin_op + i] = v;
         if (have_derivs)
         {
            double *drow = &dervs[(kin_op + i) * n_dims];
            for (int k = 0; k < n_dims; ++k)
            {
               double dv = 0.0;
               for (int m = 0; m < n_min; ++m)
                  dv += stoich[static_cast<std::size_t>(m) * ne + i] * drate[static_cast<std::size_t>(m) * n_dims + k];
               drow[k] = dv;
            }
         }
      }
   }

public:
   operator_set_gradient_evaluator_iface *inner; ///< inner field interpolator (not owned)

private:
   int n_dims, n_ops, kin_op, ne, n_min;
   std::vector<int> mineral_axes;   ///< state-axis index of each mineral's solid composition
   std::vector<double> c_coeffs;    ///< per-mineral constant c_m = -s_init*1000*86.4
   std::vector<double> p_aff, q_aff; ///< per-mineral shared affinity exponents
   std::vector<double> stoich;      ///< flat [n_min x ne] stoichiometry
   std::vector<double> rate, drate; ///< scratch
};

#endif /* KINETIC_COMPOSITION_CPU_INTERPOLATOR_HPP */
