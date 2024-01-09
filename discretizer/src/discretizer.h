#ifndef DISCRETIZER_H_
#define DISCRETIZER_H_

#include "approximation.h"

namespace dis
{
	using mesh::Mesh;
	using mesh::PhysicalTags;
	using mesh::Vector3;
	using mesh::ND;

	const value_t DARCY_CONSTANT = 0.0085267146719160104986876640419948;
	const uint8_t MAX_STENCIL = 30;
	const uint8_t MAX_FLUXES_NUM = 8;

	/**
	 * @brief Represents a generic boundary condition.
	 *
	 * This structure holds data for boundary elements, supporting both Dirichlet and Neumann types.
	 */
	struct BoundaryCondition
	{
	  std::vector<value_t> a; ///< Dirichlet type boundary condition values.
	  std::vector<value_t> b; ///< Neumann type boundary condition values.
	};

	/* Special class for 2nd rank 3x3 tensor */
	class Matrix33 : public Matrix
	{
	public:
		typedef Matrix Base;
		static const index_t N = ND * ND;

		Matrix33() : Base(ND, ND) {};
		Matrix33(value_t kx) : Base(ND, ND)
		{
			(*this)(0, 0) = kx;	(*this)(1, 1) = kx;	(*this)(2, 2) = kx;
		};
		Matrix33(value_t kx, value_t ky, value_t kz) : Base(ND, ND)
		{
			(*this)(0, 0) = kx;	(*this)(1, 1) = ky;	(*this)(2, 2) = kz;
		};
		Matrix33(std::valarray<value_t> _m) : Base(_m, ND, ND) {}
	};

	/* Here is what we call linear approximation */
	struct FlowHeatApproximation 
	{
	  FlowHeatApproximation() {};
	  FlowHeatApproximation(index_t stencil_size)
	  {
		darcy = LinearApproximation<Pvar>(1, stencil_size);
		fick = LinearApproximation<Pvar>(1, stencil_size);
		fourier = LinearApproximation<Tvar>(1, stencil_size);
	  };

	  LinearApproximation<Pvar> darcy, fick;
	  LinearApproximation<Tvar> fourier;

	  bool is_same_stencil = true;
	};

	/* Discretiser */
	class Discretizer
	{
	protected:
		std::unordered_map<index_t, Matrix> pre_grad_A_p, pre_grad_R_p, pre_grad_rhs_p;
		std::unordered_map<index_t, Matrix> pre_grad_A_th, pre_grad_R_th;
		std::unordered_map<index_t, Matrix> pre_Wsvd, pre_Zsvd, pre_w_svd;
		std::vector<FlowHeatApproximation> pre_merged_flux, fluxes;

		void calc_matrix_matrix(const mesh::Connection& conn, FlowHeatApproximation& flux, const bool with_thermal = false);
		void calc_fault_fault(const mesh::Connection& conn, FlowHeatApproximation& flux);
		void calc_matrix_boundary(const mesh::Connection& conn, FlowHeatApproximation& flux, const bool with_thermal = false);

		inline void write_trans(const FlowHeatApproximation& flux)
		{
		  assert(flux.is_same_stencil);
		  value_t buf, buf_homo;
		  // free term (gravity)
		  flux_rhs.push_back(flux.darcy.rhs.values[0]);
		  // stencil & transmissibilities
		  for (uint8_t st_id = 0; st_id < flux.darcy.stencil.size(); st_id++)
		  {
			  buf = flux.darcy.a.values[st_id];
			  buf_homo = flux.fick.a.values[st_id];
			  if (fabs(buf) > EQUALITY_TOLERANCE)
			  {
				  flux_vals.push_back(buf);
				  flux_vals_homo.push_back(buf_homo);
				  flux_stencil.push_back(flux.darcy.stencil[st_id]);
			  }
		  }
		  // offset
		  flux_offset.push_back(static_cast<index_t>(flux_stencil.size()));
		};
		inline void write_trans_thermal(const FlowHeatApproximation& flux)
		{
		  assert(flux.is_same_stencil);
		  value_t buf, buf_homo, buf_t;
		  // free term (gravity)
		  flux_rhs.push_back(flux.darcy.rhs.values[0]);
		  // stencil & transmissibilities
		  for (uint8_t st_id = 0; st_id < flux.darcy.stencil.size(); st_id++)
		  {
			buf = flux.darcy.a.values[st_id];
			buf_homo = flux.fick.a.values[st_id];
			buf_t = flux.fourier.a.values[st_id];
			if (fabs(buf) + fabs(buf_t) > EQUALITY_TOLERANCE)
			{
			  flux_vals.push_back(buf);
			  flux_vals_homo.push_back(buf_homo);
			  flux_vals_thermal.push_back(buf_t);
			  flux_stencil.push_back(flux.darcy.stencil[st_id]);
			}
		  }
		  // offset
		  flux_offset.push_back(static_cast<index_t>(flux_stencil.size()));
		};
		std::vector<index_t> find_connections_to_reconstruct_gradient(const index_t cell_id, const index_t cur_conn_id);
	public:
		Mesh* mesh;
		void set_mesh(Mesh* _mesh);
		virtual void init();

		Discretizer();
		~Discretizer();

		// Array with permeability matrices
		std::vector<Matrix33> perms;
		// Arrays of scalar permeabilities (diagonal entries in tensor) in the case of CPG
		std::vector<value_t> permx, permy, permz;
		// Array with heat conductivity matrices;
		std::vector<Matrix33> heat_conductions;
		// Array of porosities in the case of CPG
		std::vector<value_t> poro;
		// Arrays of cell IDs for each connection
		std::vector<index_t> cell_m;
		std::vector<index_t> cell_p;

		/* MPFA */
		std::vector<LinearApproximation<Pvar>> p_grads;
		std::vector<LinearApproximation<Tvar>> t_grads;

		//std::vector<double> pressuregrad;
		/* Fluxes */
		// array of fluxes through elements
		std::vector<value_t> flux_vals;
		// homogeneous transmissibilities (aka geometric part, for diffusion, heat conduction in fluids)
		std::vector<value_t> flux_vals_homo;
		// thermal transmissibilities
		std::vector<value_t> flux_vals_thermal;
		// array of fluxes through matrix elements ! USED ONLY FOR DEBUGGING PURPOSES
		std::vector<Matrix> fluxes_matrix;
		// array of offsets of fluxes
		std::vector<index_t> flux_offset;
		// Array of cell IDs
		std::vector<index_t> flux_stencil;
		// free-term (gravity) in flux approximation
		std::vector<value_t> flux_rhs;

		// Two-Point Flux Approximation
		void calc_tpfa_transmissibilities(const PhysicalTags& tags);
		// Multi-Point Flux Approximation
		void reconstruct_pressure_gradients_per_cell(const BoundaryCondition& _bc);
		void reconstruct_pressure_temperature_gradients_per_cell(const BoundaryCondition& _bc_flow, const BoundaryCondition& _bc_heat);
		// void reconstruct_pressure_gradients_per_face(const BoundaryCondition& bc);
		void calc_mpfa_transmissibilities(const bool with_thermal = false);

		void calcPermeabilitySimple(const double permx = 1, const double permy = 1, const double permz = 1);
		void set_permeability(std::vector<value_t> &permx, std::vector<value_t> &permy, std::vector<value_t> &permz);
		void set_porosity(std::vector<value_t> &new_poro);

		std::vector<index_t> get_one_way_tpfa_transmissibilities() const;
		void write_tran_cube(std::string fname, std::string fname_nnc) const;
		std::vector<value_t> get_fault_xyz() const;
		void write_tran_list(std::string fname) const;

		BoundaryCondition bc_flow, bc_heat;

		static const Matrix I3;
		static const Matrix I4;
		Matrix grav_vec; // gravity_constant * grad(z); density is multiplied in engine
    };
}

#endif /* DISCRETIZER_H_ */
