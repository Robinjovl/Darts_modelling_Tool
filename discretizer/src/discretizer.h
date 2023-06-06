#ifndef DISCRETIZER_H_
#define DISCRETIZER_H_

#include "mesh/mesh.h"

namespace dis
{
	using mesh::index_t;
	using mesh::value_t;
	using mesh::Matrix;
	using mesh::Mesh;
	using mesh::PhysicalTags;
	using mesh::Vector3;
	using mesh::ND;

	const value_t DARCY_CONSTANT = 0.0085267146719160104986876640419948;
	const uint8_t MAX_STENCIL = 12;
	const uint8_t MAX_FLUXES_NUM = 8;

	/* Boundary condition */
	class BoundaryCondition
	{
	public:
		std::vector<value_t> a, b, r;
		BoundaryCondition() {};
		~BoundaryCondition() {};
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
	struct Approximation 
	{
		Approximation() {};
		Approximation(uint8_t M, uint8_t N)
		{
			a = Matrix(M, N);
			rhs = Matrix(M, 1);
			stencil.resize(N);
		};
		Matrix a, rhs;
		std::vector<index_t> stencil;
	};

	/* Discretiser */
	class Discretizer
	{
	protected:
		Mesh* mesh;

		std::unordered_map<index_t, Matrix> pre_grad_A, pre_grad_R, pre_grad_rhs;
		std::unordered_map<index_t, Matrix> pre_Wsvd, pre_Zsvd, pre_w_svd;
		std::vector<Approximation> pre_merged_flux, fluxes;

		void calc_matrix_matrix(const mesh::Connection& conn, Approximation& flux, const index_t adj_mat_id1, const index_t adj_mat_id2);
		void calc_fault_fault(const mesh::Connection& conn, Approximation& flux);
		void calc_matrix_boundary(const mesh::Connection& conn, Approximation& flux, const index_t adj_mat_id1);

		inline void write_trans(const Approximation& flux)
		{
			value_t buf;
			// free term (gravity)
			flux_rhs.push_back(flux.rhs.values[0]);
			// stencil & transmissibilities
			for (uint8_t st_id = 0; st_id < flux.stencil.size(); st_id++)
			{
				buf = flux.a.values[st_id];
				if (fabs(buf) > EQUALITY_TOLERANCE)
				{
					flux_vals.push_back(buf);
					flux_stencil.push_back(flux.stencil[st_id]);
				}
			}
			// offset
			flux_offset.push_back(static_cast<index_t>(flux_stencil.size()));
		};
		std::vector<index_t> find_connections_to_reconstruct_gradient(const index_t cell_id, const index_t cur_conn_id);
	public:
		void set_mesh(Mesh* _mesh);
		void init();

		Discretizer();
		~Discretizer();

		// Array with permeability matrices
		std::vector<Matrix33> perms;
		// Arrays of scalar permeabilities (diagonal entries in tensor) in the case of CPG
		std::vector<value_t> permx, permy, permz;
		// Array of porosities in the case of CPG
		std::vector<value_t> poro;
		// Arrays of cell IDs for each connection
		std::vector<index_t> cell_m;
		std::vector<index_t> cell_p;

		/* MPFA */

		// pressure gradient transmissibilities
		std::vector<value_t> grad_vals;
		// pressure gradient offsets
		std::vector<index_t> grad_offset;
		// pressure gradient stencil
		std::vector<index_t> grad_stencil;
		// pressure gradient free-term (gravity)
		std::vector<value_t> grad_rhs;

		//std::vector<double> pressuregrad;
		/* Fluxes */
		// array of fluxes through elements
		std::vector<value_t> flux_vals;
		// array of fluxes through matrix elements ! USED ONLY FOR DEBUGGING PURPOSES
		std::vector<Matrix> fluxes_matrix;
		// array of offsets of fluxes
		std::vector<index_t> flux_offset;
		// Array of cell IDs
		std::vector<index_t> flux_stencil;
		// free-term (gravity) in flux approximation
		std::vector<value_t> flux_rhs;

		bool USE_CONNECTION_BASED_GRADIENTS;

		// Two-Point Flux Approximation
		void calc_tpfa_transmissibilities(const PhysicalTags& tags);
		// Multi-Point Flux Approximation
		void reconstruct_pressure_gradients_per_cell(const BoundaryCondition& bc);
		void reconstruct_pressure_gradients_per_face(const BoundaryCondition& bc);
		void calc_mpfa_transmissibilities(BoundaryCondition& _bc);

		void calcPermeabilitySimple(const double permx = 1, const double permy = 1, const double permz = 1);
		void set_permeability(std::vector<value_t> &permx, std::vector<value_t> &permy, std::vector<value_t> &permz);
		void set_porosity(std::vector<value_t> &new_poro);

		std::vector<index_t> get_one_way_tpfa_transmissibilities() const;
		void write_tran_cube(std::string fname, std::string fname_nnc) const;
		std::vector<value_t> get_fault_xyz() const;
		void write_tran_list(std::string fname) const;

		// method which computes the flux between two elements
		Matrix mergeMatrices(Matrix &m1, Matrix &m2, std::vector<index_t> &cont1, std::vector<index_t> &cont2, std::vector<index_t>& comb_cont);
		index_t nbContributors(std::vector<index_t>& cont1, std::vector<index_t>& cont2, std::vector<index_t>& comb_cont);
		BoundaryCondition bc;

		static const Matrix I3;
		static const Matrix I4;
		Matrix grav_vec;
    };
}

#endif /* DISCRETIZER_H_ */
