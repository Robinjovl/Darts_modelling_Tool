#ifndef PM_DISCRETIZER_HPP_
#define PM_DISCRETIZER_HPP_

#include <algorithm>
#include <vector>
#include "matrix.h"

#define SUM_N(N) ((N + 1) / 2 * N)

namespace pm
{
	const uint8_t ND = 3;

	typedef linalg::Matrix<value_t> Matrix;
	
	enum Ftype { MAT, BORDER, FRAC_TO_MAT, MAT_TO_FRAC, FRAC, FRAC_BOUND };
	class Face
	{
	public:
		index_t type;
		index_t cell_id1, cell_id2, face_id1, face_id2;
		Matrix n, c;
		value_t area;
		std::vector<index_t> pts;
		uint8_t is_impermeable;
	public:
		Face() : is_impermeable(0) {};
		Face(index_t _type,
			index_t _cell_id1,
			index_t _cell_id2,
			index_t _face_id1,
			index_t _face_id2,
			value_t _area,
			std::valarray<value_t>& _n,
			std::valarray<value_t>& _c) : type(_type), cell_id1(_cell_id1), cell_id2(_cell_id2),
											face_id1(_face_id1), face_id2(_face_id2), area(_area), 
											n(_n, ND, 1), c(_c, ND, 1), is_impermeable(0) {};
		Face(index_t _type,
			index_t _cell_id1,
			index_t _cell_id2,
			index_t _face_id1,
			index_t _face_id2,
			value_t _area,
			std::valarray<value_t>& _n,
			std::valarray<value_t>& _c,
			uint8_t _is_impermeable) : type(_type), cell_id1(_cell_id1), cell_id2(_cell_id2),
			face_id1(_face_id1), face_id2(_face_id2), area(_area),
			n(_n, ND, 1), c(_c, ND, 1), is_impermeable(_is_impermeable) {};
		Face(index_t _type,
			index_t _cell_id1,
			index_t _cell_id2,
			index_t _face_id1,
			index_t _face_id2,
			value_t _area,
			std::valarray<value_t>& _n,
			std::valarray<value_t>& _c,
			std::vector<index_t>& _pts) : type(_type), cell_id1(_cell_id1), cell_id2(_cell_id2),
											face_id1(_face_id1), face_id2(_face_id2), area(_area), 
											n(_n, ND, 1), c(_c, ND, 1), pts(_pts), is_impermeable(0) {};
		Face(index_t _type,
			index_t _cell_id1,
			index_t _cell_id2,
			index_t _face_id1,
			index_t _face_id2,
			value_t _area,
			std::valarray<value_t>& _n,
			std::valarray<value_t>& _c,
			std::vector<index_t>& _pts,
			uint8_t _is_impermeable) : type(_type), cell_id1(_cell_id1), cell_id2(_cell_id2),
									face_id1(_face_id1), face_id2(_face_id2), area(_area), 
									n(_n, ND, 1), c(_c, ND, 1), pts(_pts), is_impermeable(_is_impermeable) {};
		~Face() {};
	};
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

	class Stiffness : public Matrix
	{
	public:
		static const index_t N = SUM_N(ND) * SUM_N(ND);
		typedef Matrix Base;

		Stiffness() : Base(SUM_N(ND), SUM_N(ND)) {};
		Stiffness(value_t la, value_t mu) : Base(SUM_N(ND), SUM_N(ND))
		{
			(*this)(0,0) = (*this)(1,1) = (*this)(2,2) = la + 2 * mu;
			(*this)(3,3) = (*this)(4,4) = (*this)(5,5) = mu;
			(*this)(0,1) = (*this)(0,2) = (*this)(1,2) = la;
			(*this)(1,0) = (*this)(2,0) = (*this)(2,1) = la;
		};
		Stiffness(std::valarray<value_t> _c) : Base(_c, SUM_N(ND), SUM_N(ND)) {}
	};

	struct Approximation {
		Matrix a, a_biot, f, f_biot;
		std::vector<index_t> stencil;
	};

	enum Scheme { DEFAULT, APPLY_EIGEN_SPLITTING, APPLY_EIGEN_SPLITTING_NEW, AVERAGE };
	
	class pm_discretizer
	{
	public:
		struct Gradients { std::vector<index_t> stencil; Matrix mat; Matrix rhs; };
		struct InnerMatrices { Matrix T1, T2, G1, G2, A1, A2, Q1, Q2, Th1, Th2, R1, R2, y1, y2, S1, S2;	value_t r1, r2, beta_stab1, beta_stab2, k_stab1, k_stab2, c_stab1, c_stab2; };
	protected:
		static const value_t darcy_constant, heat_cond_constant;
		std::map<index_t, Matrix> pre_A, pre_rest, pre_rhs_mult;
		std::map<index_t, Matrix> pre_Ad, pre_restd, pre_rhs_multd;
		std::map<index_t, Matrix> pre_frac_grad_mult, pre_Wsvd, pre_Zsvd, pre_w_svd;
		std::map<index_t, std::map<index_t,Matrix>> pre_cur_rhs, pre_cur_rhsd;
		std::map<index_t, Matrix> pre_merged_grad;
		std::vector<index_t> pre_merged_stencil;
		std::vector<Approximation> pre_merged_flux;
		index_t st_id;

		Matrix W;
		std::vector<std::map<index_t, InnerMatrices>> inner;

		Matrix get_u_face_prev(const Matrix dr, const index_t cell_id) const;
		Matrix get_ub_prev(const Face& face) const;
		Matrix calc_grad_prev(const index_t cell_id) const;
		Matrix calc_grad_cur(const index_t cell_id) const;
		Matrix calc_vector(const Matrix& a, const Matrix& rhs, const std::vector<index_t>& stencil) const;
		Gradients merge_stencils(const std::vector<index_t>& st1, const Matrix& m1, const std::vector<index_t>& st2, const Matrix& m2);
		Approximation& merge_approximations(const Approximation& flux1, const Approximation& flux2, const index_t ws_id);
		bool check_trans_sum(const std::vector<index_t>& st, const Matrix& a) const;
		void write_trans(const std::vector<index_t>& st, const Matrix& from);
		void write_trans_biot(const std::vector<index_t>& st, const Matrix& from, const Matrix& from_biot);
		void write_trans_biot(const std::vector<index_t>& st, const Matrix& from, const Matrix& from_biot, const Matrix& from_face_unknowns);
		void write_trans_biot_therm_cond(const std::vector<index_t>& st, const Matrix& from, const Matrix& from_biot, const Matrix& from_th_cond);
		
		void contact_mixing(value_t dt, index_t cell_id, index_t fault_id, const Face& face);
		void contact_mixing_new(value_t dt, index_t cell_id, index_t fault_id, const Face& face);

		index_t counter, MERGE_BLOCK_SIZE;
		index_t id, id1, id2;
		inline index_t check_face_is_fault(const Face& face)
		{
			const auto& vec_face = faces[face.cell_id1];
			for (index_t i = vec_face.size() - 1; i > -1; --i)
			{
				const auto& fface = vec_face[i];
				if (fface.type == MAT_TO_FRAC && fface.face_id1 == face.face_id1) return i;
			}
			return -1;
		};

		std::vector<Approximation> fluxes, fluxes_th_cond, face_unknowns;
	public:
		int n_matrix, n_cells, n_fracs, n_faces, nb_faces;
		static const Matrix I3;
		static const Matrix I4;

		std::vector<index_t>::const_iterator it_find;
		std::pair<bool, size_t> res1, res2;
		inline std::pair<bool, size_t> findInVector(const std::vector<index_t>& vec, const index_t& element)
		{
			// Find given element in vector
			it_find = std::find(vec.begin(), vec.end(), element);
			if (it_find != vec.end())
			{
				return { true, std::distance(vec.begin(), it_find) };
			}
			else
			{
				return { false, -1 };
			}
		};
		inline value_t get_fault_sign(Matrix n, index_t ref_frac_id) const
		{
			const auto& vec_faces = faces[ref_frac_id];
			const auto& ref = vec_faces[vec_faces.size() - 1];
			assert(ref.type == FRAC_TO_MAT);
			return (n.transpose() * ref.n).values[0] > 0 ? 1.0 : -1.0;
		}

		std::vector<std::vector<Face>> faces;
		std::vector<Matrix33> perms;
		std::vector<Matrix33> diffs;
		std::vector<Matrix33> biots;
		std::vector<Matrix> th_expns;
		std::vector<Stiffness> stfs;
		std::vector<Matrix> cell_centers, u0;
		std::vector<Matrix> bc, bc_prev;
		std::vector<value_t> x_prev;
		std::vector<value_t> frac_apers;
		value_t visc, grav, density;
		Matrix grav_vec;
		// Results
		std::vector<index_t> cell_m, cell_p, stencil, offset;
		std::vector<value_t> tran, rhs, tran_biot, rhs_biot;
		std::vector<value_t> tran_th_expn, tran_th_cond;
		std::vector<value_t> tran_face_unknown, rhs_face_unknown;
		std::vector<Gradients> grad, grad_d;
		std::vector<Gradients> grad_prev;

		static const index_t MIN_FACE_NUM = 1;
		static const index_t MAX_FACE_NUM = 50;
		static const index_t BLOCK_SIZE;
		static const int MAX_STENCIL = 50;
		static const int MAX_POINTS_PER_CELL = 8;
		static const index_t MAX_FLUXES_NUM = 8;

		Scheme scheme;
		bool ASSEMBLE_HEAT_CONDUCTION;
		bool NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION;

		value_t min_alpha_stabilization;
		value_t max_alpha;
		std::vector<value_t> dt_max_alpha_in_domain, max_alpha_in_domain;

		// node-based gradient reconstruction
		std::vector<index_t> cells_to_node, nodes_to_face;
		std::vector<index_t> cells_to_node_offset, nodes_to_face_offset, nodes_to_face_cell_offset;
		// contact-related things
		std::vector<index_t> ref_contact_ids;
	public:
		pm_discretizer();
		~pm_discretizer();

		void init(const index_t _n_matrix, const index_t _n_fracs, std::vector<index_t>& _ref_contact_ids);
		void reconstruct_gradients_per_cell(value_t dt);
		//void reconstruct_gradients_per_node(value_t dt, index_t n_nodes);
		void reconstruct_gradients_thermal_per_cell(value_t dt);
		void calc_all_fluxes_once(value_t dt);
		void calc_border_flux(value_t dt, const Face& face, Approximation& flux, Approximation& flux_th_cond, Approximation& face_unknown);
		void calc_contact_flux(value_t dt, const Face& face, Approximation& flux, Approximation& flux_flow_biot, index_t fault_id);
		void calc_contact_flux_new(value_t dt, const Face& face, Approximation& flux, Approximation& flux_flow_biot, index_t fault_id);
		void calc_matrix_fault_flow_flux(value_t dt, const Face& face, Approximation& flux, Approximation& flux_biot, index_t fault_id);
		void calc_matrix_flux(value_t dt, const Face& face, Approximation& flux, Approximation& flux_th_cond, Approximation& face_unknown);
		void calc_matrix_flux_stabilized(value_t dt, const Face& face, Approximation& flux);
		void calc_matrix_flux_stabilized_new(value_t dt, const Face& face, Approximation& flux);
		void calc_fault_matrix(value_t dt, const Face& face, Approximation& flux);
		void calc_fault_fault(value_t dt, const Face& face, Approximation& flux);
		void calc_avg_matrix_flux(value_t dt, const Face& face, Approximation& flux);

		std::tuple<std::vector<index_t>, std::valarray<value_t>> get_gradient(const index_t cell_id);
		std::tuple<std::vector<index_t>, std::valarray<value_t>> get_thermal_gradient(const index_t cell_id);
	};
};

#endif /* PM_DISCRETIZER_HPP_ */
