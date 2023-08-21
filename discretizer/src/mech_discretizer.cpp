#include "mech_discretizer.h"


using namespace dis;
using std::vector;

template <MechDiscretizerMode MODE>
const index_t MechDiscretizer<MODE>::M = BLOCK_DIM.at(MODE).first;
template <MechDiscretizerMode MODE>
const index_t MechDiscretizer<MODE>::N = BLOCK_DIM.at(MODE).second;

template <MechDiscretizerMode MODE>
MechDiscretizer<MODE>::MechDiscretizer()
{
  NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION = true;
}

template <MechDiscretizerMode MODE>
MechDiscretizer<MODE>::~MechDiscretizer()
{
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::init()
{
  Discretizer::init();

  for (index_t i = mesh::MIN_CONNS_PER_ELEM; i < mesh::MAX_CONNS_PER_ELEM; i++)
  {
	pre_grad_A_u[i] = Matrix(ND * i, ND * ND);
	pre_grad_R_u[i] = Matrix(ND * i, i + 1);
	pre_grad_rhs_u[i] = Matrix(ND * i, 1);
  }

  /*const uint8_t BLOCK_SIZE = 1;
  fluxes.resize(MAX_FLUXES_NUM);
  pre_merged_flux.resize(MAX_FLUXES_NUM);
  for (uint8_t k = 0; k < MAX_FLUXES_NUM; k++)
  {
	// Darcy's, elastic fluxes and Biot's fluxes 
	auto& flux = fluxes[k];
	flux.a = Matrix(BLOCK_SIZE, MAX_STENCIL * BLOCK_SIZE);
	flux.a_homo = Matrix(BLOCK_SIZE, MAX_STENCIL * BLOCK_SIZE);
	flux.a_thermal = Matrix(BLOCK_SIZE, MAX_STENCIL * BLOCK_SIZE);
	flux.rhs = Matrix(BLOCK_SIZE, 1);
	flux.stencil.reserve(MAX_STENCIL);
	// Premerged fluxes
	pre_merged_flux[k].a = Matrix(BLOCK_SIZE, MAX_STENCIL * BLOCK_SIZE);
	pre_merged_flux[k].rhs = Matrix(BLOCK_SIZE, 1);
	pre_merged_flux[k].stencil.reserve(MAX_STENCIL);
  }*/
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::reconstruct_displacement_gradients_per_cell(const MechBoundaryCondition& bc)
{
  // Variables
  size_t n_cur_faces;
  std::vector<index_t> st;		st.reserve(MAX_STENCIL);
  std::vector<index_t> admissible_connections(4, 0);
  index_t el_id1, el_id2;

  // allocate memory for arrays
  u_grad_stencil.reserve(mesh->num_of_elements * MAX_STENCIL);
  u_grad_offset.reserve(mesh->num_of_elements + 1);
  u_grad_vals.reserve(ND * ND * mesh->num_of_elements * MAX_STENCIL);
  u_grad_rhs.reserve(ND * ND * mesh->num_of_elements);

  // loop through the adjacency matrix (matrix cells)
  for (index_t i = 0; i < mesh->region_ranges.at(mesh::MATRIX).second; i++)
  {
	// Build the system from the continuity at the interfaces
	n_cur_faces = 0;// vec_faces.size();
	for (index_t j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
	{
	  const auto& conn = mesh->conns[mesh->adj_matrix[j]];
	  if (conn.type == mesh::MAT_BOUND)
	  {
		// Coefficients that define boundary condition
		const auto& an = bc.a_n[conn.elem_id2 - mesh->n_cells];
		const auto& bn = bc.b_n[conn.elem_id2 - mesh->n_cells];
		const auto& at = bc.a_t[conn.elem_id2 - mesh->n_cells];
		const auto& bt = bc.b_t[conn.elem_id2 - mesh->n_cells];

		if (NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION || an != 0.0 || at != 0.0)	n_cur_faces++;
	  }
	  else if (conn.type != mesh::MAT_FRAC) n_cur_faces++;
	}


  }
}

template class MechDiscretizer<POROELASTIC>;
template class MechDiscretizer<THERMOPOROELASTIC>;