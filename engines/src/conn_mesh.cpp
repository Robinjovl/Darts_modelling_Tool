#include "conn_mesh.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdlib.h>
#include <assert.h>

using namespace std;
int 
conn_mesh::init(std::vector<index_t>& block_m, std::vector<index_t>& block_p, std::vector<value_t>& tran, std::vector<value_t>& tranD)
{
  int diff_trans;

  diff_trans = tranD.size();
  n_conns = tran.size();
  
  one_way_block_m = block_m;
  one_way_block_p = block_p;
  one_way_tran = tran;
  one_way_tranD = tranD;
  
  n_res_blocks = *(std::max_element(one_way_block_m.begin(), one_way_block_m.end())) + 1;
  n_res_blocks = std::max(n_res_blocks, *(std::max_element(one_way_block_p.begin(), one_way_block_p.end())) + 1);

  n_blocks = n_res_blocks;
  n_one_way_conns = n_conns;
  n_one_way_conns_res = n_conns;

  poro.resize(n_res_blocks);
  volume.resize(n_res_blocks);
  initial_state.resize(n_res_blocks * n_vars);
  op_num.assign(n_res_blocks, 0);
  depth.assign(n_res_blocks, 0);
  heat_capacity.assign(n_res_blocks, 0);
  rock_cond.assign(n_res_blocks, 0);

  // kinetic property
  kin_factor.assign(n_res_blocks, 1);  // if I want backwards compatibility with older version of python files I assume it needs to be filled with a 1 here (in case people don't actually use this factor!)

  // mobility multiplier
  mob_multiplier.assign(n_res_blocks * 2, 1);   // assume two phases present and default multiplier 1

  return 0;
}

/*int
conn_mesh::init_mpfa(std::vector<index_t>& block_m,
                    std::vector<index_t>& block_p,
                    std::vector<index_t>& _fstencil,
                    std::vector<index_t>& _fst_offset,
                    std::vector<value_t>& _ftran,
					std::vector<value_t>& _rhs,
					index_t _n_matrix, index_t _n_bounds)
{
	n_vars = 1;
	n_conns = block_m.size();

    one_way_block_m = block_m;
    one_way_block_p = block_p;
    one_way_stencil = _fstencil;
    one_way_offset = _fst_offset;
    one_way_tran = _ftran;
	one_way_rhs = _rhs;

	n_res_blocks = n_matrix = _n_matrix;
	n_bounds = _n_bounds;

    n_blocks = n_res_blocks;
    n_one_way_conns = n_conns;
    n_one_way_conns_res = n_conns;

    poro.resize(n_res_blocks);
    volume.resize(n_res_blocks);
	initial_state.resize(n_res_blocks * n_vars);
    op_num.assign(n_res_blocks, 0);
    depth.assign(n_res_blocks + n_bounds, 0);
    heat_capacity.assign(n_res_blocks, 0);
    rock_cond.assign(n_res_blocks, 0);
	bc.resize(3 * n_bounds);
	f.resize(2 * n_res_blocks);

    // kinetic property
    kin_factor.assign(n_res_blocks, 1);  // if I want backwards compatibility with older version of python files I assume it needs to be filled with a 1 here (in case people don't actually use this factor!)

    return 0;
}*/

int
conn_mesh::init_mpfa(std::vector<index_t>& block_m,
	std::vector<index_t>& block_p,
	std::vector<index_t>& _fstencil,
	std::vector<index_t>& _fst_offset,
	std::vector<value_t>& _ftran,
	std::vector<value_t>& _rhs,
	std::vector<value_t>& _dtran,
	std::vector<value_t>& _htran,
	index_t _n_matrix, index_t _n_bounds, index_t _n_fracs, index_t _n_vars)
{
	n_vars = 1;
	n_conns = block_m.size();

	one_way_block_m = block_m;
	one_way_block_p = block_p;
	one_way_stencil = _fstencil;
	one_way_offset = _fst_offset;
	one_way_tran = _ftran;
	one_way_tranD = _dtran;
	one_way_tran_heat_cond = _htran;
	one_way_rhs = _rhs;

	n_matrix = _n_matrix;
	n_res_blocks = _n_matrix + _n_fracs;
	n_bounds = _n_bounds;

	n_blocks = n_res_blocks;
	n_one_way_conns = n_conns;
	n_one_way_conns_res = n_conns;

	poro.resize(n_res_blocks);
	volume.resize(n_res_blocks);
	initial_state.resize(n_res_blocks * n_vars);
	op_num.assign(n_res_blocks, 0);
	depth.assign(n_res_blocks + n_bounds, 0);
	heat_capacity.assign(n_res_blocks, 0);
	rock_cond.assign(n_res_blocks, 0);
	bc.resize(_n_vars * n_bounds);
	f.resize(_n_vars * n_res_blocks);

	// kinetic property
	kin_factor.assign(n_res_blocks, 1);  // if I want backwards compatibility with older version of python files I assume it needs to be filled with a 1 here (in case people don't actually use this factor!)

	return 0;
}

/*int
conn_mesh::init_mpfa(std::vector<index_t>& block_m,
	std::vector<index_t>& block_p,
	std::vector<index_t>& _fstencil,
	std::vector<index_t>& _fst_offset,
	std::vector<value_t>& _ftran,
	std::vector<value_t>& _rhs,
	std::vector<value_t>& _flux,
	index_t _n_matrix, index_t _n_bounds, index_t _n_fracs)
{
	n_vars = 1;
	n_conns = block_m.size();

	one_way_block_m = block_m;
	one_way_block_p = block_p;
	one_way_stencil = _fstencil;
	one_way_offset = _fst_offset;
	one_way_tran = _ftran;
	one_way_rhs = _rhs;
	one_way_flux = _flux;

	n_res_blocks = _n_matrix + _n_fracs;
	n_res_blocks = _n_matrix + _n_fracs;
	n_fracs = _n_fracs;
	n_bounds = _n_bounds;

	n_blocks = n_res_blocks;
	n_one_way_conns = n_conns;
	n_one_way_conns_res = n_conns;

	poro.resize(n_res_blocks);
	volume.resize(n_res_blocks);
	initial_state.resize(n_res_blocks * n_vars);
	op_num.assign(n_res_blocks, 0);
	depth.assign(n_res_blocks + n_bounds, 0);
	heat_capacity.assign(n_res_blocks, 0);
	rock_cond.assign(n_res_blocks, 0);
	bc.resize(3 * n_bounds);
	f.resize(2 * n_res_blocks);

	// kinetic property
	kin_factor.assign(n_res_blocks, 1);  // if I want backwards compatibility with older version of python files I assume it needs to be filled with a 1 here (in case people don't actually use this factor!)

	return 0;
}*/

int
conn_mesh::init_mpsa(std::vector<index_t>& block_m,
	std::vector<index_t>& block_p,
	std::vector<index_t>& _sstencil,
	std::vector<index_t>& _sst_offset,
	std::vector<value_t>& _stran,
	uint8_t _n_dim, 
	index_t _n_matrix, index_t _n_bounds, index_t _n_fracs)
{
	n_vars = _n_dim;
	n_conns = block_m.size();

	one_way_block_m = block_m;
	one_way_block_p = block_p;
	one_way_stencil = _sstencil;
	one_way_offset = _sst_offset;
	one_way_tran = _stran;

	n_matrix = _n_matrix;
	n_fracs = _n_fracs;
	n_bounds = _n_bounds;
	n_res_blocks = n_matrix + n_fracs;
	n_blocks = n_res_blocks;
	n_one_way_conns = n_conns;
	n_one_way_conns_res = n_conns;

	poro.resize(n_res_blocks);
	volume.resize(n_res_blocks);
	displacement.resize(n_vars * n_res_blocks);
	op_num.assign(n_res_blocks, 0);
	depth.assign(n_res_blocks + n_bounds, 0);
	bc.resize((3 + n_vars) * n_bounds);
	f.resize(n_vars * n_res_blocks);

	return 0;
}

int
conn_mesh::init_mpsa(std::vector<index_t>& block_m,
	std::vector<index_t>& block_p,
	std::vector<index_t>& _sstencil,
	std::vector<index_t>& _sst_offset,
	std::vector<value_t>& _stran,
	std::vector<value_t>& _flux,
	uint8_t _n_dim,
	index_t _n_matrix, index_t _n_bounds, index_t _n_fracs)
{
	n_vars = _n_dim;
	n_conns = block_m.size();

	one_way_block_m = block_m;
	one_way_block_p = block_p;
	one_way_stencil = _sstencil;
	one_way_offset = _sst_offset;
	one_way_tran = _stran;
	one_way_flux = _flux;

	n_matrix = _n_matrix;
	n_fracs = _n_fracs;
	n_bounds = _n_bounds;
	n_res_blocks = n_matrix + n_fracs;
	n_blocks = n_res_blocks;
	n_one_way_conns = n_conns;
	n_one_way_conns_res = n_conns;

	poro.resize(n_res_blocks);
	volume.resize(n_res_blocks);
	displacement.resize(n_vars * n_res_blocks);
	op_num.assign(n_res_blocks, 0);
	depth.assign(n_res_blocks + n_bounds, 0);
	bc.resize((3 + n_vars) * n_bounds);
	f.resize(n_vars * n_res_blocks);

	return 0;
}

int
conn_mesh::init_pm(std::vector<index_t>& block_m,
	std::vector<index_t>& block_p,
	std::vector<index_t>& _stencil,
	std::vector<index_t>& _st_offset,
	std::vector<value_t>& _tran,
	std::vector<value_t>& _rhs,
	index_t _n_matrix, index_t _n_bounds, index_t _n_fracs)
{
	n_vars = 4;
	n_conns = block_m.size();

	one_way_block_m = block_m;
	one_way_block_p = block_p;
	one_way_stencil = _stencil;
	one_way_offset = _st_offset;
	one_way_tran = _tran;
	one_way_rhs = _rhs;

	n_matrix = _n_matrix;
	n_bounds = _n_bounds;
	n_fracs = _n_fracs;
	n_res_blocks = n_matrix + n_fracs;
	n_blocks = n_res_blocks;
	n_one_way_conns = n_conns;
	n_one_way_conns_res = n_conns;

	poro.resize(n_res_blocks);
	volume.resize(n_res_blocks);
	initial_state.resize(n_res_blocks * n_vars);
	displacement.resize(3 * n_res_blocks);
	op_num.assign(n_res_blocks, 0);
	depth.assign(n_res_blocks, 0);
	heat_capacity.assign(n_res_blocks, 0);
	rock_cond.assign(n_res_blocks, 0);
	bc.resize(n_vars * n_bounds);
	bc_n.resize(n_vars * n_bounds);
	bc_ref.resize(n_vars * n_bounds);
	f.resize(n_vars * n_res_blocks);

	return 0;
}

int
conn_mesh::init_pm(std::vector<index_t>& block_m,
	std::vector<index_t>& block_p,
	std::vector<index_t>& _stencil,
	std::vector<index_t>& _st_offset,
	std::vector<value_t>& _tran,
	std::vector<value_t>& _rhs,
	std::vector<value_t>& _tran_biot,
	std::vector<value_t>& _rhs_biot,
	index_t _n_matrix, index_t _n_bounds, index_t _n_fracs)
{
	n_vars = 4;
	n_conns = block_m.size();

	one_way_block_m = block_m;
	one_way_block_p = block_p;
	one_way_stencil = _stencil;
	one_way_offset = _st_offset;
	one_way_tran = _tran;
	one_way_rhs = _rhs;
	one_way_tran_biot = _tran_biot;
	one_way_rhs_biot = _rhs_biot;

	n_matrix = _n_matrix;
	n_bounds = _n_bounds;
	n_fracs = _n_fracs;
	n_res_blocks = n_matrix + n_fracs;
	n_blocks = n_res_blocks;
	n_one_way_conns = n_conns;
	n_one_way_conns_res = n_conns;

	poro.resize(n_res_blocks);
	volume.resize(n_res_blocks);
	initial_state.resize(n_res_blocks * n_vars);
	ref_pressure.resize(n_res_blocks);
	ref_eps_vol.resize(n_matrix);
	displacement.resize(3 * n_res_blocks);
	op_num.assign(n_res_blocks, 0);
	depth.assign(n_res_blocks + n_bounds, 0);
	heat_capacity.assign(n_res_blocks, 0);
	rock_cond.assign(n_res_blocks, 0);
	rock_compressibility.resize(n_res_blocks);
	bc.resize(4 * n_bounds);
	bc_n.resize(4 * n_bounds);
	bc_ref.resize(4 * n_bounds);
	fault_conn_id.resize(n_fracs);

	return 0;
}

int
conn_mesh::init_pm(std::vector<index_t>& block_m,
	std::vector<index_t>& block_p,
	std::vector<index_t>& _stencil,
	std::vector<index_t>& _st_offset,
	std::vector<value_t>& _tran,
	std::vector<value_t>& _rhs,
	std::vector<value_t>& _tran_biot,
	std::vector<value_t>& _rhs_biot,
	std::vector<value_t>& _tran_face,
	std::vector<value_t>& _rhs_face,
	index_t _n_matrix, index_t _n_bounds, index_t _n_fracs)
{
	n_vars = 4;
	n_conns = block_m.size();

	one_way_block_m = block_m;
	one_way_block_p = block_p;
	one_way_stencil = _stencil;
	one_way_offset = _st_offset;
	one_way_tran = _tran;
	one_way_rhs = _rhs;
	one_way_tran_biot = _tran_biot;
	one_way_rhs_biot = _rhs_biot;
	one_way_tran_face = _tran_face;
	one_way_rhs_face = _rhs_face;

	n_matrix = _n_matrix;
	n_bounds = _n_bounds;
	n_fracs = _n_fracs;
	n_res_blocks = n_matrix + n_fracs;
	n_blocks = n_res_blocks;
	n_one_way_conns = n_conns;
	n_one_way_conns_res = n_conns;

	poro.resize(n_res_blocks);
	volume.resize(n_res_blocks);
	initial_state.resize(n_res_blocks * n_vars);
	ref_pressure.resize(n_res_blocks);
	ref_eps_vol.resize(n_matrix);
	displacement.resize(3 * n_res_blocks);
	op_num.assign(n_res_blocks, 0);
	depth.assign(n_res_blocks + n_bounds, 0);
	heat_capacity.assign(n_res_blocks, 0);
	rock_cond.assign(n_res_blocks, 0);
	rock_compressibility.resize(n_res_blocks);
	bc.resize(4 * n_bounds);
	bc_n.resize(4 * n_bounds);
	bc_ref.resize(4 * n_bounds);
	fault_conn_id.resize(n_fracs);

	return 0;
}

int
conn_mesh::init_pm_mech_discretizer(
  std::vector<index_t>& block_m,
  std::vector<index_t>& block_p,
  std::vector<index_t>& _stencil,
  std::vector<index_t>& _st_offset,
  std::vector<value_t>& _hooke, std::vector<value_t>& _hooke_rhs,
  std::vector<value_t>& _biot, std::vector<value_t>& _biot_rhs,
  std::vector<value_t>& _darcy, std::vector<value_t>& _darcy_rhs,
  std::vector<value_t>& _vol_strain, std::vector<value_t>& _vol_strain_rhs,
  index_t _n_matrix, index_t _n_bounds, index_t _n_fracs)
{
  n_vars = 4;
  n_dim = 3;
  n_conns = block_m.size();

  one_way_block_m = block_m;
  one_way_block_p = block_p;
  one_way_stencil = _stencil;
  one_way_offset = _st_offset;

  one_way_hooke = _hooke;			
  one_way_hooke_rhs = _hooke_rhs;
  one_way_biot = _biot;			
  one_way_biot_rhs = _biot_rhs;
  one_way_darcy = _darcy;			
  one_way_darcy_rhs = _darcy_rhs;
  one_way_vol_strain = _vol_strain;			
  one_way_vol_strain_rhs = _vol_strain_rhs;

  n_matrix = _n_matrix;
  n_bounds = _n_bounds;
  n_fracs = _n_fracs;
  n_res_blocks = n_matrix + n_fracs;
  n_blocks = n_res_blocks;
  n_one_way_conns = n_conns;
  n_one_way_conns_res = n_conns;

  poro.resize(n_res_blocks);
  volume.resize(n_res_blocks);
  initial_state.resize(n_res_blocks * n_vars);
  ref_pressure.resize(n_res_blocks, 0.0);
  ref_eps_vol.resize(n_matrix, 0.0);
  displacement.resize(3 * n_res_blocks);
  op_num.assign(n_res_blocks, 0);
  depth.assign(n_res_blocks + n_bounds, 0);
  heat_capacity.assign(n_res_blocks, 0);
  rock_cond.assign(n_res_blocks, 0);
  rock_compressibility.resize(n_res_blocks);
  bc.resize(n_vars * n_bounds);
  bc_n.resize(n_vars * n_bounds);
  bc_ref.resize(n_vars * n_bounds);
  fault_conn_id.resize(n_fracs);

  return 0;
}

int
conn_mesh::init_pme_mech_discretizer(
  std::vector<index_t>& block_m,
  std::vector<index_t>& block_p,
  std::vector<index_t>& _stencil,
  std::vector<index_t>& _st_offset,
  std::vector<value_t>& _hooke, std::vector<value_t>& _hooke_rhs,
  std::vector<value_t>& _biot, std::vector<value_t>& _biot_rhs,
  std::vector<value_t>& _darcy, std::vector<value_t>& _darcy_rhs,
  std::vector<value_t>& _vol_strain, std::vector<value_t>& _vol_strain_rhs,
  std::vector<value_t>& _thermal_traction,
  std::vector<value_t>& _fourier,
  index_t _n_matrix, index_t _n_bounds, index_t _n_fracs)
{
  n_vars = 5;
  n_dim = 3;
  n_conns = block_m.size();

  one_way_block_m = block_m;
  one_way_block_p = block_p;
  one_way_stencil = _stencil;
  one_way_offset = _st_offset;

  one_way_hooke = _hooke;
  one_way_hooke_rhs = _hooke_rhs;
  one_way_biot = _biot;
  one_way_biot_rhs = _biot_rhs;
  one_way_darcy = _darcy;
  one_way_darcy_rhs = _darcy_rhs;
  one_way_vol_strain = _vol_strain;
  one_way_vol_strain_rhs = _vol_strain_rhs;
  one_way_thermal_traction = _thermal_traction;
  one_way_fourier = _fourier;

  n_matrix = _n_matrix;
  n_bounds = _n_bounds;
  n_fracs = _n_fracs;
  n_res_blocks = n_matrix + n_fracs;
  n_blocks = n_res_blocks;
  n_one_way_conns = n_conns;
  n_one_way_conns_res = n_conns;

  poro.resize(n_res_blocks);
  volume.resize(n_res_blocks);
  initial_state.resize(n_res_blocks * n_vars);
  ref_pressure.resize(n_res_blocks, 0.0);
  ref_temperature.resize(n_res_blocks, 0.0);
  ref_eps_vol.resize(n_matrix, 0.0);
  displacement.resize(3 * n_res_blocks);
  op_num.assign(n_res_blocks, 0);
  depth.assign(n_res_blocks + n_bounds, 0);
  heat_capacity.assign(n_res_blocks, 0);
  rock_cond.assign(n_res_blocks, 0);
  th_poro.resize(n_res_blocks);
  rock_compressibility.resize(n_res_blocks);
  bc.resize(n_vars * n_bounds);
  bc_n.resize(n_vars * n_bounds);
  bc_ref.resize(n_vars * n_bounds);
  fault_conn_id.resize(n_fracs);

  return 0;
}

int
conn_mesh::add_conn (index_t block_m, index_t block_p, value_t trans, value_t transD)
{
  one_way_block_m.push_back (block_m);
  one_way_block_p.push_back (block_p);
  one_way_tran.push_back (trans);
  if (one_way_tranD.size())
    one_way_tranD.push_back (transD);

  n_conns++;
  return 0;
}

int
conn_mesh::add_conn_block(index_t block_m, index_t block_p, value_t trans, value_t transD, const uint8_t P_VAR)
{
  // for pm_discretizer output
  vector<value_t> tblock_pos(n_vars * n_vars, 0.0), tblock_neg(n_vars * n_vars, 0.0), 
				trhs(n_vars, 0.0), tblock_zero(n_vars * n_vars, 0.0);
  tblock_pos[P_VAR * n_vars + P_VAR] = trans;
  tblock_neg[P_VAR * n_vars + P_VAR] = -trans;
  // for mech_discretizer output
  vector<value_t> hooke_zeros(n_dim * n_vars, 0.0), vec_3d(n_dim, 0.0), vec_nvars(n_vars);

  // m -> p
  one_way_block_m.push_back(block_m);
  one_way_block_p.push_back(block_p);
  one_way_stencil.push_back(block_p);
  one_way_stencil.push_back(block_m);
  one_way_offset.push_back(one_way_stencil.size());

  if (one_way_tran.size())
  {
	one_way_tran.insert(one_way_tran.end(), tblock_neg.begin(), tblock_neg.end());
	one_way_tran.insert(one_way_tran.end(), tblock_pos.begin(), tblock_pos.end());
	one_way_rhs.insert(one_way_rhs.end(), trhs.begin(), trhs.end());
  }	
  if (one_way_flux.size()) one_way_flux.insert(one_way_flux.end(), trhs.begin(), trhs.end());
  if (one_way_gravity_flux.size()) one_way_gravity_flux.insert(one_way_gravity_flux.end(), trhs.begin(), trhs.end());

  if (one_way_tranD.size())
  {
	one_way_tranD.push_back(transD);
	one_way_tranD.push_back(-transD);
  }
  if (one_way_tran_heat_cond.size())
  {
	one_way_tran_heat_cond.push_back(transD);
	one_way_tran_heat_cond.push_back(-transD);
  }
  if (one_way_tran_biot.size())
  {
	one_way_tran_biot.insert(one_way_tran_biot.end(), tblock_zero.begin(), tblock_zero.end());
	one_way_tran_biot.insert(one_way_tran_biot.end(), tblock_zero.begin(), tblock_zero.end());
	one_way_rhs_biot.insert(one_way_rhs_biot.end(), trhs.begin(), trhs.end());
  }
  if (one_way_tran_face.size())
  {
	one_way_tran_face.insert(one_way_tran_face.end(), tblock_zero.begin(), tblock_zero.end());
	one_way_tran_face.insert(one_way_tran_face.end(), tblock_zero.begin(), tblock_zero.end());
	one_way_rhs_face.insert(one_way_rhs_face.end(), trhs.begin(), trhs.end());
  }
  if (one_way_darcy.size())
  {
	// darcy
	one_way_darcy.push_back(-trans);
	one_way_darcy.push_back(trans);
	one_way_darcy_rhs.push_back(0.0);
	// hooke
	one_way_hooke.insert(one_way_hooke.end(), hooke_zeros.begin(), hooke_zeros.end());
	one_way_hooke.insert(one_way_hooke.end(), hooke_zeros.begin(), hooke_zeros.end());
	one_way_hooke_rhs.insert(one_way_hooke_rhs.end(), vec_3d.begin(), vec_3d.end());
	// biot traction
	one_way_biot.insert(one_way_biot.end(), vec_3d.begin(), vec_3d.end());
	one_way_biot.insert(one_way_biot.end(), vec_3d.begin(), vec_3d.end());
	one_way_biot_rhs.insert(one_way_biot_rhs.end(), vec_3d.begin(), vec_3d.end());
	// biot volumetric strain
	one_way_vol_strain.insert(one_way_vol_strain.end(), vec_nvars.begin(), vec_nvars.end());
	one_way_vol_strain.insert(one_way_vol_strain.end(), vec_nvars.begin(), vec_nvars.end());
	one_way_vol_strain_rhs.push_back(0.0);
  }
  if (one_way_thermal_traction.size())
  {
	// thermal traction
	one_way_thermal_traction.insert(one_way_thermal_traction.end(), vec_3d.begin(), vec_3d.end());
	one_way_thermal_traction.insert(one_way_thermal_traction.end(), vec_3d.begin(), vec_3d.end());
	// Fourier (heat conduction)
	one_way_fourier.push_back(0.0);
	one_way_fourier.push_back(0.0);
  }
  n_conns++;
  n_links += 2;
  // p -> m
  one_way_block_m.push_back(block_p);
  one_way_block_p.push_back(block_m);
  one_way_stencil.push_back(block_m);
  one_way_stencil.push_back(block_p);
  one_way_offset.push_back(one_way_stencil.size());

  if (one_way_tran.size())
  {
	one_way_tran.insert(one_way_tran.end(), tblock_neg.begin(), tblock_neg.end());
	one_way_tran.insert(one_way_tran.end(), tblock_pos.begin(), tblock_pos.end());
	one_way_rhs.insert(one_way_rhs.end(), trhs.begin(), trhs.end());
  }
  if (one_way_flux.size()) one_way_flux.insert(one_way_flux.end(), trhs.begin(), trhs.end());
  if (one_way_gravity_flux.size()) one_way_gravity_flux.insert(one_way_gravity_flux.end(), trhs.begin(), trhs.end());

  if (one_way_tranD.size())
  {
	one_way_tranD.push_back(transD);
	one_way_tranD.push_back(-transD);
  }
  if (one_way_tran_heat_cond.size())
  {
	one_way_tran_heat_cond.push_back(transD);
	one_way_tran_heat_cond.push_back(-transD);
  }
  if (one_way_tran_biot.size())
  {
	one_way_tran_biot.insert(one_way_tran_biot.end(), tblock_zero.begin(), tblock_zero.end());
	one_way_tran_biot.insert(one_way_tran_biot.end(), tblock_zero.begin(), tblock_zero.end());
	one_way_rhs_biot.insert(one_way_rhs_biot.end(), trhs.begin(), trhs.end());
  }
  if (one_way_tran_face.size())
  {
	one_way_tran_face.insert(one_way_tran_face.end(), tblock_zero.begin(), tblock_zero.end());
	one_way_tran_face.insert(one_way_tran_face.end(), tblock_zero.begin(), tblock_zero.end());
	one_way_rhs_face.insert(one_way_rhs_face.end(), trhs.begin(), trhs.end());
  }
  if (one_way_darcy.size())
  {
	// darcy
	one_way_darcy.push_back(-trans);
	one_way_darcy.push_back(trans);
	one_way_darcy_rhs.push_back(0.0);
	// hooke
	one_way_hooke.insert(one_way_hooke.end(), hooke_zeros.begin(), hooke_zeros.end());
	one_way_hooke.insert(one_way_hooke.end(), hooke_zeros.begin(), hooke_zeros.end());
	one_way_hooke_rhs.insert(one_way_hooke_rhs.end(), vec_3d.begin(), vec_3d.end());
	// biot traction
	one_way_biot.insert(one_way_biot.end(), vec_3d.begin(), vec_3d.end());
	one_way_biot.insert(one_way_biot.end(), vec_3d.begin(), vec_3d.end());
	one_way_biot_rhs.insert(one_way_biot_rhs.end(), vec_3d.begin(), vec_3d.end());
	// biot volumetric strain
	one_way_vol_strain.insert(one_way_vol_strain.end(), vec_nvars.begin(), vec_nvars.end());
	one_way_vol_strain.insert(one_way_vol_strain.end(), vec_nvars.begin(), vec_nvars.end());
	one_way_vol_strain_rhs.push_back(0.0);
  }
  if (one_way_thermal_traction.size())
  {
	// thermal traction
	one_way_thermal_traction.insert(one_way_thermal_traction.end(), vec_3d.begin(), vec_3d.end());
	one_way_thermal_traction.insert(one_way_thermal_traction.end(), vec_3d.begin(), vec_3d.end());
	// Fourier (heat conduction)
	one_way_fourier.push_back(0.0);
	one_way_fourier.push_back(0.0);
  }

  n_conns++;
  n_links += 2;

  return 0;
}

int
conn_mesh::reverse_and_sort()
{
  int diff_trans = one_way_tranD.size();

  block_m.resize(2 * n_conns);
  block_p.resize(2 * n_conns);
  tran.resize(2 * n_conns);
  if (diff_trans)
    tranD.resize(2 * n_conns);
  grav_coef.assign(2 * n_conns, 0);

  one_way_to_conn_index_forward.resize(n_conns);
  one_way_to_conn_index_reverse.resize(n_conns);
  conn_index_to_one_way.resize(2 * n_conns);

  // Sort connection list + add reversed connections
  n_blocks = *(std::max_element(one_way_block_m.begin(), one_way_block_m.end())) + 1;
  n_blocks = std::max(n_blocks, *(std::max_element(one_way_block_p.begin(), one_way_block_p.end())) + 1);

  tmp_index.assign(n_blocks + 1, 0);
  cout << "Processing mesh: " << n_res_blocks << " reservoir blocks, " << n_blocks - n_res_blocks << " well blocks, " << n_conns << " connections\n";

  // run 1 - calc indices
  for (index_t j = 0; j < n_conns; ++j)
  {
    tmp_index[one_way_block_m[j] + 1] ++;  // 1 for direct connection
    tmp_index[one_way_block_p[j] + 1] ++;  // and 1 for reverse connection
  }
  // run 2 - sum indices
  for (index_t i = 0; i < n_blocks; ++i)
  {
    tmp_index[i + 1] += tmp_index[i]; // 1 for direct and 1 for reverse connection
  }
  // run 2 - set values and check
  index_t idx;
  index_t need_sort = 0;
  for (index_t j = 0; j < n_conns; ++j)
  {
    idx = tmp_index[one_way_block_m[j]]++;
    block_m[idx] = one_way_block_m[j];
    block_p[idx] = one_way_block_p[j];
    one_way_to_conn_index_forward[j] = idx;
    conn_index_to_one_way[idx] = j;

    tran[idx] = one_way_tran[j];
    if (diff_trans)
      tranD[idx] = one_way_tranD[j];


    if (!need_sort && idx > 0 && block_m[idx] == block_m[idx - 1] && block_p[idx] < block_p[idx - 1])
    {
      // columns were not sorted in initial file
      //cout << "Warning: block " << block_m[idx] << " has unsorted connections to " << block_p[idx] << " and " << block_p[idx - 1] << endl;
      need_sort = 1;
    }

    // reverse
    
    idx = tmp_index[one_way_block_p[j]]++;
    block_m[idx] = one_way_block_p[j];
    block_p[idx] = one_way_block_m[j];
    tran[idx] = one_way_tran[j];
    one_way_to_conn_index_reverse[j] = idx;
    conn_index_to_one_way[idx] = j;
    if (diff_trans)
      tranD[idx] = one_way_tranD[j];


    if (!need_sort &&block_m[idx] == block_m[idx - 1] && block_p[idx] < block_p[idx - 1])
    {
      // columns were not sorted in initial file
      //cout << "Warning: block " << block_m[idx] << " has unsorted connections to " << block_p[idx] << " and " << block_p[idx - 1] << endl;
      need_sort = 1;
    }
  }

  // run 3 - bubble sort columns within each block
  if (need_sort)
  {
    index_t j = 0;
    index_t i_tmp;
    value_t v_tmp;
    for (index_t i = 0; i < n_blocks; ++i)
    {
      for (; j < tmp_index[i] - 1; j++)
        for (index_t k = j + 1; k < tmp_index[i]; k++)
        {
          if (block_p[k] < block_p[j])
          {
            i_tmp = block_p[k];
            block_p[k] = block_p[j];
            block_p[j] = i_tmp;

            if (one_way_to_conn_index_forward[conn_index_to_one_way[j]] == j)
              one_way_to_conn_index_forward[conn_index_to_one_way[j]] = k;
            else
              one_way_to_conn_index_reverse[conn_index_to_one_way[j]] = k;

            if (one_way_to_conn_index_forward[conn_index_to_one_way[k]] == k)
              one_way_to_conn_index_forward[conn_index_to_one_way[k]] = j;
            else
              one_way_to_conn_index_reverse[conn_index_to_one_way[k]] = j;

            i_tmp = conn_index_to_one_way[k];
            conn_index_to_one_way[k] = conn_index_to_one_way[j];
            conn_index_to_one_way[j] = i_tmp;

            v_tmp = tran[k];
            tran[k] = tran[j];
            tran[j] = v_tmp;

            if (diff_trans)
            {
              v_tmp = tranD[k];
              tranD[k] = tranD[j];
              tranD[j] = v_tmp;
            }
          }
        }
      j++;
    }
  }
  // including wells now
  n_one_way_conns = n_conns;

  // with reversed connections
  n_conns *= 2;

  std::vector<value_t> test_t;
  std::vector<value_t> test_tD;

  get_res_tran(test_t, test_tD);
  set_res_tran(test_t, test_tD);

  return 0;
}

int
conn_mesh::reverse_and_sort_dvel()
{
	int diff_trans = one_way_tranD.size();

	block_m.resize(2 * n_conns);
	block_p.resize(2 * n_conns);
	tran.resize(2 * n_conns);
	tran_const.resize(2 * n_conns);
	if (diff_trans)
		tranD.resize(2 * n_conns);
	grav_coef.assign(2 * n_conns, 0);

	one_way_to_conn_index_forward.resize(n_conns);
	one_way_to_conn_index_reverse.resize(n_conns);
	conn_index_to_one_way.resize(2 * n_conns);

	// Sort connection list + add reversed connections
	n_blocks = *(std::max_element(one_way_block_m.begin(), one_way_block_m.end())) + 1;
	n_blocks = std::max(n_blocks, *(std::max_element(one_way_block_p.begin(), one_way_block_p.end())) + 1);

	tmp_index.assign(n_blocks + 1, 0);
	cout << "Processing mesh: " << n_res_blocks << " reservoir blocks, " << n_blocks - n_res_blocks << " well blocks, " << n_conns << " connections\n";

	// run 1 - calc indices
	for (index_t j = 0; j < n_conns; ++j)
	{
		tmp_index[one_way_block_m[j] + 1] ++;  // 1 for direct connection
		tmp_index[one_way_block_p[j] + 1] ++;  // and 1 for reverse connection
	}
	// run 2 - sum indices
	for (index_t i = 0; i < n_blocks; ++i)
	{
		tmp_index[i + 1] += tmp_index[i]; // 1 for direct and 1 for reverse connection
	}
	// run 2 - set values and check
	index_t idx;
	index_t need_sort = 0;
	for (index_t j = 0; j < n_conns; ++j)
	{
		idx = tmp_index[one_way_block_m[j]]++;
		block_m[idx] = one_way_block_m[j];
		block_p[idx] = one_way_block_p[j];
		one_way_to_conn_index_forward[j] = idx;
		conn_index_to_one_way[idx] = j;

		tran[idx] = one_way_tran[j];
		if (diff_trans)
			tranD[idx] = one_way_tranD[j];


		if (!need_sort && idx > 0 && block_m[idx] == block_m[idx - 1] && block_p[idx] < block_p[idx - 1])
		{
			// columns were not sorted in initial file
			//cout << "Warning: block " << block_m[idx] << " has unsorted connections to " << block_p[idx] << " and " << block_p[idx - 1] << endl;
			need_sort = 1;
		}

		// reverse

		idx = tmp_index[one_way_block_p[j]]++;
		block_m[idx] = one_way_block_p[j];
		block_p[idx] = one_way_block_m[j];
		tran[idx] = one_way_tran[j];
		one_way_to_conn_index_reverse[j] = idx;
		conn_index_to_one_way[idx] = j;
		if (diff_trans)
			tranD[idx] = one_way_tranD[j];


		if (!need_sort &&block_m[idx] == block_m[idx - 1] && block_p[idx] < block_p[idx - 1])
		{
			// columns were not sorted in initial file
			//cout << "Warning: block " << block_m[idx] << " has unsorted connections to " << block_p[idx] << " and " << block_p[idx - 1] << endl;
			need_sort = 1;
		}
	}

	// run 3 - bubble sort columns within each block
	if (need_sort)
	{
		index_t j = 0;
		index_t i_tmp;
		value_t v_tmp;
		for (index_t i = 0; i < n_blocks; ++i)
		{
			for (; j < tmp_index[i] - 1; j++)
				for (index_t k = j + 1; k < tmp_index[i]; k++)
				{
					if (block_p[k] < block_p[j])
					{
						i_tmp = block_p[k];
						block_p[k] = block_p[j];
						block_p[j] = i_tmp;

						if (one_way_to_conn_index_forward[conn_index_to_one_way[j]] == j)
							one_way_to_conn_index_forward[conn_index_to_one_way[j]] = k;
						else
							one_way_to_conn_index_reverse[conn_index_to_one_way[j]] = k;

						if (one_way_to_conn_index_forward[conn_index_to_one_way[k]] == k)
							one_way_to_conn_index_forward[conn_index_to_one_way[k]] = j;
						else
							one_way_to_conn_index_reverse[conn_index_to_one_way[k]] = j;

						i_tmp = conn_index_to_one_way[k];
						conn_index_to_one_way[k] = conn_index_to_one_way[j];
						conn_index_to_one_way[j] = i_tmp;

						v_tmp = tran[k];
						tran[k] = tran[j];
						tran[j] = v_tmp;

						if (diff_trans)
						{
							v_tmp = tranD[k];
							tranD[k] = tranD[j];
							tranD[j] = v_tmp;
						}
					}
				}
			j++;
		}
	}
	// including wells now
	n_one_way_conns = n_conns;

	// with reversed connections
	n_conns *= 2;

	std::vector<value_t> test_t;
	std::vector<value_t> test_tD;

	get_res_tran(test_t, test_tD);
	set_res_tran(test_t, test_tD);


	// renumerate correct velocity mapper
	// sorting one_way_block_m and one_way_block_p 
	int ctr = 0;
	for (index_t j = 0; j < n_conns; j++)
	{
		if (block_m[j] < block_p[j])
		{
			one_way_block_m[ctr] = block_m[j];
			one_way_block_p[ctr] = block_p[j];
			ctr += 1;
		}
	}
	// constructing again tmp_index
	tmp_index.assign(n_blocks + 1, 0);
	for (index_t j = 0; j < n_conns / 2; ++j)
	{
		tmp_index[one_way_block_m[j] + 1] ++;  // 1 for direct connection
		tmp_index[one_way_block_p[j] + 1] ++;  // and 1 for reverse connection
	}
	// run 2 - sum indices
	for (index_t i = 0; i < n_blocks; ++i)
	{
		tmp_index[i + 1] += tmp_index[i]; // 1 for direct and 1 for reverse connection
	}

	// fill velocity mapper
	index_t idx_velocity;
	for (index_t j = 0; j < n_conns / 2; ++j)
	{
		idx_velocity = tmp_index[one_way_block_m[j]]++;
		conn_index_to_one_way[idx_velocity] = j;
		//tran[idx_velocity] = 1;
		// reverse
		idx_velocity = tmp_index[one_way_block_p[j]]++;
		conn_index_to_one_way[idx_velocity] = j;
		// tran[idx_velocity] = 1;
	}

	return 0;
}

int
conn_mesh::reverse_and_sort_mpfa()
{
	const size_t diff_trans = one_way_tranD.size();
	const size_t thermal_trans = one_way_tran_heat_cond.size();

	cout << "Processing mesh: " << n_blocks << " reservoir blocks, " << n_blocks - n_res_blocks << " well blocks, " << n_bounds << " boundary segments, " << n_conns << " connections\n";
	n_matrix = n_blocks;
	cell_stencil.resize(n_blocks);
	struct ClosestCmp {
		index_t second;
		index_t conn_id;
		bool operator()(const ClosestCmp& a, const ClosestCmp& b)
		{
			return a.second < b.second;
		}
	};
	/// [n_blocks] map of connections [block_p, conn_idx] per block
	std::vector<std::vector<ClosestCmp>> t_idxs;
	t_idxs.resize(n_matrix + n_bounds);
	bool notBound;
	index_t n_two_way_stencil = 0;
	for (index_t i = 0; i < n_conns; i++)
	{
		notBound = (one_way_block_m[i] < n_blocks);

		if (notBound)
			n_two_way_stencil += one_way_offset[i + 1] - one_way_offset[i];

		for (index_t k = one_way_offset[i]; k < one_way_offset[i + 1]; k++)
		{
			if (one_way_stencil[k] < n_blocks)
			{
				if (notBound)
				{
					auto& cell1 = cell_stencil[one_way_block_m[i]];
					auto it1 = find(cell1.begin(), cell1.end(), one_way_stencil[k]);
					if (it1 == cell1.end())
						cell1.insert(lower_bound(cell1.begin(), cell1.end(), one_way_stencil[k]), one_way_stencil[k]);
				}
			}
		}
		// save connection's id per cell
		auto& t_m = t_idxs[one_way_block_m[i]];
		ClosestCmp s_m = { one_way_block_p[i], i };
		t_m.insert(lower_bound(t_m.begin(), t_m.end(), s_m, ClosestCmp()), s_m); // store in increasing order
	}

	// number of two-way connection (only from blocks)
	index_t n_two_way_conns = 0;
	// n_links is a number of non-zero elements in jacobian
	n_links = 0;
	for (index_t i = 0; i < n_blocks; ++i)
	{
		n_links += cell_stencil[i].size();
		n_two_way_conns += t_idxs[i].size();
	}

	// store two-way sorted connections
	grav_coef.assign(n_two_way_conns, 0);
	block_m.resize(n_two_way_conns);
	block_p.resize(n_two_way_conns);
	tran.resize(n_two_way_stencil);
	if (diff_trans)
		tranD.resize(n_two_way_stencil);
	if (thermal_trans) 
		tran_heat_cond.resize(n_two_way_stencil);
	stencil.resize(n_two_way_stencil);
	offset.resize(n_two_way_conns + 1);
	rhs.resize(n_two_way_conns);
	if (one_way_flux.size()) flux.resize(n_two_way_conns);
	if (one_way_gravity_flux.size()) grav_flux.resize(n_two_way_conns);
	index_t f_acc = 0, conn_id, conn_counter = 0, size;
	vector<index_t> ind;
	for (index_t i = 0; i < n_blocks; i++)
	{
		const auto& cur_cell = t_idxs[i];
		for (const auto& conn : cur_cell)
		{
			conn_id = conn.conn_id;
			block_m[conn_counter] = i;
			block_p[conn_counter] = conn.second;
			offset[conn_counter] = f_acc;
			rhs[conn_counter] = one_way_rhs[conn_id];
			if (one_way_flux.size()) flux[conn_counter] = one_way_flux[conn_id];
			if (one_way_gravity_flux.size()) grav_flux[conn_counter] = one_way_gravity_flux[conn_id];
			conn_counter++;

			size = one_way_offset[conn_id + 1] - one_way_offset[conn_id];
			ind.resize(size);
			iota(ind.begin(), ind.end(), one_way_offset[conn_id]);
			stable_sort(ind.begin(), ind.end(),
			    [this](index_t i1, index_t i2) {return one_way_stencil[i1] < one_way_stencil[i2]; });
			for (index_t j = 0; j < size; j++)
			{
			    stencil[j + f_acc] = one_way_stencil[ind[j]];
			    tran[j + f_acc] = one_way_tran[ind[j]];
				if (diff_trans)
					tranD[j + f_acc] = one_way_tranD[ind[j]];
				if (thermal_trans)
					tran_heat_cond[j + f_acc] = one_way_tran_heat_cond[ind[j]];
			}

			f_acc += size;
		}
	}
	offset.back() = f_acc;

	n_conns = n_two_way_conns;

	return 0;
}

int
conn_mesh::reverse_and_sort_mpsa()
{
    cout << "Processing mesh: " << n_blocks << " reservoir blocks including " << n_bounds << " boundary blocks, " << n_conns << " connections\n";

    cell_stencil.resize(n_blocks);
    struct ClosestCmp { 
        index_t second;
        index_t conn_id;
        bool operator()(const ClosestCmp& a, const ClosestCmp& b)
        { return a.second < b.second; }
    };
    /// [n_blocks] map of connections [block_p, conn_idx] per block
    std::vector<std::vector<ClosestCmp>> t_idxs;
    t_idxs.resize(n_blocks + n_bounds);
    bool notBound;
    index_t n_two_way_stencil = 0;
	std::vector<index_t>::iterator it1;
	ClosestCmp s_m;
    for (index_t i = 0; i < n_conns; i++)
    {
        notBound = (one_way_block_m[i] < n_blocks);

        if (notBound)
            n_two_way_stencil += one_way_offset[i + 1] - one_way_offset[i];

        for (index_t k = one_way_offset[i]; k < one_way_offset[i + 1]; k++)
        {
            if (one_way_stencil[k] < n_blocks)
            {
                if (notBound)
                {
                    auto& cell1 = cell_stencil[one_way_block_m[i]];
                    it1 = find(cell1.begin(), cell1.end(), one_way_stencil[k]);
                    if (it1 == cell1.end())
                        cell1.insert(lower_bound(cell1.begin(), cell1.end(), one_way_stencil[k]), one_way_stencil[k]);
                }
            }
        }
        // save connection's id per cell
        auto& t_m = t_idxs[one_way_block_m[i]];
		s_m.second = one_way_block_p[i];
		s_m.conn_id = i;
        t_m.insert(lower_bound(t_m.begin(), t_m.end(), s_m, ClosestCmp()), s_m); // store in increasing order
    }

    // number of two-way connection (only from blocks)
    index_t n_two_way_conns = 0;
    // n_links is a number of non-zero elements in jacobian
    n_links = 0;
    for (index_t i = 0; i < n_blocks; ++i)
    {
        n_links += cell_stencil[i].size();
        n_two_way_conns += t_idxs[i].size();
    }

    // store two-way sorted connections
    grav_coef.assign(n_two_way_conns, 0);
    block_m.resize(n_two_way_conns);
    block_p.resize(n_two_way_conns);
    //tran.resize(2 * one_way_tran.size());       // fluid
    //fstencil.resize(2 * one_way_fstencil.size());
    //fst_offset.resize(2 * n_conns + 1);
    tran.resize(n_two_way_stencil * n_vars * n_vars);     // stress
    stencil.resize(n_two_way_stencil);
    offset.resize(n_two_way_conns + 1);
	if (one_way_flux.size()) flux.resize(n_vars * n_two_way_conns);
    index_t f_acc = 0, s_acc = 0, conn_id, conn_counter = 0, size;
    vector<index_t> ind;
    for (index_t i = 0; i < n_blocks; i++)
    {
        const auto& cur_cell = t_idxs[i];
        for (const auto& conn: cur_cell)
        {
			conn_id = conn.conn_id;
            block_m[conn_counter] = i;
            block_p[conn_counter] = conn.second;
            //fst_offset[conn_counter] = f_acc;
            offset[conn_counter] = s_acc;
			if (one_way_flux.size()) copy_n(one_way_flux.begin() + n_vars * conn_id, n_vars, flux.begin() + n_vars * conn_counter);
			conn_counter++;
 
            //size = one_way_fst_offset[conn_id + 1] - one_way_fst_offset[conn_id];
            //ind.resize(size);
            //iota(ind.begin(), ind.end(), one_way_fst_offset[conn_id]);
            //stable_sort(ind.begin(), ind.end(),
            //    [this](index_t i1, index_t i2) {return one_way_fstencil[i1] < one_way_fstencil[i2]; });
            //for (index_t j = 0; j < size; j++)
            //{
            //    fstencil[j + f_acc] = one_way_fstencil[ind[j]];
            //    tran[j + f_acc] = one_way_tran[ind[j]];
            //}
            //f_acc += size;

            size = one_way_offset[conn_id + 1] - one_way_offset[conn_id];
            ind.resize(size);
            iota(ind.begin(), ind.end(), one_way_offset[conn_id]);
            stable_sort(ind.begin(), ind.end(),
                [this](index_t i1, index_t i2) {return one_way_stencil[i1] < one_way_stencil[i2]; });
            for (index_t j = 0; j < size; j++)
            {
                stencil[j + s_acc] = one_way_stencil[ind[j]];
                copy_n(one_way_tran.begin() + n_vars * n_vars * ind[j], n_vars * n_vars, tran.begin() + (j + s_acc) * n_vars * n_vars);
            }
            s_acc += size;
        }
    }
    //fst_offset.back() = f_acc;
    offset.back() = s_acc;

    n_conns = n_two_way_conns;

    return 0;
}

int
conn_mesh::reverse_and_sort_pm()
{
	cout << "Processing mesh: " << n_blocks << " reservoir blocks including " << n_conns << " connections\n";

	cell_stencil.resize(n_blocks);
	struct ClosestCmp {
		index_t second;
		index_t conn_id;
		bool operator()(const ClosestCmp& a, const ClosestCmp& b)
		{
			return a.second < b.second;
		}
	};
	/// [n_blocks] map of connections [block_p, conn_idx] per block
	std::vector<std::vector<ClosestCmp>> t_idxs;
	t_idxs.resize(n_blocks + n_bounds);
	bool notBound;
	index_t n_two_way_stencil = 0, k;
	std::vector<index_t>::iterator it1;
	for (index_t i = 0; i < n_conns; i++)
	{
		notBound = (one_way_block_m[i] < n_blocks);
		if (notBound)
			n_two_way_stencil += one_way_offset[i + 1] - one_way_offset[i];

		for (k = one_way_offset[i]; k < one_way_offset[i + 1]; k++)
		{
			if (one_way_stencil[k] < n_blocks)
			{
				if (notBound)
				{
					auto& cell1 = cell_stencil[one_way_block_m[i]];
					it1 = find(cell1.begin(), cell1.end(), one_way_stencil[k]);
					if (it1 == cell1.end())
						cell1.insert(lower_bound(cell1.begin(), cell1.end(), one_way_stencil[k]), one_way_stencil[k]);
				}
			}
		}
		// save connection's id per cell
		auto& t_m = t_idxs[one_way_block_m[i]];
		ClosestCmp s_m = { one_way_block_p[i], i };
		t_m.push_back(s_m);
		//t_m.insert(lower_bound(t_m.begin(), t_m.end(), s_m, ClosestCmp()), s_m); // store in increasing order
	}

	// number of two-way connection (only from blocks)
	index_t n_two_way_conns = 0;
	// n_links is a number of non-zero elements in jacobian
	n_links = 0;
	for (index_t i = 0; i < n_blocks; ++i)
	{
		n_links += cell_stencil[i].size();
		n_two_way_conns += t_idxs[i].size();
	}

	// save current conn ids
	for (index_t i = n_matrix; i < n_res_blocks; i++)
	{
		const auto& face1 = t_idxs[i][t_idxs[i].size() - 1];
		const auto& face2 = t_idxs[i][t_idxs[i].size() - 2];
		contact_cell_ids.push_back({ face1.second, face2.second });
	}
	std::vector<std::pair<index_t, index_t>>::iterator it;

	// store two-way sorted connections
	n_vars = 4;
	grav_coef.assign(n_two_way_conns, 0);
	block_m.resize(n_two_way_conns);
	block_p.resize(n_two_way_conns);
	tran.resize(n_two_way_stencil * n_vars * n_vars);			// mechanics
	tran_biot.resize(n_two_way_stencil * n_vars * n_vars);		// poromechanics
	tran_face.resize(n_two_way_stencil * n_vars * n_vars);		// unknowns on interfaces
	stencil.resize(n_two_way_stencil);
	offset.resize(n_two_way_conns + 1);
	sorted_conn_ids.resize(n_two_way_conns);
	sorted_stencil_ids.reserve(n_two_way_stencil);
	index_t f_acc = 0, s_acc = 0, conn_id, conn_counter = 0, size;
	vector<index_t> ind;
	const bool is_face_unknowns_delivered = (one_way_tran_face.size() == one_way_tran_biot.size());
	rhs.resize(n_two_way_conns * n_vars);
	rhs_biot.resize(n_two_way_conns * n_vars);
	if (one_way_rhs_face.size())
	  rhs_face.resize(n_two_way_conns * n_vars);
	
	for (index_t i = 0; i < n_blocks; i++)
	{
		const auto& cur_cell = t_idxs[i];
		for (const auto& conn : cur_cell)
		{
			block_m[conn_counter] = i;
			block_p[conn_counter] = conn.second;
			offset[conn_counter] = s_acc;
			conn_id = conn.conn_id;
			// copy rhs
			copy_n(one_way_rhs.begin() + n_vars * conn_id, n_vars, rhs.begin() + conn_counter * n_vars);
			copy_n(one_way_rhs_biot.begin() + n_vars * conn_id, n_vars, rhs_biot.begin() + conn_counter * n_vars);
			if (one_way_rhs_face.size())
			  copy_n(one_way_rhs_face.begin() + n_vars * conn_id, n_vars, rhs_face.begin() + conn_counter * n_vars);

			size = one_way_offset[conn_id + 1] - one_way_offset[conn_id];
			ind.resize(size);
			iota(ind.begin(), ind.end(), one_way_offset[conn_id]);
			stable_sort(ind.begin(), ind.end(),
				[this](index_t i1, index_t i2) {return one_way_stencil[i1] < one_way_stencil[i2]; });

			for (index_t j = 0; j < size; j++)
			{
				stencil[j + s_acc] = one_way_stencil[ind[j]];
				copy_n(one_way_tran.begin() + n_vars * n_vars * ind[j], n_vars*n_vars, tran.begin() + (j + s_acc) * n_vars * n_vars);
				copy_n(one_way_tran_biot.begin() + n_vars * n_vars * ind[j], n_vars*n_vars, tran_biot.begin() + (j + s_acc) * n_vars * n_vars);
				if (is_face_unknowns_delivered)
					copy_n(one_way_tran_face.begin() + n_vars * n_vars * ind[j], n_vars*n_vars, tran_face.begin() + (j + s_acc) * n_vars * n_vars);
				sorted_stencil_ids.push_back(ind[j]);
			}
			s_acc += size;

			// store sorted conn ids
			it = std::find(contact_cell_ids.begin(), contact_cell_ids.end(), std::make_pair(i, conn.second));
			if ( it != contact_cell_ids.end() )	
				fault_conn_id[std::distance(contact_cell_ids.begin(), it)].push_back(conn_counter);
			it = std::find(contact_cell_ids.begin(), contact_cell_ids.end(), std::make_pair(conn.second, i));
			if (it != contact_cell_ids.end())	
				fault_conn_id[std::distance(contact_cell_ids.begin(), it)].push_back(conn_counter);

			sorted_conn_ids[conn_counter] = conn_id;
			conn_counter++;
		}
	}
	//fst_offset.back() = f_acc;
	offset.back() = s_acc;

	n_conns = n_two_way_conns;

	// take stencil for contact into account
	index_t prev_num;
	for (index_t i = 0; i < n_fracs; i++)
	{
		const index_t cell_ids[] = { i + n_matrix, contact_cell_ids[i].first, contact_cell_ids[i].second };
		for (const auto& cell_id : cell_ids)
		{
			auto& cell1 = cell_stencil[cell_id];
			prev_num = cell1.size();
			for (const index_t& conn_id : fault_conn_id[i])
			{
				for (k = offset[conn_id]; k < offset[conn_id + 1]; k++)
				{
					if (stencil[k] < n_blocks)
					{
						it1 = find(cell1.begin(), cell1.end(), stencil[k]);
						if (it1 == cell1.end())
							cell1.insert(lower_bound(cell1.begin(), cell1.end(), stencil[k]), stencil[k]);
					}
				}
			}
			n_links += cell1.size() - prev_num;
		}
	}

	return 0;
}

int
conn_mesh::reverse_and_sort_pm_mech_discretizer()
{
  cout << "Processing mesh: " << n_blocks << " reservoir blocks including " << n_conns << " connections\n";

  cell_stencil.resize(n_blocks);
  struct ClosestCmp {
	index_t second;
	index_t conn_id;
	bool operator()(const ClosestCmp& a, const ClosestCmp& b)
	{
	  return a.second < b.second;
	}
  };
  /// [n_blocks] map of connections [block_p, conn_idx] per block
  std::vector<std::vector<ClosestCmp>> t_idxs;
  t_idxs.resize(n_blocks + n_bounds);
  bool notBound;
  index_t n_two_way_stencil = 0, k;
  std::vector<index_t>::iterator it1;
  for (index_t i = 0; i < n_conns; i++)
  {
	notBound = (one_way_block_m[i] < n_blocks);
	if (notBound)
	  n_two_way_stencil += one_way_offset[i + 1] - one_way_offset[i];

	for (k = one_way_offset[i]; k < one_way_offset[i + 1]; k++)
	{
	  if (one_way_stencil[k] < n_blocks)
	  {
		if (notBound)
		{
		  auto& cell1 = cell_stencil[one_way_block_m[i]];
		  it1 = find(cell1.begin(), cell1.end(), one_way_stencil[k]);
		  if (it1 == cell1.end())
			cell1.insert(lower_bound(cell1.begin(), cell1.end(), one_way_stencil[k]), one_way_stencil[k]);
		}
	  }
	}
	// save connection's id per cell
	auto& t_m = t_idxs[one_way_block_m[i]];
	ClosestCmp s_m = { one_way_block_p[i], i };
	t_m.push_back(s_m);
	//t_m.insert(lower_bound(t_m.begin(), t_m.end(), s_m, ClosestCmp()), s_m); // store in increasing order
  }

  // number of two-way connection (only from blocks)
  index_t n_two_way_conns = 0;
  // n_links is a number of non-zero elements in jacobian
  n_links = 0;
  for (index_t i = 0; i < n_blocks; ++i)
  {
	n_links += cell_stencil[i].size();
	n_two_way_conns += t_idxs[i].size();
  }

  // save current conn ids
  for (index_t i = n_matrix; i < n_res_blocks; i++)
  {
	const auto& face1 = t_idxs[i][t_idxs[i].size() - 1];
	const auto& face2 = t_idxs[i][t_idxs[i].size() - 2];
	contact_cell_ids.push_back({ face1.second, face2.second });
  }
  std::vector<std::pair<index_t, index_t>>::iterator it;

  // store two-way sorted connections
  grav_coef.assign(n_two_way_conns, 0);
  block_m.resize(n_two_way_conns);
  block_p.resize(n_two_way_conns);
  size_t n_hooke = n_dim * n_vars, n_biot = n_dim, n_darcy = 1, n_vol_strain = n_vars;
  hooke_tran.resize(n_two_way_stencil * n_hooke);
  biot_tran.resize(n_two_way_stencil * n_biot);
  darcy_tran.resize(n_two_way_stencil * n_darcy);
  vol_strain_tran.resize(n_two_way_stencil * n_vol_strain);
  hooke_rhs.resize(n_two_way_conns * n_dim);
  biot_rhs.resize(n_two_way_conns * n_dim);
  darcy_rhs.resize(n_two_way_conns);
  vol_strain_rhs.resize(n_two_way_conns);
  stencil.resize(n_two_way_stencil);
  offset.resize(n_two_way_conns + 1);
  sorted_conn_ids.resize(n_two_way_conns);
  sorted_stencil_ids.reserve(n_two_way_stencil);

  index_t f_acc = 0, s_acc = 0, conn_id, conn_counter = 0, size;
  vector<index_t> ind;
  for (index_t i = 0; i < n_blocks; i++)
  {
	const auto& cur_cell = t_idxs[i];
	for (const auto& conn : cur_cell)
	{
	  block_m[conn_counter] = i;
	  block_p[conn_counter] = conn.second;
	  offset[conn_counter] = s_acc;
	  conn_id = conn.conn_id;
	  
	  copy_n(one_way_hooke_rhs.begin() + conn_id * n_dim, n_dim, hooke_rhs.begin() + conn_counter * n_dim);
	  copy_n(one_way_biot_rhs.begin() + conn_id * n_dim, n_dim, biot_rhs.begin() + conn_counter * n_dim);
	  darcy_rhs[conn_counter] = one_way_darcy_rhs[conn_id];
	  vol_strain_rhs[conn_counter] = one_way_vol_strain_rhs[conn_id];

	  size = one_way_offset[conn_id + 1] - one_way_offset[conn_id];
	  ind.resize(size);
	  iota(ind.begin(), ind.end(), one_way_offset[conn_id]);
	  stable_sort(ind.begin(), ind.end(),
		[this](index_t i1, index_t i2) {return one_way_stencil[i1] < one_way_stencil[i2]; });

	  for (index_t j = 0; j < size; j++)
	  {
		stencil[j + s_acc] = one_way_stencil[ind[j]];
		copy_n(one_way_hooke.begin() + n_hooke * ind[j], n_hooke, hooke_tran.begin() + (j + s_acc) * n_hooke);
		copy_n(one_way_biot.begin() + n_biot * ind[j], n_biot, biot_tran.begin() + (j + s_acc) * n_biot);
		copy_n(one_way_darcy.begin() + n_darcy * ind[j], n_darcy, darcy_tran.begin() + (j + s_acc) * n_darcy);
		copy_n(one_way_vol_strain.begin() + n_vol_strain * ind[j], n_vol_strain, vol_strain_tran.begin() + (j + s_acc) * n_vol_strain);
		sorted_stencil_ids.push_back(ind[j]);
	  }
	  s_acc += size;

	  // store sorted conn ids
	  it = std::find(contact_cell_ids.begin(), contact_cell_ids.end(), std::make_pair(i, conn.second));
	  if (it != contact_cell_ids.end())
		fault_conn_id[std::distance(contact_cell_ids.begin(), it)].push_back(conn_counter);
	  it = std::find(contact_cell_ids.begin(), contact_cell_ids.end(), std::make_pair(conn.second, i));
	  if (it != contact_cell_ids.end())
		fault_conn_id[std::distance(contact_cell_ids.begin(), it)].push_back(conn_counter);

	  sorted_conn_ids[conn_counter] = conn_id;
	  conn_counter++;
	}
  }
  offset.back() = s_acc;

  n_conns = n_two_way_conns;

  // take stencil for contact into account
  index_t prev_num;
  for (index_t i = 0; i < n_fracs; i++)
  {
	const index_t cell_ids[] = { i + n_matrix, contact_cell_ids[i].first, contact_cell_ids[i].second };
	for (const auto& cell_id : cell_ids)
	{
	  auto& cell1 = cell_stencil[cell_id];
	  prev_num = cell1.size();
	  for (const index_t& conn_id : fault_conn_id[i])
	  {
		for (k = offset[conn_id]; k < offset[conn_id + 1]; k++)
		{
		  if (stencil[k] < n_blocks)
		  {
			it1 = find(cell1.begin(), cell1.end(), stencil[k]);
			if (it1 == cell1.end())
			  cell1.insert(lower_bound(cell1.begin(), cell1.end(), stencil[k]), stencil[k]);
		  }
		}
	  }
	  n_links += cell1.size() - prev_num;
	}
  }

  return 0;
}

int
conn_mesh::reverse_and_sort_pme_mech_discretizer()
{
  cout << "Processing mesh: " << n_blocks << " reservoir blocks including " << n_conns << " connections\n";

  cell_stencil.resize(n_blocks);
  struct ClosestCmp {
	index_t second;
	index_t conn_id;
	bool operator()(const ClosestCmp& a, const ClosestCmp& b)
	{
	  return a.second < b.second;
	}
  };
  /// [n_blocks] map of connections [block_p, conn_idx] per block
  std::vector<std::vector<ClosestCmp>> t_idxs;
  t_idxs.resize(n_blocks + n_bounds);
  bool notBound;
  index_t n_two_way_stencil = 0, k;
  std::vector<index_t>::iterator it1;
  for (index_t i = 0; i < n_conns; i++)
  {
	notBound = (one_way_block_m[i] < n_blocks);
	if (notBound)
	  n_two_way_stencil += one_way_offset[i + 1] - one_way_offset[i];

	for (k = one_way_offset[i]; k < one_way_offset[i + 1]; k++)
	{
	  if (one_way_stencil[k] < n_blocks)
	  {
		if (notBound)
		{
		  auto& cell1 = cell_stencil[one_way_block_m[i]];
		  it1 = find(cell1.begin(), cell1.end(), one_way_stencil[k]);
		  if (it1 == cell1.end())
			cell1.insert(lower_bound(cell1.begin(), cell1.end(), one_way_stencil[k]), one_way_stencil[k]);
		}
	  }
	}
	// save connection's id per cell
	auto& t_m = t_idxs[one_way_block_m[i]];
	ClosestCmp s_m = { one_way_block_p[i], i };
	t_m.push_back(s_m);
	//t_m.insert(lower_bound(t_m.begin(), t_m.end(), s_m, ClosestCmp()), s_m); // store in increasing order
  }

  // number of two-way connection (only from blocks)
  index_t n_two_way_conns = 0;
  // n_links is a number of non-zero elements in jacobian
  n_links = 0;
  for (index_t i = 0; i < n_blocks; ++i)
  {
	n_links += cell_stencil[i].size();
	n_two_way_conns += t_idxs[i].size();
  }

  // save current conn ids
  for (index_t i = n_matrix; i < n_res_blocks; i++)
  {
	const auto& face1 = t_idxs[i][t_idxs[i].size() - 1];
	const auto& face2 = t_idxs[i][t_idxs[i].size() - 2];
	contact_cell_ids.push_back({ face1.second, face2.second });
  }
  std::vector<std::pair<index_t, index_t>>::iterator it;

  // store two-way sorted connections
  grav_coef.assign(n_two_way_conns, 0);
  block_m.resize(n_two_way_conns);
  block_p.resize(n_two_way_conns);
  size_t n_hooke = n_dim * n_vars, n_biot = n_dim, n_darcy = 1, n_vol_strain = n_vars, n_thermal = n_dim, n_fourier = 1;
  hooke_tran.resize(n_two_way_stencil * n_hooke);
  biot_tran.resize(n_two_way_stencil * n_biot);
  darcy_tran.resize(n_two_way_stencil * n_darcy);
  vol_strain_tran.resize(n_two_way_stencil * n_vol_strain);
  thermal_traction_tran.resize(n_two_way_stencil * n_thermal);
  fourier_tran.resize(n_two_way_stencil * n_fourier);
  hooke_rhs.resize(n_two_way_conns * n_dim);
  biot_rhs.resize(n_two_way_conns * n_dim);
  darcy_rhs.resize(n_two_way_conns);
  vol_strain_rhs.resize(n_two_way_conns);
  stencil.resize(n_two_way_stencil);
  offset.resize(n_two_way_conns + 1);
  sorted_conn_ids.resize(n_two_way_conns);
  sorted_stencil_ids.reserve(n_two_way_stencil);

  index_t f_acc = 0, s_acc = 0, conn_id, conn_counter = 0, size;
  vector<index_t> ind;
  for (index_t i = 0; i < n_blocks; i++)
  {
	const auto& cur_cell = t_idxs[i];
	for (const auto& conn : cur_cell)
	{
	  block_m[conn_counter] = i;
	  block_p[conn_counter] = conn.second;
	  offset[conn_counter] = s_acc;
	  conn_id = conn.conn_id;

	  copy_n(one_way_hooke_rhs.begin() + conn_id * n_dim, n_dim, hooke_rhs.begin() + conn_counter * n_dim);
	  copy_n(one_way_biot_rhs.begin() + conn_id * n_dim, n_dim, biot_rhs.begin() + conn_counter * n_dim);
	  darcy_rhs[conn_counter] = one_way_darcy_rhs[conn_id];
	  vol_strain_rhs[conn_counter] = one_way_vol_strain_rhs[conn_id];

	  size = one_way_offset[conn_id + 1] - one_way_offset[conn_id];
	  ind.resize(size);
	  iota(ind.begin(), ind.end(), one_way_offset[conn_id]);
	  stable_sort(ind.begin(), ind.end(),
		[this](index_t i1, index_t i2) {return one_way_stencil[i1] < one_way_stencil[i2]; });

	  for (index_t j = 0; j < size; j++)
	  {
		stencil[j + s_acc] = one_way_stencil[ind[j]];
		copy_n(one_way_hooke.begin() + n_hooke * ind[j], n_hooke, hooke_tran.begin() + (j + s_acc) * n_hooke);
		copy_n(one_way_biot.begin() + n_biot * ind[j], n_biot, biot_tran.begin() + (j + s_acc) * n_biot);
		copy_n(one_way_darcy.begin() + n_darcy * ind[j], n_darcy, darcy_tran.begin() + (j + s_acc) * n_darcy);
		copy_n(one_way_vol_strain.begin() + n_vol_strain * ind[j], n_vol_strain, vol_strain_tran.begin() + (j + s_acc) * n_vol_strain);
		copy_n(one_way_thermal_traction.begin() + n_thermal * ind[j], n_thermal, thermal_traction_tran.begin() + (j + s_acc) * n_thermal);
		copy_n(one_way_fourier.begin() + n_fourier * ind[j], n_fourier, fourier_tran.begin() + (j + s_acc) * n_fourier);
		sorted_stencil_ids.push_back(ind[j]);
	  }
	  s_acc += size;

	  // store sorted conn ids
	  it = std::find(contact_cell_ids.begin(), contact_cell_ids.end(), std::make_pair(i, conn.second));
	  if (it != contact_cell_ids.end())
		fault_conn_id[std::distance(contact_cell_ids.begin(), it)].push_back(conn_counter);
	  it = std::find(contact_cell_ids.begin(), contact_cell_ids.end(), std::make_pair(conn.second, i));
	  if (it != contact_cell_ids.end())
		fault_conn_id[std::distance(contact_cell_ids.begin(), it)].push_back(conn_counter);

	  sorted_conn_ids[conn_counter] = conn_id;
	  conn_counter++;
	}
  }
  offset.back() = s_acc;

  n_conns = n_two_way_conns;

  // take stencil for contact into account
  index_t prev_num;
  for (index_t i = 0; i < n_fracs; i++)
  {
	const index_t cell_ids[] = { i + n_matrix, contact_cell_ids[i].first, contact_cell_ids[i].second };
	for (const auto& cell_id : cell_ids)
	{
	  auto& cell1 = cell_stencil[cell_id];
	  prev_num = cell1.size();
	  for (const index_t& conn_id : fault_conn_id[i])
	  {
		for (k = offset[conn_id]; k < offset[conn_id + 1]; k++)
		{
		  if (stencil[k] < n_blocks)
		  {
			it1 = find(cell1.begin(), cell1.end(), stencil[k]);
			if (it1 == cell1.end())
			  cell1.insert(lower_bound(cell1.begin(), cell1.end(), stencil[k]), stencil[k]);
		  }
		}
	  }
	  n_links += cell1.size() - prev_num;
	}
  }

  return 0;
}

int
conn_mesh::init_grav_coef(value_t grav_const)
{

  for (index_t j = 0; j < n_conns; ++j)
  {
    grav_coef[j] = (depth[block_m[j]] - depth[block_p[j]]) * grav_const;
  }

  return 0;
}

int conn_mesh::get_res_tran(std::vector<value_t>& res_tran, std::vector<value_t>& res_tranD)
{
  res_tran.resize(n_one_way_conns_res);
  
  
  for (index_t j = 0; j < n_one_way_conns_res; ++j)
  {
    res_tran[j] = tran[one_way_to_conn_index_forward[j]];
  }

  if (tranD.size())
  {
    res_tranD.resize(n_one_way_conns_res);

    for (index_t j = 0; j < n_one_way_conns_res; ++j)
    {
      res_tranD[j] = tranD[one_way_to_conn_index_forward[j]];
    }
  }

  return 0;
}

int conn_mesh::set_res_tran(std::vector<value_t>& res_tran, std::vector<value_t>& res_tranD)
{

  for (index_t j = 0; j < n_one_way_conns_res; ++j)
  {
    tran[one_way_to_conn_index_forward[j]] = res_tran[j];
    tran[one_way_to_conn_index_reverse[j]] = res_tran[j];
  }

  if (tranD.size())
  {
    for (index_t j = 0; j < n_one_way_conns_res; ++j)
    {
      tranD[one_way_to_conn_index_forward[j]] = res_tranD[j];
      tranD[one_way_to_conn_index_reverse[j]] = res_tranD[j];
    }
  }

  return 0;
}

int conn_mesh::get_wells_tran(std::vector<value_t>& well_tran)
{
  well_tran.resize(n_perfs);
  index_t i = 0;

  for (index_t j = 0; j < n_conns; ++j)
  {
    if ((block_m[j] < n_res_blocks) && (block_p[j] > n_res_blocks))
    {
      // this is a well perforation in forward direction (reservoir->well)
      well_tran[i++] = tran[j];
    }
  }
  
  return 0;
}

int conn_mesh::set_wells_tran(std::vector<value_t>& well_tran)
{
  index_t i = 0;

  for (index_t j = 0; j < n_conns; ++j)
  {
    if ((block_m[j] < n_res_blocks) && (block_p[j] > n_res_blocks))
    {
      // update correspondent forward and reverse connections
      tran[one_way_to_conn_index_forward[conn_index_to_one_way[j]]] = well_tran[i];
      tran[one_way_to_conn_index_reverse[conn_index_to_one_way[j]]] = well_tran[i];
      
      i++;
    }
  }

  return 0;
}

int conn_mesh::add_wells(std::vector<ms_well *> &wells)
{
  index_t well_head_idx = n_res_blocks;
  n_perfs = 0;

  // Wells are modeled as a 1D sequence of small grid blocks (W-blocks) representing segments, 
  // which are connected to the reservoir. In addition, there is one more grid block (H-block)
  // per well, which is at the top, connected to the first well segment,
  // served as a container for well control equations.

  //  Add well connections
  for (index_t iw = 0; iw < wells.size(); iw++)
  {
    wells[iw]->well_head_idx = well_head_idx; // well head
    wells[iw]->well_body_idx = well_head_idx + 1; // well body
    
    index_t n_segments = 0;
    // connections between well segments and reservoir
    for (index_t p = 0; p < wells[iw]->perforations.size(); p++)
    {
      index_t i_w, i_r;
      value_t wi, wid;
      std::tie(i_w, i_r, wi, wid) = wells[iw]->perforations[p];
      add_conn(i_w + well_head_idx + 1, i_r, wi, wid);
      n_perfs++;
      n_segments = max(n_segments, i_w + 1);
    }
    // connections between segments
    for (index_t p = 0; p < n_segments; p++)
    {
      add_conn(well_head_idx + p, well_head_idx + p + 1, wells[iw]->segment_transmissibility, 0); // connection between them
    }
    well_head_idx += n_segments + 1;
    wells[iw]->n_segments = n_segments;
  }

  // connect_segments(wells[0], wells[1], wells[0]->n_segments, wells[1]->n_segments);

  // Resize mesh arrays by number of well blocks and head blocks (one per well)
  n_blocks = well_head_idx;
  volume.resize(n_blocks);
  poro.resize(n_blocks);
  initial_state.resize(n_blocks * n_vars);
  op_num.resize(n_blocks);
  depth.resize(n_blocks + n_bounds);

  heat_capacity.resize(n_blocks);
  rock_cond.resize(n_blocks + n_bounds);
  mob_multiplier.resize(2 * n_blocks);

  for (index_t iw = 0; iw < wells.size(); iw++)
  {
    // depth of the well head block - well controls work at this depth
    depth[wells[iw]->well_head_idx] = wells[iw]->well_head_depth;
    for (index_t p = 0; p < wells[iw]->n_segments + 1; p++)
    {
      volume[wells[iw]->well_head_idx + p] = wells[iw]->segment_volume;
      poro[wells[iw]->well_head_idx + p] = 1;
      op_num[wells[iw]->well_head_idx + p] = 0;
      heat_capacity[wells[iw]->well_head_idx + p] = 0;
	  mob_multiplier[wells[iw]->well_head_idx * 2 + p * 2] = 1;
	  mob_multiplier[wells[iw]->well_head_idx * 2 + p * 2 + 1] = 1;
      if (p > 0)// p==0 is a ghost cell for the well treatment
      {
        int r_i = std::get<1>(wells[iw]->perforations[p - 1]);
        int w_i = wells[iw]->well_head_idx + p;
        // copy properties for the well blocks from the reservoir blocks
        rock_cond[w_i] = rock_cond[r_i];
        // depth of well segments
        depth[wells[iw]->well_head_idx + p] = wells[iw]->well_body_depth + (p - 1) * wells[iw]->segment_depth_increment;
      }
    }
  }

  return 0;
}

// segment_transmissibility of the first well used
int conn_mesh::connect_segments(ms_well* well1, ms_well* well2, int iseg1, int iseg2, int verbose)
{
	if (verbose)
		cout << "Added connection between well " << well1->name << " head idx=" << well1->well_head_idx << " segment idx="  << iseg1 << " and well " <<
																								well2->name << " head idx=" << well2->well_head_idx << " segment idx="  << iseg2 << endl;
	add_conn(well1->well_head_idx + iseg1, well2->well_head_idx + iseg2, well1->segment_transmissibility, 0);
	return 0;
}

int conn_mesh::add_wells_mpfa(std::vector<ms_well *> &wells, const uint8_t P_VAR)
{
	index_t well_head_idx = n_res_blocks;
	n_perfs = 0;

	// calculate number of additional unknowns will be added
	index_t dofs_num = 0, n_segments, i_w, i_r;
	value_t wi, wid;
	for (index_t iw = 0; iw < wells.size(); iw++)
	{
		n_segments = 0;
		for (index_t p = 0; p < wells[iw]->perforations.size(); p++)
		{
			std::tie(i_w, i_r, wi, wid) = wells[iw]->perforations[p];
			n_segments = max(n_segments, i_w + 1);
		}
		dofs_num += n_segments + 1;
	}
	// shift boundary indices & arrays with boundary properties
	shift_boundary_ids_mpfa(dofs_num);

	// Resize mesh arrays by number of well blocks and head blocks (one per well)
	n_blocks += dofs_num;
	volume.resize(volume.size() + dofs_num);
	poro.resize(poro.size() + dofs_num);
	initial_state.resize(initial_state.size() + dofs_num * n_vars);
	op_num.resize(op_num.size() + dofs_num);
	//depth.resize(depth.size() + dofs_num);
	if (displacement.size())
		displacement.resize(displacement.size() + 3 * dofs_num);
	if (ref_pressure.size())
	  ref_pressure.resize(ref_pressure.size() + dofs_num);
	if (ref_temperature.size())
	  ref_temperature.resize(ref_temperature.size() + dofs_num);
	if (th_poro.size())
	  th_poro.resize(th_poro.size() + dofs_num);
	heat_capacity.resize(heat_capacity.size() + dofs_num);
	//rock_cond.resize(n_blocks + n_bounds);

	// Wells are modeled as a 1D sequence of small grid blocks (W-blocks) representing segments, 
	// which are connected to the reservoir. In addition, there is one more grid block (H-block)
	// per well, which is at the top, connected to the first well segment,
	// served as a container for well control equations.

	//  Add well connections
	for (index_t iw = 0; iw < wells.size(); iw++)
	{
		wells[iw]->well_head_idx = well_head_idx; // well head
		wells[iw]->well_body_idx = well_head_idx + 1; // well body

		n_segments = 0;
		// connections between well segments and reservoir
		for (index_t p = 0; p < wells[iw]->perforations.size(); p++)
		{
			std::tie(i_w, i_r, wi, wid) = wells[iw]->perforations[p];
			add_conn_block(i_w + well_head_idx + 1, i_r, wi, wid, P_VAR);
			n_perfs++;
			n_segments = max(n_segments, i_w + 1);
		}
		// connections between segments
		for (index_t p = 0; p < n_segments; p++)
		{
			add_conn_block(well_head_idx + p, well_head_idx + p + 1, wells[iw]->segment_transmissibility, 0, P_VAR); // connection between them
		}
		well_head_idx += n_segments + 1;
		wells[iw]->n_segments = n_segments;
	}

	for (index_t iw = 0; iw < wells.size(); iw++)
	{
		// depth of the well head block - well controls work at this depth
		depth[wells[iw]->well_head_idx] = wells[iw]->well_head_depth;
		for (index_t p = 0; p < wells[iw]->n_segments + 1; p++)
		{
			volume[wells[iw]->well_head_idx + p] = wells[iw]->segment_volume;
			poro[wells[iw]->well_head_idx + p] = 1;
			if (th_poro.size())
			  th_poro[wells[iw]->well_head_idx + p] = 0;
			op_num[wells[iw]->well_head_idx + p] = 0;
			heat_capacity[wells[iw]->well_head_idx + p] = 0;
			rock_cond[wells[iw]->well_head_idx + p] = 0;
			if (p > 0)
			{
				int r_i = std::get<1>(wells[iw]->perforations[p - 1]);
				int w_i = wells[iw]->well_head_idx + p;
				// depth of well segments
				depth[wells[iw]->well_head_idx + p] = wells[iw]->well_body_depth + (p - 1) * wells[iw]->segment_depth_increment;
			}
		}
	}

	return 0;
}

void conn_mesh::shift_boundary_ids_mpfa(const int n)
{
	for (int conn_id = 0; conn_id < n_conns; conn_id++)
	{
		if (one_way_block_p[conn_id] >= n_blocks)
			one_way_block_p[conn_id] += n;

		for (int conn_st_id = one_way_offset[conn_id]; conn_st_id < one_way_offset[conn_id + 1]; conn_st_id++)
		{
			if (one_way_stencil[conn_st_id] >= n_blocks)
				one_way_stencil[conn_st_id] += n;
		}
	}

	// shift properties
	depth.insert(depth.begin() + n_blocks, n, 0.0);
	rock_cond.insert(rock_cond.begin() + n_blocks, n, 0.0);
}
