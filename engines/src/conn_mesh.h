//*************************************************************************
//    Copyright (c) 2018
//            Mark Khait         M.Khait@tudelft.nl
//            Denis Voskov    D.V.Voskov@tudelft.nl
//    Delft University of Technology, the Netherlands
//
//    This file is part of the Delft Advanced Research Terra Simulator (DARTS)
//
//    DARTS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as 
//    published by the Free Software Foundation, either version 3 of the 
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public 
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************

#ifndef CONN_MESH_H
#define CONN_MESH_H

#include <vector>
#include <string>
#include "globals.h"
#include "ms_well.h"
#include "block_well.h"


/// This class defines mesh and corresponding arrays
class conn_mesh
{

public:
  conn_mesh () {};                                          // default constructor


  int init_poro (std::string poro_filename);                // init mesh porosity by reading PORO keyword file

  int init_grav_coef(value_t grav_const = 9.80665e-5);      // discretize ms wells into reservoir

  int save_keyword_compressed (std::string filename, 
    std::string keyword, value_t* data, index_t length);    // dump keyword in compressed (like 2*50 3*100) format

  int get_res_tran(std::vector<value_t> &res_tran, 
                   std::vector<value_t> &res_tranD);        // get trans for reservoir part

  int set_res_tran(std::vector<value_t> &res_tran,
                   std::vector<value_t> &res_tranD);        // set trans for reservoir part

  int get_wells_tran(std::vector<value_t> &wells_tran);     // get trans for wells part (well indexes)

  int set_wells_tran(std::vector<value_t> &wells_tran);     // set trans for wells part (well indexes)

  int save_volume (std::string filename);                   // dump VOLUME
  int save_poro (std::string filename);                     // dump PORO

  int save_pressure (std::string filename);                 // dump PRESSURE
  int save_zmf (std::string filename);                      // dump ZMF
  int save_temperature (std::string filename);              // dump TEMPERATURE
  int save_enthalpy (std::string filename);                 // dump ENTHALPY

  int save_wells (std::string filename, 
    std::vector <block_well> &well_list, 
    sim_params &params);                                    // dump wells and timestepping params

public:
  index_t n_res_blocks;                                     // number of reservoir blocks in the mesh            (R)
  index_t n_res_well_blocks;                                // number of reservoir and well blocks in the mesh   (R+W)
  index_t n_blocks;                                         // number of all blocks in the mesh including ghost  (R+W+G)
  index_t n_conns;                                          // number of connections between the blocks
  index_t n_perfs;                                          // number of well perforations
  index_t n_matrix;                                         // number of matrix blocks
  index_t n_bounds = 0;                                     // number of boundary blocks
  index_t n_fracs = 0;                                     // number of fracture blocks

  // mapping from one-way list to actual list (forward connections) 
  std::vector <index_t> one_way_to_conn_index_forward;
  // mapping from one-way list to actual list (reversed connections) 
  std::vector <index_t> one_way_to_conn_index_reverse;
  // mapping actual list to one_way
  std::vector <index_t> conn_index_to_one_way;


  /** @defgroup Connection_mesh
   *  Parameters and methods in mesh class exposed to Python
   *  @{
   */

  /// @brief init mesh by reading TPFACONNS keyword file
  int init(std::string conn2p_filename);                   

  /// @brief init mesh by reading array of left/right neighbours 
  int init(std::vector<index_t> &block_m,
    std::vector<index_t> &block_p,
    std::vector<value_t> &tran,
    std::vector<value_t> &tranD);                    

  /// @brief init mesh by reading MPFA connections
  int init_mpfa(std::vector<index_t>& block_m,
      std::vector<index_t>& block_p,
      std::vector<index_t>& _fstencil,
      std::vector<index_t>& _fst_offset,
      std::vector<value_t>& _ftran,
	  std::vector<value_t>& _rhs,
	  index_t _n_matrix, index_t _n_bounds);

  /// @brief init mesh by reading MPFA connections for both fluid mass and heat fluxes
  int init_mpfa_e(std::vector<index_t>& block_m,
	  std::vector<index_t>& block_p,
	  std::vector<index_t>& _fstencil,
	  std::vector<index_t>& _fst_offset,
	  std::vector<value_t>& _ftran,
	  std::vector<value_t>& _htran,
	  std::vector<value_t>& _rhs,
	  index_t _n_matrix, index_t _n_bounds, index_t _n_fracs);

  /// @brief init mesh by reading MPFA connections with provided flux
  int init_mpfa(std::vector<index_t>& block_m,
	  std::vector<index_t>& block_p,
	  std::vector<index_t>& _fstencil,
	  std::vector<index_t>& _fst_offset,
	  std::vector<value_t>& _ftran,
	  std::vector<value_t>& _rhs,
	  std::vector<value_t>& _flux,
	  index_t _n_matrix, index_t _n_bounds, index_t _n_fracs);

  /// @brief init mesh by reading MPSA connections
  int init_mpsa(std::vector<index_t>& block_m,
	  std::vector<index_t>& block_p,
	  std::vector<index_t>& _sstencil,
	  std::vector<index_t>& _sst_offset,
	  std::vector<value_t>& _stran,
	  uint8_t _n_dim, 
	  index_t _n_matrix, index_t _n_bounds, index_t _n_fracs);

  int init_mpsa(std::vector<index_t>& block_m,
	  std::vector<index_t>& block_p,
	  std::vector<index_t>& _sstencil,
	  std::vector<index_t>& _sst_offset,
	  std::vector<value_t>& _stran,
	  std::vector<value_t>& _flux,
	  uint8_t _n_dim,
	  index_t _n_matrix, index_t _n_bounds, index_t _n_fracs);

  /// @brief init mesh by reading both MPFA and MPSA connections
  int init_pm(std::vector<index_t>& block_m,
	  std::vector<index_t>& block_p,
	  std::vector<index_t>& _stencil,
	  std::vector<index_t>& _st_offset,
	  std::vector<value_t>& _tran,
	  std::vector<value_t>& _rhs,
	  index_t _n_matrix, index_t _n_bounds, index_t _n_fracs);
  int init_pm(std::vector<index_t>& block_m,
      std::vector<index_t>& block_p,
      std::vector<index_t>& _stencil,
      std::vector<index_t>& _st_offset,
      std::vector<value_t>& _tran,
	  std::vector<value_t>& _rhs,
	  std::vector<value_t>& _tran_biot,
	  std::vector<value_t>& _rhs_biot,
      index_t _n_matrix, index_t _n_bounds, index_t _n_fracs);
  int init_pm(std::vector<index_t>& block_m,
	  std::vector<index_t>& block_p,
	  std::vector<index_t>& _stencil,
	  std::vector<index_t>& _st_offset,
	  std::vector<value_t>& _tran,
	  std::vector<value_t>& _rhs,
	  std::vector<value_t>& _tran_biot,
	  std::vector<value_t>& _rhs_biot,
	  std::vector<value_t>& _tran_face,
	  std::vector<value_t>& _rhs_face,
	  index_t _n_matrix, index_t _n_bounds, index_t _n_fracs);
  int init_pme(std::vector<index_t>& block_m,
	  std::vector<index_t>& block_p,
	  std::vector<index_t>& _stencil,
	  std::vector<index_t>& _st_offset,
	  std::vector<value_t>& _tran,
	  std::vector<value_t>& _rhs,
	  std::vector<value_t>& _tran_biot,
	  std::vector<value_t>& _rhs_biot,
	  std::vector<value_t>& _tran_thermal,
	  std::vector<value_t>& _tran_thermal_expn,
	  index_t _n_matrix, index_t _n_bounds, index_t _n_fracs);

  /// @brief init mesh for 1D reservoir with 'nb' blocks
  int init_const_1d(double trans_const, index_t nb);     

  /// @brief add a new connection to connection list
  int add_conn(index_t block_m, index_t block_p,
    value_t trans, value_t transD);
  int add_conn_block(index_t block_m, index_t block_p,
    value_t trans, value_t transD, const uint8_t P_VAR);

  /// @brief reverse connections and sort them by both row and col
  int reverse_and_sort(); 
  /// @brief reverse connections and renumerate velocity mappers and sort them by both row and col
  int reverse_and_sort_dvel();
  /// @brief reverse mpsa connections and sort them by both row and col
  int reverse_and_sort_mpfa();
  int reverse_and_sort_mpsa();
  int reverse_and_sort_pm();
  int reverse_and_sort_pme();

  /// @brief discretize ms wells into reservoir
  int add_wells(std::vector<ms_well*> &wells);         
  int add_wells_mpfa(std::vector<ms_well*> &wells, const uint8_t P_VAR);
  int connect_segments(ms_well* well1, ms_well* well2, int iseg1, int iseg2, int verbose=0);

  void shift_boundary_ids_mpfa(const int n);

  // two-way, sorted connection list    
  /// [n_conns] array of indices of blocks on the minus side of a connection (smaller index)
  std::vector<index_t> block_m;         
  /// [n_conns] array of indices of blocks on the plus side of a connection (bigger index)                        
  std::vector<index_t> block_p;         
  /// [n_conns] array of transissibility values for given connection                        
  std::vector<value_t> tran;            
  /// [n_conns] array of diffusion transissibility values for given connection (transmis value)                        
  std::vector<value_t> tranD;          
  /// [n_conns] array of transmissibilities that describe the forces due to thermal dilation
  std::vector<value_t> tran_th_expn;
  /// [n_conns] array of gravity coefficient for every connection ( = (depth[block_m] - depth[block_p]) * g)                        
  std::vector<value_t> grav_coef;
  /// [n_conns] array of initial velocity values (Decouple - velocity engine)                        
  std::vector<value_t> velocity;
  /// [n_conns] array of temporary const transmissibilities                       
  std::vector<value_t> tran_const;
  /*
  * Multi-point stuff
  */
  /// Number of unknowns per block
  uint8_t n_vars;
  /// array of indices of blocks are neccessary for each connection
  std::vector<index_t> stencil;
  /// [n_conns + 1] array of offsets of the first block of connection in 'stencil'
  std::vector<index_t> offset;
  /// [n_conns] array of transissibility values for biot contribution to the given connection                        
  std::vector<value_t> tran_biot;
  // [n_blocks] stencil per block
  std::vector<std::vector<int>> cell_stencil;
  // [n_conns] the rest part of the flux
  std::vector<value_t> rhs;
  // [n_conns] the rest part of the biot contribution to flux
  std::vector<value_t> rhs_biot;
  // [n_blocks] vector of volumetric force
  std::vector<value_t> f;
  // [n_conns] array of the discretization of unknown pressure and displacements on faces
  std::vector<value_t> tran_face;
  // [n_conns] array of free-terms in the discretization of unknown pressure and displacements on faces
  std::vector<value_t> rhs_face;
  /// number of non-zero links 
  index_t n_links;

  /*
  * Previous time step
  */
  /// [n_conns] array of transissibility values for biot contribution to the given connection                        
  std::vector<value_t> tran_biot_n;
  // [n_conns] the rest part of the biot contribution to flux
  std::vector<value_t> rhs_biot_n;

  /*
  * Reference state
  */
  /// [n_conns] array of transissibility values for given connection                        
  std::vector<value_t> tran_ref;
  // [n_conns] the rest part of the flux
  std::vector<value_t> rhs_ref;
  /// [n_conns] array of transissibility values for biot contribution to the given connection                        
  std::vector<value_t> tran_biot_ref;
  // [n_conns] the rest part of the biot contribution to flux
  std::vector<value_t> rhs_biot_ref;


  /// [n_blocks] array of volumes of mesh blocks 
  std::vector<value_t> volume;          
  /// [n_blocks] array of porosities of mesh blocks                        
  std::vector<value_t> poro;            
  /// [n_blocks] array of depths                        
  std::vector<value_t> depth;           
  /// [n_blocks] array of heat capacity of rock                        
  std::vector<value_t> heat_capacity;
  /// [n_blocks] array of heat conduction of rock;                       
  std::vector<value_t> rock_cond;      
  /// [n_blocks] array of kinetic rate constants (dependent on the initial porosity and other factors!);                       
  std::vector<value_t> kin_factor;
  /// [np * n_blocks] array of phase mobility multiplier (dependent on phase index!);                       
  std::vector<value_t> mob_multiplier;
                                        
  /// [n_blocks] array of initial pressure values
  std::vector<value_t> pressure;     
  /// [n_blocks] array of reference pressure values
  std::vector<value_t> ref_pressure;
  /// [n_blocks] array of reference temperature values
  std::vector<value_t> ref_temperature;
  /// [n_blocks] array of reference volumetric strain
  std::vector<value_t> ref_eps_vol;
  /// [n_blocks] array of initial composition values                        
  std::vector<value_t> composition;     
  /// [n_blocks] array of initial temperature values                          
  std::vector<value_t> temperature;     
  /// [n_blocks] array of initial enthalpy values                        
  std::vector<value_t> enthalpy;      
  /// [n_dim * n_blocks] array of initial displacements values
  std::vector<value_t> displacement;
  /// [(1 + n_dim) * n_bounds] array of boundary conditions
  std::vector<value_t> bc;
  /// [(1 + n_dim) * n_bounds] array of boundary conditions
  std::vector<value_t> bc_n;
  /// [(1 + n_dim) * n_bounds] array of boundary conditions
  std::vector<value_t> bc_ref;
  /// [nc * n_bounds] array of pressures and (inflow) fractions at boundaries
  std::vector<value_t> pz_bounds;
  /// [9 * n_blocks] array of biot coefficients of mesh blocks                        
  std::vector<value_t> biot;
  /// [n_blocks] array of drained compressibility of mesh blocks                        
  std::vector<value_t> drained_compressibility;
  /// [n_blocks] array of calculated fluxes                        
  std::vector<value_t> flux;
  /// [n_blocks] array of calculated gravity contribution
  std::vector<value_t> grav_flux;
  /// [n_blocks] array of volumetric thermal dilation coefficient related to porosity
  std::vector<value_t> th_poro;

  /// [n_conns] array of connection ids in the order sorted for jacobian assembly
  std::vector<index_t> sorted_conn_ids;
  /// [n_conns] array of stencil ids in the order sorted for jacobian assembly
  std::vector<index_t> sorted_stencil_ids;
  /// [n_blocks] array of operator set index for every block
  std::vector<index_t> op_num;
  /// [n_fracs] array of pairs of connection ids at fault cells
  std::vector<std::vector<index_t>> fault_conn_id;
  /// [n_fracs] array of pairs of ids of contact cells
  std::vector<std::pair<index_t, index_t>> contact_cell_ids;
  std::vector<value_t> fault_normals;
  /// @} // end of Mesh

  std::vector<value_t> one_way_flux;
  std::vector<value_t> one_way_gravity_flux;
private:
  // one-way, unsorted conection list
  std::vector <index_t> one_way_block_m;
  std::vector <index_t> one_way_block_p;
  std::vector <value_t> one_way_tran;
  std::vector <value_t> one_way_tranD;
  std::vector <value_t> one_way_tran_th_expn;
  // arrays for multi-point approximation
  std::vector<index_t> one_way_stencil;
  std::vector<index_t> one_way_offset;                 
  std::vector<value_t> one_way_rhs;
  std::vector<value_t> one_way_tran_biot;
  std::vector<value_t> one_way_rhs_biot;
  std::vector<value_t> one_way_tran_face;
  std::vector<value_t> one_way_rhs_face;

  index_t n_one_way_conns;
  index_t n_one_way_conns_res;

  std::vector <index_t> tmp_index;
};

#endif
