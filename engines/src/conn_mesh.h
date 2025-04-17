#ifndef CONN_MESH_H
#define CONN_MESH_H

#include <vector>
#include <string>
#include "globals.h"
#include "ms_well.h"


/// This class defines mesh and corresponding arrays
class conn_mesh
{

public:
  conn_mesh () {};                                          // default constructor

  int init_grav_coef(value_t grav_const = 9.80665e-5);      // discretize ms wells into reservoir

  int get_res_tran(std::vector<value_t> &res_tran, 
                   std::vector<value_t> &res_tranD);        // get trans for reservoir part

  int set_res_tran(std::vector<value_t> &res_tran,
                   std::vector<value_t> &res_tranD);        // set trans for reservoir part

  int get_wells_tran(std::vector<value_t> &wells_tran);     // get trans for wells part (well indexes)

  int set_wells_tran(std::vector<value_t> &wells_tran);     // set trans for wells part (well indexes)

public:
  index_t n_res_blocks;                                     // number of reservoir blocks in the mesh            (R)
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

  /// @brief init mesh by reading array of left/right neighbours 
  int init(std::vector<index_t> &block_m,
    std::vector<index_t> &block_p,
    std::vector<value_t> &tran,
    std::vector<value_t> &tranD);                    

  /// @brief init mesh by reading MPFA connections
  /*int init_mpfa(std::vector<index_t>& block_m,
      std::vector<index_t>& block_p,
      std::vector<index_t>& _fstencil,
      std::vector<index_t>& _fst_offset,
      std::vector<value_t>& _ftran,
	  std::vector<value_t>& _rhs,
	  index_t _n_matrix, index_t _n_bounds);*/

  /// @brief init mesh by reading MPFA connections for both fluid mass and heat fluxes
  int init_mpfa(std::vector<index_t>& block_m,
	  std::vector<index_t>& block_p,
	  std::vector<index_t>& _fstencil,
	  std::vector<index_t>& _fst_offset,
	  std::vector<value_t>& _ftran,
	  std::vector<value_t>& _rhs,
	  std::vector<value_t>& _dtran,
	  std::vector<value_t>& _htran,
	  index_t _n_matrix, index_t _n_bounds, index_t _n_fracs, index_t _n_vars);

  /// @brief init mesh by reading MPFA connections with provided flux
  /*int init_mpfa(std::vector<index_t>& block_m,
	  std::vector<index_t>& block_p,
	  std::vector<index_t>& _fstencil,
	  std::vector<index_t>& _fst_offset,
	  std::vector<value_t>& _ftran,
	  std::vector<value_t>& _rhs,
	  std::vector<value_t>& _flux,
	  index_t _n_matrix, index_t _n_bounds, index_t _n_fracs);*/

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
  int init_pm_mech_discretizer(
	  std::vector<index_t>& block_m,
	  std::vector<index_t>& block_p,
	  std::vector<index_t>& _stencil,
	  std::vector<index_t>& _st_offset,
	  std::vector<value_t>& _hooke, std::vector<value_t>& _hooke_rhs,
	  std::vector<value_t>& _biot, std::vector<value_t>& _biot_rhs,
	  std::vector<value_t>& _darcy, std::vector<value_t>& _darcy_rhs,
	  std::vector<value_t>& _vol_strain, std::vector<value_t>& _vol_strain_rhs,
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
  int init_pme_mech_discretizer(
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
	index_t _n_matrix, index_t _n_bounds, index_t _n_fracs);

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
  int reverse_and_sort_pm_mech_discretizer();
  int reverse_and_sort_pme_mech_discretizer();

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
  /// [n_conns] array of heat conduction transissibility values for given connection (transmis value)                        
  std::vector<value_t> tran_heat_cond;
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
  /// Number of spatial dimensions
  uint8_t n_dim;
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
                                        
  /// [n_blocks * n_vars] array of initial state for solution
  std::vector<value_t> initial_state;
  /// [n_blocks] array of reference pressure values
  std::vector<value_t> ref_pressure;
  /// [n_blocks] array of reference temperature values
  std::vector<value_t> ref_temperature;
  /// [n_blocks] array of reference volumetric strain
  std::vector<value_t> ref_eps_vol;
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
  /// [n_blocks] array of rock compressibility of mesh blocks for mechanical models                        
  std::vector<value_t> rock_compressibility;
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

  /* New mechanical discretizer */
  std::vector<value_t> hooke_tran, hooke_rhs;
  std::vector<value_t> biot_tran, biot_rhs;
  std::vector<value_t> darcy_tran, darcy_rhs;
  std::vector<value_t> vol_strain_tran, vol_strain_rhs;
  std::vector<value_t> thermal_traction_tran;
  std::vector<value_t> fourier_tran;

  // adjoint method
  std::vector <index_t> cell_m_one_way;
  std::vector <index_t> cell_p_one_way;
  std::vector <index_t> conn_idx_to_one_way;

  std::vector <index_t> test_debug;

  /* to reconstruct velocities */
  std::vector<index_t> velocity_offset;
  std::vector<value_t> velocity_appr;

private:
  // one-way, unsorted conection list
  std::vector <index_t> one_way_block_m;
  std::vector <index_t> one_way_block_p;
  std::vector <value_t> one_way_tran;
  std::vector <value_t> one_way_tran_heat_cond;
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
  // arrays for the new mechanical discretizer
  std::vector<value_t> one_way_hooke, one_way_hooke_rhs;
  std::vector<value_t> one_way_biot, one_way_biot_rhs;
  std::vector<value_t> one_way_darcy, one_way_darcy_rhs;
  std::vector<value_t> one_way_vol_strain, one_way_vol_strain_rhs;
  std::vector<value_t> one_way_thermal_traction;
  std::vector<value_t> one_way_fourier;

  index_t n_one_way_conns;
  index_t n_one_way_conns_res;

  std::vector <index_t> tmp_index;
};

#endif
