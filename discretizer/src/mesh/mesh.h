#ifndef MESH_H_
#define MESH_H_

#include <map>
#include <set>
#include <string>
#include <array>

#include "elem.h"

namespace mesh
{
	typedef std::unordered_map<ElemLoc, std::set<index_t>> PhysicalTags;

	enum MESH_TYPE {GMSH, CPG};

	class Mesh
	{
	public:
		// mesh name
		std::string name;
		// plain array of node indices where every index stores the elements that index belongs to and array with offsets
		std::vector<index_t> elems_of_node, elems_of_node_offset;
		// array of grid nodes
		std::vector<Vector3> nodes;

		// number of nodes in the mesh
		index_t num_of_nodes;
		// number of elements
		index_t num_of_elements;
		
		// array of elements
		std::vector<Elem> elems;
		// array of volumes for 3D, areas for 2D, lengths for 1D
		std::vector<value_t> volumes;
		// array of centroids of elements
		std::vector<Vector3> centroids;
		// array of depths of elements
		std::vector<value_t> depths;
		// array of active cells
		std::vector<index_t> actnum;
		// Array of porosities in the case of CPG
		std::vector<value_t> poro;
		// array of node ids belonging to elements (to preserve Gmsh order of element points)
		std::vector<index_t> elem_nodes;
		// array of sorted node ids belonging to elements (for faster intersection)
		std::vector<index_t> elem_nodes_sorted;
		// (start, end) id for every region
		std::map<ElemLoc, std::pair<index_t, index_t>> region_ranges;
		// number of element in every region
		std::map<ElemLoc, index_t> region_elems_num;
		// vector of initial (reference) fault apertures
		std::vector<value_t> init_apertures;
		// vector of elements' tags
		std::vector<index_t> element_tags;

		int mesh_type;
		// CPG ==============================================
		// length of vector of unknowns (only active cells)
		index_t n_cells; // number of active cells
		index_t nx;
		index_t ny;
		index_t nz;
		std::vector<value_t> coord;
		std::vector<value_t> zcorn; //TODO: we can change type to float but need to change it in opm code too
		std::vector<index_t> local_to_global; // convert indices from active elements to full grid
		std::vector<index_t> global_to_local; // convert indices from full grid to active elements

		// ==============================================

		/*  Connections   */
		// array of connections
		std::vector<Connection> conns;
		// array of node ids belonging to connection
		std::vector<index_t> conn_nodes;
		// vector of connection ids for every element == adjacency matrix
		std::vector<index_t> adj_matrix;
		// vector of connection ids for every element == adjacency matrix
		std::vector<index_t> adj_matrix_cols;
		// vector of offsets for the adjacency matrix
		std::vector<index_t> adj_matrix_offset;
		
		// elem ids for each elem type
		std::unordered_map<ElemType, std::vector<index_t>> elem_type_map;
		
		// conn ids for each conn type
		std::unordered_map<ConnType, std::vector<index_t>> conn_type_map;

		/* Fluxes */
		// array of fluxes through elements
		std::vector<value_t> flux_elems;
		// array of offsets of fluxes
		std::vector<index_t> flux_offset;

		Mesh();
		~Mesh();

		void gmsh_mesh_processing(std::string filename, const PhysicalTags& tags);

		void print_elems_nodes();

		// get cell IJK indices (i1, j1, k1) in 0-based from 1D global index idx (0 <= idx < n_cells)
		// assume loops order is K-J-I, i.e. I is fastest index
		void inline get_ijk_from_global_idx (index_t idx, int &i1, int &j1, int &k1) const {
			k1 = idx / (nx * ny);
			j1 = (idx - k1 * (nx * ny)) / nx;
			i1 = idx % nx;
		}

		// get cell IJK indices (i1, j1, k1) in 0-based from 1D local index idx (0 <= idx < n_active_cells)
		// assume loops order is K-J-I, i.e. I is fastest index
		void inline get_ijk_from_local_idx(index_t idx, int &i1, int &j1, int &k1) const {
			index_t global_idx = local_to_global[idx];
			get_ijk_from_global_idx(global_idx, i1, j1, k1);
		}

		// get cell IJK indices (i1, j1, k1) in 0-based from 1D local index idx (0 <= idx < n_active_cells)
		// assume loops order is K-J-I, i.e. I is fastest index
		// is_global should be true  if idx in range of all elements, including inactive (input_data)
		// is_global should be false if idx in range of active elements (internal arrays)
		void inline get_ijk(index_t idx, int &i1, int &j1, int &k1, const bool is_global) const {
			if (is_global)
			  get_ijk_from_global_idx(idx, i1, j1, k1);
			else
				get_ijk_from_local_idx(idx, i1, j1, k1);
		}

		void construct_local_global(std::vector<index_t>& global_cell);

		// assume loops order is K-J-I, i.e. I is fastest index. I,J,K are 0-based indices
		index_t inline get_global_index(const int i1, const int j1, const int k1) const {
			return k1 * nx * ny + j1 * nx + i1;
		}

		// get cell IJK indices (i1, j1, k1)  from 1D-index idx (0 <= idx < n_cells) and return str (i,j,k) in 1-based
		std::string inline get_ijk_as_str(int idx, bool idx_is_global) const {
			std::string s;
			int i1, j1, k1;
			get_ijk(idx, i1, j1, k1, idx_is_global);
			s = "(" + std::to_string(i1 + 1) + "," + std::to_string(j1 + 1) + "," + std::to_string(k1 + 1) + ")";
			return s;
		}

		// returns total number of cells
		int inline get_n_cells_total() const {
			return nx * ny * nz;
		}

		// fills X,Y,Z vectors with nodes of cell (i,j,k). ijk is 0-based
		void calc_cell_nodes(const int i, const int j, const int k,
												std::array<double, 8>& X, std::array<double, 8>& Y, std::array<double, 8>& Z) const;

		// return array with sizes (dx, dy, dz) of cell (i,j,k). ijk is 0-based
		std::array<value_t, 3> calc_cell_sizes(const int i, const int j, const int k) const;

		// return array with center (x, y, z) of cell (i,j,k). ijk is 0-based
		std::array<value_t, 3> calc_cell_center(const int i, const int j, const int k) const;

		// arr - array with length number of active cells
		// num_of_cells - number of all cells
		template <typename T>
		void write_array_to_file(const std::string filename,
		        const std::string keyword,
		        const std::vector<T> &arr, const std::vector<int> &actnum,
		        const int num_of_cells,
		        const double multiplier,
		        const bool append) const
		{
		        std::cout << "Writing array " << keyword << " to file " << filename << "\n";
		
		        std::ofstream f;
		        if (append)
		          f.open(filename, std::ios_base::app);
		        else
		                f.open(filename);
		        f << keyword << "\n";

		        int numbers_in_line = 6; // limit to shorten the lines in a file
						int inactive_value = 0;  // value for inactive cells

		        for (int i = 0, j = 0; i < num_of_cells; i++) {
		                if (actnum.empty() || actnum[i])
		                f << arr[j++] * multiplier << " ";
		                else
		                        f << inactive_value << " ";
		                if ((i+1) % numbers_in_line == 0)
		                        f << "\n";
		        }

		        f << "\n / \n";
		        f.close();
		}

		void write_cell_sizes(const std::string fname) const;
		std::vector<value_t> get_prisms() const;
		std::vector<value_t> get_centers() const;
		std::vector<value_t> get_nodes_array() const;
		void print_arrays() const;

		std::vector<int> cpg_elems_nodes(
			const int _number_of_nodes,
			const int number_of_cells,// number of active cells 
			const int number_of_faces,
			const std::vector<double>& node_coords,
			const std::vector<int>& face_nodes,
			const std::vector<int>& face_nodepos,
			const std::vector<int>& face_cells,
			const std::vector<int>& cell_faces,
			const std::vector<int>& cell_facepos,
			const std::vector<double>& cell_volumes,
			std::vector<int>& face_order);

		void cpg_cell_props(
			const int _number_of_nodes,
			const int num_of_cells,// number of active cells 
			const int number_of_faces,
			const std::vector<double>& cell_volumes,
			const std::vector<double>& cell_centroids,
			const std::vector<int>& global_cell,
			const std::vector<double>& face_areas,
			const std::vector<double>& face_centroids,
			//output
			const int bnd_faces_num,
			const std::vector<int>& face_order);

		void cpg_connections(
			const int num_of_cells,// number of active cells 
			const int number_of_faces,
			const std::vector<double>& node_coords,
			const std::vector<int>& face_nodes,
			const std::vector<int>& face_nodepos,
			const std::vector<int>& face_cells,
			const std::vector<int>& cell_faces,
			const std::vector<int>& cell_facepos,
			const std::vector<double>& face_centroids,
			const std::vector<double>& face_areas,
			const std::vector<double>& face_normals,
			const std::vector<int>& cell_facetag,
			const PhysicalTags& tags);

		void gmsh_mesh_reading(std::string filename, const PhysicalTags& tags);
		void gmsh_mesh_construct_connections(const PhysicalTags& tags);
		void generate_adjacency_matrix();

	}; //class Mesh

}//namespace mesh

#endif /* MESH_H_ */