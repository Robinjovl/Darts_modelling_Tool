#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "mesh.h"
#include "mshio/mshio.h"
#include "linalg/matrix.h"
#include "linalg/vector3.h"
#include "utils.h"

using namespace mesh;
using linalg::Vector3;
using std::unordered_map;
using std::unordered_set;
using std::string;
using std::vector;
using std::set;
using std::pair;
using std::find;
using std::distance;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::cout;
using std::endl;

namespace
{
	std::vector<std::vector<index_t>> topology_faces(const ElemType type)
	{
		switch (type)
		{
		case TETRA:
			return {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
		case HEX:
			return {{0, 1, 2, 3}, {4, 7, 6, 5}, {0, 4, 5, 1},
					{1, 5, 6, 2}, {2, 6, 7, 3}, {3, 7, 4, 0}};
		case PRISM:
			return {{0, 2, 1}, {3, 4, 5}, {0, 1, 4, 3},
					{1, 2, 5, 4}, {2, 0, 3, 5}};
		case PYRAMID:
			return {{0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
		default:
			return {};
		}
	}

	Vector3 face_centroid(const std::vector<Vector3>& nodes, const std::vector<index_t>& face)
	{
		Vector3 centroid;
		for (const index_t node : face)
			centroid += nodes[node];
		if (!face.empty())
			centroid /= static_cast<value_t>(face.size());
		return centroid;
	}

	value_t face_area(const std::vector<Vector3>& nodes, const std::vector<index_t>& face)
	{
		value_t area = 0.0;
		for (size_t i = 1; i + 1 < face.size(); ++i)
			area += 0.5 * cross(nodes[face[i]] - nodes[face[0]],
							 nodes[face[i + 1]] - nodes[face[0]]).norm();
		return area;
	}
}

Mesh::Mesh()
{
	elem_type_map[LINE] = vector<index_t>();
	elem_type_map[TRI] = vector<index_t>();
	elem_type_map[QUAD] = vector<index_t>();
	elem_type_map[TETRA] = vector<index_t>();
	elem_type_map[HEX] = vector<index_t>();
	elem_type_map[PRISM] = vector<index_t>();
	elem_type_map[PYRAMID] = vector<index_t>();
}

Mesh::~Mesh()
{
}

void Mesh::gmsh_mesh_processing(string filename, const PhysicalTags& tags)
{
	gmsh_mesh_reading(filename, tags);
	gmsh_mesh_construct_connections(tags);
	generate_adjacency_matrix();
	build_gmsh_weno_face_sides();
}

void Mesh::build_gmsh_weno_face_sides()
{
	weno_face_sides.clear();
	weno_face_nodes.clear();
	weno_cell_face_offset.clear();
	weno_cell_face_offset.reserve(static_cast<size_t>(n_cells) + 1);
	weno_cell_face_offset.push_back(0);

	const auto matrix_range_it = region_ranges.find(MATRIX);
	const index_t matrix_begin = matrix_range_it == region_ranges.end() ? 0 : matrix_range_it->second.first;
	const index_t matrix_end = matrix_range_it == region_ranges.end() ? 0 : matrix_range_it->second.second;

	for (index_t cell = 0; cell < n_cells; ++cell)
	{
		if (cell < matrix_begin || cell >= matrix_end)
		{
			weno_cell_face_offset.push_back(static_cast<index_t>(weno_face_sides.size()));
			continue;
		}

		const Elem& elem = elems[cell];
		const auto local_faces = topology_faces(elem.type);
		std::map<std::vector<index_t>, index_t> connection_by_nodes;
		for (index_t adj = adj_matrix_offset[cell]; adj < adj_matrix_offset[cell + 1]; ++adj)
		{
			const index_t connection_id = adj_matrix[adj];
			const Connection& connection = conns[connection_id];
			std::vector<index_t> key(conn_nodes.begin() + connection.pts_offset,
									 conn_nodes.begin() + connection.pts_offset + connection.n_pts);
			std::sort(key.begin(), key.end());
			connection_by_nodes.emplace(std::move(key), connection_id);
		}

		for (index_t local_face = 0; local_face < static_cast<index_t>(local_faces.size()); ++local_face)
		{
			std::vector<index_t> face;
			face.reserve(local_faces[local_face].size());
			for (const index_t local_node : local_faces[local_face])
				face.push_back(elem_nodes[elem.pts_offset + local_node]);

			std::vector<index_t> key = face;
			std::sort(key.begin(), key.end());
			const auto connection_it = connection_by_nodes.find(key);

			WenoFaceSide side;
			side.cell = cell;
			side.logical_face = local_face;
			side.node_offset = static_cast<index_t>(weno_face_nodes.size());
			side.node_count = static_cast<index_t>(face.size());
			weno_face_nodes.insert(weno_face_nodes.end(), face.begin(), face.end());

			if (connection_it == connection_by_nodes.end())
			{
				side.neighbour = -1;
				side.connection_id = -1;
				side.centroid = face_centroid(nodes, face);
				side.area = face_area(nodes, face);
			}
			else
			{
				const Connection& connection = conns[connection_it->second];
				side.connection_id = connection.conn_id;
				side.centroid = connection.elem_id1 == cell ? connection.c : connection.c_2;
				side.area = connection.area;

				const index_t other = connection.elem_id1 == cell ? connection.elem_id2 : connection.elem_id1;
				if (connection.type == MAT_MAT && other >= matrix_begin && other < matrix_end)
					side.neighbour = other;
				else if (connection.type == MAT_BOUND)
					side.neighbour = -1;
				else
					side.neighbour = -2;
			}
			weno_face_sides.push_back(side);
		}
		weno_cell_face_offset.push_back(static_cast<index_t>(weno_face_sides.size()));
	}
}

void Mesh::build_structured_weno_geometry(
	index_t nx_, index_t ny_, index_t nz_,
	const std::vector<index_t>& global_to_local_,
	const std::vector<value_t>& cell_centroids,
	const std::vector<value_t>& cell_sizes)
{
	const index_t n_global = nx_ * ny_ * nz_;
	if (static_cast<index_t>(global_to_local_.size()) != n_global ||
		static_cast<index_t>(cell_centroids.size()) != ND * n_global ||
		static_cast<index_t>(cell_sizes.size()) != ND * n_global)
		throw std::invalid_argument("Invalid structured WENO geometry array size");

	mesh_type = STRUCTURED;
	nx = nx_;
	ny = ny_;
	nz = nz_;
	global_to_local = global_to_local_;
	n_cells = 0;
	for (const index_t local : global_to_local)
		if (local >= 0)
			n_cells = std::max(n_cells, local + 1);
	num_of_elements = n_cells;
	local_to_global.assign(n_cells, -1);
	for (index_t global = 0; global < n_global; ++global)
		if (global_to_local[global] >= 0)
			local_to_global[global_to_local[global]] = global;

	nodes.clear();
	elems.clear();
	elem_nodes.clear();
	centroids.assign(n_cells, Vector3());
	volumes.assign(n_cells, 0.0);
	element_tags.assign(n_cells, 0);
	weno_face_sides.clear();
	weno_face_nodes.clear();
	weno_cell_face_offset.clear();
	elem_type_map.clear();
	elem_type_map[HEX].reserve(n_cells);
	region_ranges.clear();
	region_elems_num.clear();
	region_ranges[MATRIX] = {0, n_cells};
	region_elems_num[MATRIX] = n_cells;

	nodes.reserve(static_cast<size_t>(n_cells) * 8);
	elems.reserve(n_cells);
	elem_nodes.reserve(static_cast<size_t>(n_cells) * 8);
	weno_face_sides.reserve(static_cast<size_t>(n_cells) * 6);
	weno_face_nodes.reserve(static_cast<size_t>(n_cells) * 24);
	weno_cell_face_offset.reserve(static_cast<size_t>(n_cells) + 1);
	weno_cell_face_offset.push_back(0);

	const auto hex_faces = topology_faces(HEX);
	const std::array<std::array<index_t, 3>, 6> directions = {{{0, 0, -1}, {0, 0, 1},
		{0, -1, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}}};
	std::map<std::pair<index_t, index_t>, index_t> connection_ids;
	index_t next_connection = 0;

	for (index_t cell = 0; cell < n_cells; ++cell)
	{
		const index_t global = local_to_global[cell];
		if (global < 0)
			throw std::invalid_argument("Structured WENO local numbering contains a gap");
		const index_t i = global % nx;
		const index_t j = (global / nx) % ny;
		const index_t k = global / (nx * ny);
		const Vector3 centre(cell_centroids[ND * global], cell_centroids[ND * global + 1],
						 cell_centroids[ND * global + 2]);
		const value_t hx = 0.5 * cell_sizes[ND * global];
		const value_t hy = 0.5 * cell_sizes[ND * global + 1];
		const value_t hz = 0.5 * cell_sizes[ND * global + 2];
		if (!(hx > 0.0 && hy > 0.0 && hz > 0.0))
			throw std::invalid_argument("Structured WENO cell sizes must be positive");

		centroids[cell] = centre;
		volumes[cell] = 8.0 * hx * hy * hz;
		const index_t node_offset = static_cast<index_t>(nodes.size());
		const std::array<Vector3, 8> cell_nodes = {{
			{centre.x - hx, centre.y - hy, centre.z - hz},
			{centre.x + hx, centre.y - hy, centre.z - hz},
			{centre.x + hx, centre.y + hy, centre.z - hz},
			{centre.x - hx, centre.y + hy, centre.z - hz},
			{centre.x - hx, centre.y - hy, centre.z + hz},
			{centre.x + hx, centre.y - hy, centre.z + hz},
			{centre.x + hx, centre.y + hy, centre.z + hz},
			{centre.x - hx, centre.y + hy, centre.z + hz}}};
		nodes.insert(nodes.end(), cell_nodes.begin(), cell_nodes.end());
		for (index_t local_node = 0; local_node < 8; ++local_node)
			elem_nodes.push_back(node_offset + local_node);

		Elem elem(HEX, cell, static_cast<index_t>(elem_nodes.size()) - 8);
		elem.loc = MATRIX;
		elems.push_back(elem);
		elem_type_map[HEX].push_back(cell);

		for (index_t face_id = 0; face_id < 6; ++face_id)
		{
			std::vector<index_t> face;
			face.reserve(4);
			for (const index_t local_node : hex_faces[face_id])
				face.push_back(node_offset + local_node);

			const index_t ni = i + directions[face_id][0];
			const index_t nj = j + directions[face_id][1];
			const index_t nk = k + directions[face_id][2];
			index_t neighbour = -1;
			if (ni >= 0 && ni < nx && nj >= 0 && nj < ny && nk >= 0 && nk < nz)
				neighbour = global_to_local[nk * nx * ny + nj * nx + ni];

			WenoFaceSide side;
			side.cell = cell;
			side.neighbour = neighbour;
			side.logical_face = face_id;
			side.node_offset = static_cast<index_t>(weno_face_nodes.size());
			side.node_count = static_cast<index_t>(face.size());
			side.centroid = face_centroid(nodes, face);
			side.area = face_area(nodes, face);
			weno_face_nodes.insert(weno_face_nodes.end(), face.begin(), face.end());

			if (neighbour >= 0)
			{
				const auto key = std::minmax(cell, neighbour);
				const std::pair<index_t, index_t> pair_key(key.first, key.second);
				auto [it, inserted] = connection_ids.emplace(pair_key, next_connection);
				if (inserted)
					++next_connection;
				side.connection_id = it->second;
			}
			weno_face_sides.push_back(side);
		}
		weno_cell_face_offset.push_back(static_cast<index_t>(weno_face_sides.size()));
	}
	num_of_nodes = static_cast<index_t>(nodes.size());
}

// fills:
// nodes, elems, elems_of_node, elem_nodes, elem_nodes_sorted,
// elem_type_map, region_ranges, region_elems_num,
// volumes, centroids
void Mesh::gmsh_mesh_reading(string filename, const PhysicalTags& tags)
{
	steady_clock::time_point t1, t2;
	mesh_type = GMSH;
	// mesh loading
	t1 = steady_clock::now();
	// load mesh file, which is a list of points in 3D
	mshio::MshSpec spec = mshio::load_msh(filename);
	t2 = steady_clock::now();
	cout << "Reading of " + filename + ":\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;

	t1 = steady_clock::now();

	// count all the points
	num_of_nodes = 0;
	for (const auto& block : spec.nodes.entity_blocks)
		num_of_nodes += static_cast<index_t>(block.num_nodes_in_block);

	// reserve memory for points
	nodes.reserve(num_of_nodes);

	// insert all the points
	for (const auto& block : spec.nodes.entity_blocks)
	{
		for (index_t i = 0; i < block.num_nodes_in_block; i++)
			nodes.emplace_back(block.data[ND * i], block.data[ND * i + 1], block.data[ND * i + 2]);
	}

	// count all the elements
	num_of_elements = 0;
	for (const auto& block : spec.elements.entity_blocks)
	{
		num_of_elements += static_cast<index_t>(block.num_elements_in_block);
	}

	// reserve memory for arrays
	elems.reserve(num_of_elements);
	volumes.resize(num_of_elements);
	centroids.resize(num_of_elements);
	element_tags.resize(num_of_elements);
	elem_nodes.reserve(MAX_PTS_PER_3D_ELEM * num_of_elements);
	elem_nodes_sorted.reserve(MAX_PTS_PER_3D_ELEM * num_of_elements);
	std::vector<index_t> num_elems_of_node(num_of_nodes, 0);

	// lambda function to find specific entity tag
	auto findInVector = [](auto& vec, index_t entity_tag) -> index_t
	{
		auto it = std::find_if(vec.begin(), vec.end(), [entity_tag](const auto& ent)
		  { return ent.tag == entity_tag; });
		if (it == vec.end())
		{
		  printf("Entity tag %d not found\n", entity_tag);
		  return -1;
		}
		else
		  return it->physical_group_tags[0];
	};

	// create all the elements
	uint8_t stride;
	index_t counter = 0, offset = 0;
	for (const auto& region : elem_order)
	{
		region_ranges[region].first = counter;
		for (const auto& block : spec.elements.entity_blocks)
		{
			const auto& loc_tags = tags.at(region);

			int block_tag;
			if (block.entity_dim == 0)
				block_tag = findInVector(spec.entities.points, block.entity_tag);
			else if (block.entity_dim == 1)
				block_tag = findInVector(spec.entities.curves, block.entity_tag);
			else if (block.entity_dim == 2)
				block_tag = findInVector(spec.entities.surfaces, block.entity_tag);
			else if (block.entity_dim == 3)
				block_tag = findInVector(spec.entities.volumes, block.entity_tag);

			if (loc_tags.find(block_tag) == loc_tags.end()) continue;

			stride = Etype_PTS.at(static_cast<ElemType>(block.element_type)) + 1;

			for (index_t i = 0; i < block.num_elements_in_block; i++)
			{
				// element
				Elem el;
				el.loc = region;
				el.type = static_cast<ElemType>(block.element_type);
				el.elem_id = counter++;
				el.n_pts = Etype_PTS.at(el.type);
				element_tags[el.elem_id] = block_tag;
				el.pts_offset = offset;

				offset += el.n_pts;

				// TODO: evaluate centroid and volume (surface for 2D, length for 1D) of the element
				// nodes, call Elem methods for el
				for (index_t j = stride * i + 1; j < stride * (i + 1); j++)
				{
					elem_nodes.push_back(static_cast<index_t>(block.data[j] - 1));
					elem_nodes_sorted.push_back(static_cast<index_t>(block.data[j] - 1));
					//if (j > stride * i + 1)
					//	assert(elem_nodes.back() > elem_nodes[elem_nodes.size() - 2]);
				}

				std::sort(elem_nodes_sorted.end() - (stride - 1), elem_nodes_sorted.end());

				// count for every node number of elements it belongs to
				for (index_t j = el.pts_offset; j < el.pts_offset + el.n_pts; j++)
					num_elems_of_node[elem_nodes[j]]++;
				// calculate the volume and centroid for the element

				el.calculate_volume_and_centroid(nodes, elem_nodes, volumes[el.elem_id], centroids[el.elem_id]);

#ifdef DEBUG_TRANS
				cout << "CPP GMSH ID=" << el.elem_id << ", L= " << el.loc << ", T= " << el.type << ", V= " << volumes[el.elem_id] << ", C= " << centroids[el.elem_id] << "\n";
#endif // DEBUG_TRANS
				//std::cout << "volume: " << el.volume << std::endl;
				//std::cout << "centroid: " << el.c << "\n" << std::endl;


				elems.push_back(el);
			}
		}
		region_ranges[region].second = counter;
		region_elems_num[region] = region_ranges[region].second - region_ranges[region].first;
	}

	n_cells = region_ranges[FRACTURE].second;
	assert(*std::max_element(elem_nodes.begin(), elem_nodes.end()) < num_of_nodes);

	// reserve elem types map
	for (auto& type : elem_type_map)
	{
		type.second.reserve(elems.size());
	}

	// creating array of elem ids per node
	elems_of_node_offset.resize(num_of_nodes + 8);
	elems_of_node_offset[0] = 0;
	std::partial_sum(num_elems_of_node.begin(), num_elems_of_node.end(), elems_of_node_offset.begin() + 1);
	std::fill(num_elems_of_node.begin(), num_elems_of_node.end(), 0);

	elems_of_node.resize(offset);
	index_t node_id, pos;
	if (elems.size() != num_of_elements)
	  {
		std::cout << "elems.size " << elems.size() << " != num_of_elements " << num_of_elements << "\n";
		exit(1);
	  }
	for (index_t i = 0; i < num_of_elements; i++)
	{
		const auto& el = elems[i];
		if (i < n_cells) elem_type_map[el.type].push_back(el.elem_id);
		for (index_t j = el.pts_offset; j < el.pts_offset + el.n_pts; j++)
		{
			node_id = elem_nodes[j];
			pos = elems_of_node_offset[node_id] + num_elems_of_node[node_id];
			elems_of_node[pos] = i;
			num_elems_of_node[node_id]++;
		}
	}

	// erase zeros in elem types map
	for (auto it = std::begin(elem_type_map); it != std::end(elem_type_map);)
	{
		if (it->second.size() == 0) it = elem_type_map.erase(it);
		else ++it;
	}

	t2 = steady_clock::now();
	cout << "Processing " + std::to_string(num_of_nodes) + " nodes, " +
		std::to_string(num_of_elements) + " elements:\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}

// uses: elems, elems_of_node_offset, elem_nodes, elems_of_node
// fills: conns
void Mesh::gmsh_mesh_construct_connections(const PhysicalTags& tags)
{
	steady_clock::time_point t1, t2;
	t1 = steady_clock::now();

	index_t nebr_id, counter = 0, offset = 0, node_id;
	size_t len;

	// vector of cell's indices pairs to check if particular one was already created
	vector<vector<index_t>> conn_set;
	conn_set.resize(num_of_elements);
	for (index_t i = 0; i < num_of_elements; i++)
		conn_set[i].reserve(MAX_CONNS_PER_ELEM_GMSH);

	std::vector<size_t> conn_set_size(num_of_elements, 0);
	vector<index_t>::const_iterator it1, it2;
	vector<index_t>::iterator it1_end, it2_end;

	conns.reserve(MAX_CONNS_PER_ELEM_GMSH * num_of_elements);
	// vector to store the result of intersections
	vector<index_t> intersect;
	intersect.reserve(4 * ND * num_of_elements * 2 * MAX_PTS_PER_3D_ELEM_GMSH);
	conn_nodes.reserve(4 * MAX_CONNS_PER_ELEM_GMSH * num_of_elements);

	ElemConnectionTable::const_iterator conn_type_it;

	// fault nodes
	unordered_set<set<index_t>, integer_set_hash> fault_nodes;
	fault_nodes.reserve(region_ranges[FRACTURE].second - region_ranges[FRACTURE].first);

	// loop through all elements
	for (index_t i = 0; i < num_of_elements; i++)
	{
		//cout << "\n==============\n i " << i << "\n";
		const auto& el1 = elems[i];
		// loop through all nodes belonging to particular element
		for (index_t l = el1.pts_offset; l < el1.pts_offset + el1.n_pts; l++)
		{
			node_id = elem_nodes[l];
			// loop through all elements that share this node
			for (index_t k = elems_of_node_offset[node_id]; k < elems_of_node_offset[node_id + 1]; k++)
			{
				nebr_id = elems_of_node[k];

				// skip the same element
				if (nebr_id == i) continue;
				//cout << "node_id=" << node_id << ", nebr_id=" << nebr_id << "\n";
				const auto& el2 = elems[nebr_id];
				auto& id1 = conn_set[el1.elem_id];
				size_t& id_size1 = conn_set_size[el1.elem_id];
				auto& id2 = conn_set[el2.elem_id];
				size_t& id_size2 = conn_set_size[el2.elem_id];
				it1_end = id1.begin() + id_size1;
				it2_end = id2.begin() + id_size2;
				it1 = find(id1.begin(), it1_end, el2.elem_id);
				it2 = find(id2.begin(), it2_end, el1.elem_id);

				if (it1 == it1_end && it2 == it2_end)
				{
#if 0 // debug
						int *e1 = elem_nodes_sorted.data();
						cout << "E1: " << el1.pts_offset << " , " << int(el1.n_pts) << " : ";
						for (auto i1 = el1.pts_offset; i1 < el1.pts_offset + el1.n_pts; i1++)
						{
							cout << e1[i1] << " ";
						}
						cout << "\n";

						int *e2 = elem_nodes_sorted.data();
						cout << "E2: "<< el2.pts_offset << " , " << int(el2.n_pts) << " : ";
						for (auto i1 = el2.pts_offset; i1 < el2.pts_offset + el2.n_pts; i1++)
						{
							cout << e2[i1] << " ";
						}
						cout << "\n\n";
#endif //debug

					len = intersect.size();
					std::set_intersection(elem_nodes_sorted.data() + el1.pts_offset, elem_nodes_sorted.data() + el1.pts_offset + el1.n_pts,
						elem_nodes_sorted.data() + el2.pts_offset, elem_nodes_sorted.data() + el2.pts_offset + el2.n_pts,
						std::back_inserter(intersect));
					len = intersect.size() - len;
					assert(len > 0); // intersection of nodes sets for two elements which have common node, should be nonzero
					if (len > 2)
					{
						conn_type_it = CONN_TYPE_TABLE.find({ el1.loc, el2.loc });
						if (conn_type_it == CONN_TYPE_TABLE.end())
							conn_type_it = CONN_TYPE_TABLE.find({ el2.loc, el1.loc });
						if (conn_type_it != CONN_TYPE_TABLE.end())
						{
							Connection conn;
							conn.conn_id = counter;
							conn.pts_offset = offset;
							conn.n_pts = static_cast<uint8_t>(len);
							conn_nodes.insert(conn_nodes.end(), intersect.end() - len, intersect.end());
							conn.elem_id1 = el1.elem_id;
							conn.elem_id2 = el2.elem_id;
							conn.calculate_centroid(nodes, conn_nodes);
							conn.calculate_area(nodes, conn_nodes);
							if (conn.area == 0.0)
							{
								conn.n.values[0] = conn.n.values[1] = conn.n.values[2] = 0.0;
							}
							else
							{
								conn.calculate_normal(nodes, conn_nodes, elems, elem_nodes);
							}
							if (conn_type_it->second == MAT_FRAC)
							{
								conn.c -= init_apertures[el2.elem_id - region_ranges[FRACTURE].first] * conn.n / 2.0;
								fault_nodes.insert(set<index_t>(intersect.end() - len, intersect.end()));
							}
							else if (conn_type_it->second == FRAC_MAT)
							{
								conn.c += init_apertures[el1.elem_id - region_ranges[FRACTURE].first] * conn.n / 2.0;
								fault_nodes.insert(set<index_t>(intersect.end() - len, intersect.end()));
							}

							id1.push_back(el2.elem_id);
							id_size1++;
							offset += conn.n_pts;

							conn.type = conn_type_it->second;
							conns.push_back(conn);
							counter++;
						}
					}
					else if (len == 2)
					{
						if (el1.loc == FRACTURE && el2.loc == FRACTURE)
						{
							Connection conn;
							conn.conn_id = counter;
							conn.pts_offset = offset;
							conn.n_pts = static_cast<uint8_t>(len);
							conn_nodes.insert(conn_nodes.end(), intersect.end() - len, intersect.end());
							conn.elem_id1 = el1.elem_id;
							conn.elem_id2 = el2.elem_id;
							conn.calculate_centroid(nodes, conn_nodes);
							conn.calculate_area(nodes, conn_nodes);
							conn.area *= ( init_apertures[el1.elem_id - region_ranges[FRACTURE].first] + init_apertures[el2.elem_id - region_ranges[FRACTURE].first] ) / 2.0;
							conn.calculate_normal(nodes, conn_nodes, elems, elem_nodes);
							id1[id_size1++] = el2.elem_id;
							offset += conn.n_pts;

							conn.type = CONN_TYPE_TABLE.at( { el1.loc, el2.loc } );
							conns.push_back(conn);
							counter++;
						}
						//else if(el2.loc == FRACTURE || el2.loc == FRACTURE_BOUNDARY)
					}
				}
			}
		}
	}

	elem_nodes_sorted.clear();

	//// erase matrix-matrix connections across faults
	// find the indices of remaining connections
	vector<index_t> conns_to_remain;
	conns_to_remain.reserve(region_ranges[FRACTURE].second - region_ranges[FRACTURE].first);
	auto it_faults = fault_nodes.cend();
	for (const auto& conn : conns)
	{
		if (conn.type == MAT_MAT)
		{
			set<index_t> cur(conn_nodes.begin() + conn.pts_offset, conn_nodes.begin() + conn.pts_offset + conn.n_pts);
			it_faults = fault_nodes.find(cur);
			if (it_faults == fault_nodes.end())
				conns_to_remain.push_back(conn.conn_id);
		}
		else
			conns_to_remain.push_back(conn.conn_id);
	}
	// copy what remains
	std::vector<Connection> new_conns;
	new_conns.reserve(conns_to_remain.size());
	std::vector<index_t> new_conn_nodes;
	new_conn_nodes.reserve(4 * MAX_CONNS_PER_ELEM_GMSH * num_of_elements);
	counter = offset = 0;
	for (const auto& conn_id: conns_to_remain)
	{
		auto& old_conn = conns[conn_id];
		// nodes
		new_conn_nodes.insert(new_conn_nodes.end(), conn_nodes.begin() + old_conn.pts_offset, conn_nodes.begin() + old_conn.pts_offset + old_conn.n_pts);
		// connection
		old_conn.conn_id = counter++;
		old_conn.pts_offset = offset;
		new_conns.push_back(old_conn);

		offset += old_conn.n_pts;
	}

	conns = new_conns;
	conn_nodes = new_conn_nodes;

	// save conn ids for each type
	conn_type_map[MAT_MAT].reserve(conns.size());
	conn_type_map[MAT_FRAC].reserve(conns.size());
	conn_type_map[MAT_BOUND].reserve(conns.size());
	conn_type_map[FRAC_FRAC].reserve(conns.size());
	conn_type_map[FRAC_BOUND].reserve(conns.size());

	for (const auto& conn : conns)
	  conn_type_map[conn.type].push_back(conn.conn_id);

	// erase absent
	for (auto it = conn_type_map.begin(); it != conn_type_map.end();)
	{
	  if (it->second.size())
		it++;
	  else
		it = conn_type_map.erase(it);
	}

	// sort boundary connections
	std::sort(conn_type_map[MAT_BOUND].begin(), conn_type_map[MAT_BOUND].end(), [&](const index_t& id1, const index_t& id2)
	  {
		return conns[id1].elem_id2 < conns[id2].elem_id2;
	  });

	t2 = steady_clock::now();
	cout << conns.size() << " connections:\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}


// uses: conns, num_of_elements
// fills: adj_matrix, adj_matrix_cols, adj_matrix_offset
void Mesh::generate_adjacency_matrix()
{
	steady_clock::time_point t1, t2;

	t1 = steady_clock::now();
	// Temporary arrays to identify all connections per element
	std::vector<std::vector<index_t>> adj_2d;
	std::vector<size_t> conn_per_element(num_of_elements, 0);
	std::vector<std::vector<bool>> conn_signs;
	// Reserve number of elements for number of rows in connection signs matrix
	// adjacency matrix
	adj_2d.resize(num_of_elements);
	conn_signs.resize(num_of_elements);


#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"       // GCC bug #108088: false positive in vector<bool>::reserve, fixed in GCC 14
#pragma GCC diagnostic ignored "-Wstringop-overflow"  // GCC bug #108088: same root cause, emitted under a different flag depending on optimisation level
#endif
	for (index_t i = 0; i < num_of_elements; i++)
	{
		adj_2d[i].reserve(MAX_CONNS_PER_ELEM);
		conn_signs[i].reserve(MAX_CONNS_PER_ELEM);
	}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

	// Append connections per element to 2D array
	for (auto& conn : conns)
	{
		auto& conn_size1 = conn_per_element[conn.elem_id1];
		adj_2d[conn.elem_id1].push_back(conn.conn_id);
		conn_signs[conn.elem_id1].push_back(true);
		conn_size1++;

		if (conn.type != MAT_BOUND && conn.type != FRAC_BOUND)
		{
			auto& conn_size2 = conn_per_element[conn.elem_id2];
			adj_2d[conn.elem_id2].push_back(conn.conn_id);
			conn_signs[conn.elem_id2].push_back(false);
			conn_size2++;
		}
	}

	// Reserve space
	adj_matrix.reserve(2 * conns.size());
	adj_matrix_cols.reserve(2 * conns.size());
	adj_matrix_offset.reserve(num_of_elements + 1);

	// First starts from zero
	index_t offset = 0;
	adj_matrix_offset.push_back(offset);

	// Flatten to 1D array and calculate offsets
	for (index_t i = 0; i < num_of_elements; i++)
	{
		offset += static_cast<index_t>(conn_per_element[i]);
		adj_matrix_offset.push_back(offset);
		const auto& elem_conns = adj_2d[i];
		const auto& elem_conn_signs = conn_signs[i];
		for (uint8_t j = 0; j < conn_per_element[i]; j++)
		{
			adj_matrix.push_back(elem_conns[j]);
			if (elem_conn_signs[j])
				adj_matrix_cols.push_back(conns[elem_conns[j]].elem_id2);
			else
				adj_matrix_cols.push_back(conns[elem_conns[j]].elem_id1);
		}
	}

	t2 = steady_clock::now();
	cout << "Adjacency matrix:\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}

// fills:
// nodes, elems, elems_of_node, elem_nodes, elem_nodes_sorted,
// elem_type_map, region_ranges, region_elems_num, //TODO
// volumes, centroids
// nx, ny, nz, bnd_faces_num2

// print internal arrays to screen (for debugging)
void Mesh::print_elems_nodes()
{
	cout << "Elements:\n";

	// loop through all elements
	for (index_t i = 0; i < num_of_elements; i++)
	{
		const auto& el1 = elems[i];

		cout << "\t id=" << el1.elem_id << " " << get_ijk_as_str(el1.elem_id, false) << " n_pts=" << int(el1.n_pts) << " pts_offset=" << el1.pts_offset << "\n";
		// loop through all nodes belonging to particular element
		for (index_t l = el1.pts_offset; l < el1.pts_offset + el1.n_pts; l++)
		{
			index_t node_id = elem_nodes[l];
			cout << "\t Node=" << node_id << " Elems:\t";
			// loop through all elements that share this node
			for (index_t k = elems_of_node_offset[node_id]; k < elems_of_node_offset[node_id + 1]; k++)
			{
				index_t nebr_id = elems_of_node[k];
				cout << nebr_id << " ";
			}
			cout << "\n";
		}
	}
}

//i,j,k are 0-based
void Mesh::calc_cell_nodes(const int i, const int j, const int k,
	std::array<double, 8>& X,
	std::array<double, 8>& Y,
	std::array<double, 8>& Z) const
{

	std::array<int, 8> zind;
	std::array<int, 4> pind;

	// calculate indices for grid pillars in COORD arrray
	const size_t p_offset = j * (nx + 1) * 6 + i * 6;

	pind[0] = p_offset;
	pind[1] = p_offset + 6;
	pind[2] = p_offset + (nx + 1) * 6;
	pind[3] = pind[2] + 6;

	// pind[0]-----pind[1]---> I-axis
	//   |           |
	//   |           |
	// pind[2]-----pind[3]
	//   |
	//   V
	// J-axis

	// get depths from zcorn array in ZCORN array
	const size_t z_offset = k * nx * ny * 8 + j * nx * 4 + i * 2;
	// top
	zind[0] = z_offset;
	zind[1] = z_offset + 1;
	zind[2] = z_offset + nx * 2;
	zind[3] = zind[2] + 1;
	// bottom
	for (int n = 0; n < 4; n++)
		zind[n + 4] = zind[n] + nx * ny * 4;

	for (int n = 0; n < 8; n++)
		Z[n] = zcorn[zind[n]];

	for (int n = 0; n < 4; n++) {
		// top
		double xt = coord[pind[n]];
		double yt = coord[pind[n] + 1];
		double zt = coord[pind[n] + 2];
		// bottom
		double xb = coord[pind[n] + 3];
		double yb = coord[pind[n] + 4];
		double zb = coord[pind[n] + 5];

		if (zt == zb) {
			X[n] = xt;
			X[n + 4] = xt;

			Y[n] = yt;
			Y[n + 4] = yt;
		}
		else {
			double t = (xb - xt) / (zt - zb);
			X[n]     = xt + t * (zt - Z[n]);
			X[n + 4] = xt + t * (zt - Z[n + 4]);

			t =  (yb - yt) / (zt - zb);
			Y[n]		 = yt + t * (zt - Z[n]);
			Y[n + 4] = yt + t * (zt - Z[n + 4]);
		}
	}
}

//i,j,k are 0-based
std::array<value_t, 3>
Mesh::calc_cell_sizes(const int i, const int j, const int k) const
{
	std::array<double, 8> X;
	std::array<double, 8> Y;
	std::array<double, 8> Z;

	calc_cell_nodes(i, j, k, X, Y, Z);

	//DEBUG
	//for (int ii = 0; ii < 8; ii++){
	//  cout << "(" << X[ii] << ", " << Y[ii] <<  ", " << Z[ii] << ")\n";
	//}
	double x1 = X[0] + X[2] + X[6] + X[4];
	double x2 = X[1] + X[3] + X[7] + X[5];
	double dx = fabs(x2 - x1) / 4.;

	double y1 = Y[2] + Y[3] + Y[7] + Y[6];
	double y2 = Y[0] + Y[1] + Y[5] + Y[4];
	double dy = fabs(y2 - y1) / 4.;

	double z1 = Z[0] + Z[1] + Z[2] + Z[3];
	double z2 = Z[4] + Z[5] + Z[6] + Z[7];
	double dz = fabs(z2 - z1) / 4.;

	std::array<value_t, 3> d{ dx, dy, dz };

	//DEBUG
	//cout << "cell (" << i << ", " << j << ", " << k << ") dx=" << d[0] << " dy=" << d[1] <<  " dz=" << d[2] << "\n";

	return d;
}

//
std::array<value_t, 3>
Mesh::calc_cell_center(const int i, const int j, const int k) const
{
	std::array<double, 8> X;
	std::array<double, 8> Y;
	std::array<double, 8> Z;

	calc_cell_nodes(i, j, k, X, Y, Z);

	double x1 = X[0] + X[2] + X[6] + X[4];
	double x2 = X[1] + X[3] + X[7] + X[5];
	double x = (x2 + x1) / 8.;

	double y1 = Y[2] + Y[3] + Y[7] + Y[6];
	double y2 = Y[0] + Y[1] + Y[5] + Y[4];
	double y = (y2 + y1) / 8.;

	double z1 = Z[0] + Z[1] + Z[2] + Z[3];
	double z2 = Z[4] + Z[5] + Z[6] + Z[7];
	double z = (z2 + z1) / 8.;

	std::array<value_t, 3> d{ x, y, z };

	//DEBUG
	//cout << "cell (" << i << ", " << j << ", " << k << ") Cx=" << d[0] << " Cy=" << d[1] <<  " Cz=" << d[2] << "\n";

	return d;
}

//
void Mesh::construct_local_global(std::vector<index_t> & global_cell)
{
  index_t n_all = get_n_cells_total();

	local_to_global.reserve(n_cells);
	global_to_local.resize(n_all);
#if 0
	index_t j = 0;
	for (index_t k1 = 0; k1 < nz; k1++)
		for (index_t j1 = 0; j1 < ny; j1++)
			for (index_t i1 = 0; i1 < nx; i1++)
			{
				index_t i = get_global_index(i1, j1, k1);
#else
	index_t gidx = 0, lidx = 0;
	for (auto gi : global_cell) {
#endif
		if (gidx && gi == 0) {// gi=0 means we run out active cells, but for the first cell zero index is OK
			while (gidx < n_all) {
				global_to_local[gidx++] = -1; // fill the rest array (inactive cells)
			}
			break;
		}

		// local_to_global is the same as global_cell except size (it doesn't contain zeros in the end)
		local_to_global.push_back(gi);

		while (gidx < n_all) { // infinite loop, added condition just in case
			if (gi == gidx) {
				global_to_local[gidx++] = lidx;
				break;
			}
			else { // store -1 for inactive cells
				global_to_local[gidx++] = -1;
			}
		}
		lidx++;
	}
}

// print centroids, volumes and cell sizes
void Mesh::print_arrays() const
{
	for (index_t idx = 0, counter = 0; idx < get_n_cells_total(); idx++) {
		if (!actnum[idx])
			continue;

		int i1, j1, k1;
		std::string ijk_str = get_ijk_as_str(idx, true);

		get_ijk(idx, i1, j1, k1, true);
		std::array<value_t, 3> d = calc_cell_sizes(i1, j1, k1);

		//if (i < 500 )
		cout<<idx<<" "<<ijk_str<<" C="<<centroids[counter]<<"\tV="<<volumes[counter]<<"\tDX="<<d[0]<<"\tDY="<<d[1]<<"\tDZ="<<d[2]<<"\n";

		counter++;
	}//loop by cells
}

//
void Mesh::write_cell_sizes(std::string fname) const
{
	std::vector<value_t> dx, dy, dz;
	for (size_t idx = 0; idx < get_n_cells_total(); ++idx) {
		int i1, j1, k1;
		get_ijk(idx, i1, j1, k1, true);
		std::array<value_t, 3> d = calc_cell_sizes(i1, j1, k1);
		dx[idx] = d[0];
		dy[idx] = d[1];
		dz[idx] = d[2];
	}
	std::vector<int> actnum;
	write_array_to_file(fname, "DX", dx, actnum, get_n_cells_total(), 1.0, true);
	write_array_to_file(fname, "DY", dy, actnum, get_n_cells_total(), 1.0, true);
	write_array_to_file(fname, "DZ", dz, actnum, get_n_cells_total(), 1.0, true);
}


// order: Y1, Y2, X1, X2, Z2, Z1
std::vector<value_t>
Mesh::get_prisms() const
{
	std::array<double, 8> X;
	std::array<double, 8> Y;
	std::array<double, 8> Z;

	std::vector<value_t> prisms;
	prisms.resize(6 * n_cells);
	int idx_cell = 0, idx = 0;

  for (int k = 0; k < nz; k++)
		for (int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++) {
				if (global_to_local[idx_cell++] < 0) // cell is inactive
					continue;
				calc_cell_nodes(i, j, k, X, Y, Z);

				//DEBUG
				//for (int ii = 0; ii < 8; ii++){
				//  cout << "(" << X[ii] << ", " << Y[ii] <<  ", " << Z[ii] << ")\n";
				//}
				double x1 = X[0] + X[2] + X[6] + X[4];
				double x2 = X[1] + X[3] + X[7] + X[5];
				double dx = fabs(x2 - x1) / 4.;

				double y1 = Y[2] + Y[3] + Y[7] + Y[6];
				double y2 = Y[0] + Y[1] + Y[5] + Y[4];
				double dy = fabs(y2 - y1) / 4.;

				double z1 = Z[0] + Z[1] + Z[2] + Z[3];
				double z2 = Z[4] + Z[5] + Z[6] + Z[7];
				double dz = fabs(z2 - z1) / 4.;

				prisms[idx++] = Y[0];
				prisms[idx++] = Y[0] + dy;
				prisms[idx++] = X[0];
				prisms[idx++] = X[0] + dx;
				prisms[idx++] = Z[0] + dz;
				prisms[idx++] = Z[0];
			}

	//cout <<"centers: " << idx_cell << " " << idx / 6 << '\n';
	return prisms;
}

// prder: YXZ
std::vector<value_t>
Mesh::get_centers() const
{
	std::vector<value_t> centers_vec;
	centers_vec.resize(3 * n_cells);

	int idx = 0, idx_cell = 0;
	for (auto c : centroids) {
		if (idx_cell >= n_cells) //skip bnd faces in the end of centroids
			break;
		centers_vec[idx++] = c.y;
		centers_vec[idx++] = c.x;
		centers_vec[idx++] = c.z;
		idx_cell++;
	}

	return centers_vec;
}

// compute node coords for vtk output
std::vector<value_t>
Mesh::get_nodes_array() const
{
	std::array<double, 8> X;
	std::array<double, 8> Y;
	std::array<double, 8> Z;

	std::vector<value_t> nodes_array;
	nodes_array.resize(8 * 3 * n_cells); // 8 nodes per cell, 3 coordinates (x,y,z) per node

	int idx_cell = 0, idx = 0;
	std::array<int, 8> permute = { 0,4,5,1,2,6,7,3 }; // need this order for vtk
	for (int k = 0; k < nz; k++)
		for (int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++) {
				if (global_to_local[idx_cell++] < 0) // cell is inactive => skip
					continue;
				calc_cell_nodes(i, j, k, X, Y, Z);

				index_t start = idx * 8 * 3;
				for (int ii = 0; ii < 8; ii++) {
					index_t jj = permute[ii];
					nodes_array[start + ii * 3 + 0] = X[jj];
					nodes_array[start + ii * 3 + 1] = Y[jj];
					nodes_array[start + ii * 3 + 2] = Z[jj];
				}
				idx++;
			}
	return nodes_array;
}

std::vector<int>
Mesh::cpg_elems_nodes(
	const int _number_of_nodes,
	const int num_of_cells,// number of active cells
	const int number_of_faces,
	const std::vector<double>& node_coords,
	const std::vector<int>& face_nodes,
	const std::vector<int>& face_nodepos,
	const std::vector<int>& face_cells,
	const std::vector<int>& cell_faces,
	const std::vector<int>& cell_facepos,
	const std::vector<double>& cell_volumes,
	std::vector<int>& face_order)
{
	mesh_type = CPG;

	num_of_nodes = _number_of_nodes;
	nodes.reserve(num_of_nodes);

	// fill nodes coordinates (x,y,z)
	for (index_t i = 0; i < num_of_nodes * ND; i += ND) {
		nodes.emplace_back(node_coords[i], node_coords[i + 1], node_coords[i + 2]);
		//cout << "Node " << i / ND << ": (" << node_coords[i] << ", " << node_coords[i + 1] << ", " << node_coords[i + 2] << ")\n";
	}

	// create elements for cells, faces will be processed later

	// fill elem_nodes and
	// collect nodes that belong to each face or cell, using intermediate sets to remove duplicates
	std::map<int, std::set<index_t>> elems_of_node_set; // map<node, elem_face_set> - use set to make elements from nodes unique
	std::map<int, std::set<index_t>> face_nodes_set; // map<face, node_set> - use set to make nodes from faces unique

	// calculate temporary array shift_faces
	// will be used for shift faces indexes to remove internal (non-boundary) faces from faces indexation
	int bnd_faces_num = 0;
	for (int f = 0, non_bnd_faces_counter = 0; f < cell_facepos[num_of_cells]; ++f) {
		int face = cell_faces[f];
		bool face_is_boundary = face_cells[2 * face] < 0 || face_cells[2 * face + 1] < 0;
		//cout << "Face " << face << " is boundary: " << face_is_boundary << "\n";
		bool face_processed = face_nodes_set.find(face) != face_nodes_set.end(); // process the face only once
		if (face_processed) // if we have already met this face from the previous cells, skip it to avoid duplicate work
			continue;
		face_nodes_set[face]; // add face to dict

		if (!face_is_boundary) {
			non_bnd_faces_counter++;
		}
		else
			bnd_faces_num++;
	}// loop by all faces

	face_nodes_set.clear();
	size_t initial_points_per_element = 8;
	num_of_elements = num_of_cells + bnd_faces_num;
	elems.resize(num_of_elements);
	element_tags.resize(num_of_elements);
	elem_nodes.reserve(initial_points_per_element * num_of_elements);	// Remove the max points per 3D element and change it to 8 nodes
	elem_nodes_sorted.reserve(initial_points_per_element * num_of_elements); // Remove the max points per 3D element and change it to 8 nodes
	volumes.resize(num_of_elements);
	centroids.resize(num_of_elements);

	index_t face_counter = 0, counter = 0, offset = 0, cell_offset = 0;
	for (index_t c = 0; c < num_of_cells; ++c) {
		std::set<index_t> cell_nodes_set; // use set to make nodes from faces unique
		//cout << "Cell " << c << "\n";
		// for each cell go through it's faces
		for (int f = cell_facepos[c]; f < cell_facepos[c + 1]; ++f) {
			int face = cell_faces[f];
			//int face_2 = face - shift_faces[face];
			bool face_processed = face_nodes_set.find(face_counter) != face_nodes_set.end(); // process the face only once
			if (face_processed) // if we have already met this face from the previous cells, skip it to avoid duplicate work
				continue;
			bool face_is_boundary = face_cells[2 * face] < 0 || face_cells[2 * face + 1] < 0;
			std::set<index_t>* f_nodes = 0;
			if (face_is_boundary) {
				f_nodes = &face_nodes_set[face_counter];
			}
			//if (face_is_boundary) cout << "\t Face " << face << " (" << num_of_cells + face_counter << "). Nodes: ";
			//for each face go through it's nodes
			for (int k = face_nodepos[face]; k < face_nodepos[face + 1]; ++k) {
				int node = face_nodes[k];
				cell_nodes_set.insert(node);
				elems_of_node_set[node].insert(c);
				if (face_is_boundary) {// need only boundary faces
					f_nodes->insert(node);
					elems_of_node_set[node].insert(num_of_cells + face_counter);// faces after cells
				}
				//if (face_is_boundary) cout << node << " ";
			} // loop by nodes of face

			// fill elems by boundary faces
			if (!face_processed && face_is_boundary) {
				// element
				Elem el;
				el.loc = BOUNDARY;
				el.type = QUAD;
				el.elem_id = num_of_cells + face_counter;
				el.n_pts = static_cast<uint8_t>(f_nodes->size());
				element_tags[el.elem_id] = 0;

				elems[num_of_cells + face_counter] = el;
				face_order.push_back(face);
				face_counter++;
			}
			//if (face_is_boundary) cout << "\n";
		} // loop by faces of cell

		// fill elems by cell
		Elem el;
		el.loc = MATRIX;
		el.type = HEX;
		el.elem_id = counter;
		el.n_pts = static_cast<uint8_t>(cell_nodes_set.size());
		element_tags[el.elem_id] = 0;
		elems[counter] = el;

		counter++;

		// add cells to elem_nodes
#if 0 // for volume calc by gmsh method
		// need a specific order to properly calculate geometry (volume)
		std::vector<index_t> cell_nodes_vec(cell_nodes_set.begin(), cell_nodes_set.end());
		elem_nodes.push_back(cell_nodes_vec[0]);
		elem_nodes.push_back(cell_nodes_vec[2]);
		elem_nodes.push_back(cell_nodes_vec[6]);
		elem_nodes.push_back(cell_nodes_vec[4]);
		elem_nodes.push_back(cell_nodes_vec[1]);
		elem_nodes.push_back(cell_nodes_vec[3]);
		elem_nodes.push_back(cell_nodes_vec[7]);
		elem_nodes.push_back(cell_nodes_vec[5]);
#else
		for (auto n : cell_nodes_set)
			elem_nodes.push_back(n);
#endif
		for (auto n : cell_nodes_set) //set is already sorted
			elem_nodes_sorted.push_back(n);
	} // loop by cells

	offset = 0;
	for (auto& el : elems) {
		el.pts_offset = offset;
		offset += el.n_pts;
	}

	// add boundary faces to elem_nodes
	//for (auto face_nodes_i : face_nodes_set) {
	//for (auto face_2 : face_order) {
	for (int face_2 = 0; face_2 < face_order.size(); face_2++) {
		auto face_nodes_i = &face_nodes_set[face_2];
		// need a specific order to properly calculate geometry (area)
		//std::vector<index_t> face_nodes_vec(face_nodes_i.second.begin(), face_nodes_i.second.end());
#if 0 // for volume calc by gmsh method
		std::vector<index_t> face_nodes_vec(face_nodes_i->begin(), face_nodes_i->end());
		elem_nodes.push_back(face_nodes_vec[0]);
		elem_nodes.push_back(face_nodes_vec[1]);
		elem_nodes.push_back(face_nodes_vec[3]);
		elem_nodes.push_back(face_nodes_vec[2]);
#else
		for (auto n : *face_nodes_i) {
			elem_nodes.push_back(n);
		}
#endif
		for (auto n : *face_nodes_i)//set is already sorted
			elem_nodes_sorted.push_back(n);
	}

	cout << "num_of_elements: " << num_of_elements << "\n";
	cout << "num_of_cells:    " << nx * ny * nz << "\n";
	cout << "active_cells:    " << num_of_cells << "\n";
	cout << "number_of_faces: " << number_of_faces << "\n";
	cout << "bnd_faces_num:   " << bnd_faces_num << "\n";

	// fill elems_of_node, elems_of_node_offset
	elems_of_node_offset.reserve(num_of_nodes * 8); // This is estimated value. Internal node belongs to 8 cells. Boundary faces are not counted.
	size_t n_elems_accum = 0;
	elems_of_node_offset.push_back(0);

	index_t max_node_id = *std::max_element(face_nodes.begin(), face_nodes.end());

	for (index_t node_id = 0; node_id <= max_node_id; node_id++)
	{
		if (elems_of_node_set.find(node_id) != elems_of_node_set.end())
		{
			const auto& elems = elems_of_node_set.at(node_id);
			for (const auto& n : elems)
			{
				elems_of_node.push_back(n);
			}
			n_elems_accum += elems.size();
		}
		elems_of_node_offset.push_back(static_cast<index_t>(n_elems_accum));
	}

#if 0 //works wrong in fault case because number of points for cell might be > 8
	// calculate the volume and centroid for the element
	for (auto el : elems) {
		el.calculate_volume_and_centroid(nodes, elem_nodes, volumes[el.elem_id], centroids[el.elem_id]);

#ifdef DEBUG_TRANS
		cout << "CPP CPG ID=" << el.elem_id << ", L= " << el.loc << ", T= " << el.type << ", V= " << volumes[el.elem_id] << ", C= " << centroids[el.elem_id] << "\n";
#endif // DEBUG_TRANS
	}//loop by elements
#endif //0

	std::vector<int> result;
	result.push_back(bnd_faces_num);
	return result;
}


// uses: bnd_faces_num, face_order array
// fills: cell volumes, cell centroids, cell depths
void Mesh::cpg_cell_props(
	const int _number_of_nodes,
	const int num_of_cells,// number of active cells
	const int number_of_faces,
	const std::vector<double>& cell_volumes,
	const std::vector<double>& cell_centroids,
	const std::vector<int>& global_cell,
	const std::vector<double>& face_areas,
	const std::vector<double>& face_centroids,
	const int bnd_faces_num,
	const std::vector<int>& face_order)
{
	volumes.resize(num_of_cells + bnd_faces_num);
	centroids.resize(num_of_cells + bnd_faces_num);
	depths.resize(num_of_cells + bnd_faces_num);

	//fill cell properties. cell ordering in OPM and discretizer is the same
	for (int i = 0, j = 0; i < num_of_cells; i++) {
		volumes[j] = fabs(cell_volumes[i]);
#if 0 //baricenters
		centroids[j] = Vector3(cell_centroids[3 * i], cell_centroids[3 * i + 1], cell_centroids[3 * i + 2]);
#else //simple centers
		int i1, j1, k1;
		get_ijk(i, i1, j1, k1, false);
		std::array<value_t, 3> center = calc_cell_center(i1, j1, k1);
		centroids[j] = Vector3(center[0], center[1], center[2]);
#endif
		depths[j] = centroids[j].z;
		j++;
	}

	//fill face properties, the order in OPM and discretizer is different, so use face_order
	for (int i = 0; i < bnd_faces_num; i++) {
		int j = face_order[i];
		volumes[num_of_cells + i] = fabs(face_areas[j]);
		centroids[num_of_cells + i] = Vector3(face_centroids[3 * j], face_centroids[3 * j + 1], face_centroids[3 * j + 2]);
		depths[num_of_cells + i] = face_centroids[3 * j + 2];
	}

}


// fills: conns
// difference from prev. method: two centers `c` and `c_2` for the connection are calculated
// centers computed for original face, not the splitted face because of fault
// these centers will used in compute half-trans to make this consistent with reservoir simulators
void Mesh::cpg_connections(
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
	const PhysicalTags& tags)
{
	index_t nebr_id, counter = 0, offset = 0, node_id;
	std::map<int, int> face_nodes_set; // map<face, node_set> - use set to make nodes from faces unique
	conns.reserve(number_of_faces);
	weno_face_sides.clear();
	weno_face_nodes.clear();
	weno_cell_face_offset.clear();
	weno_face_sides.reserve(static_cast<size_t>(cell_faces.size()));
	weno_face_nodes.reserve(static_cast<size_t>(face_nodes.size()) * 2);
	weno_cell_face_offset.reserve(static_cast<size_t>(num_of_cells) + 1);
	weno_cell_face_offset.push_back(0);

	std::vector<double_t> x, y, z; // only for the current face direction (face_tag)
	x.reserve(100); y.reserve(100); z.reserve(100);

	conn_type_map[MAT_MAT].reserve(number_of_faces);
	conn_type_map[MAT_FRAC].reserve(number_of_faces);
	conn_type_map[MAT_BOUND].reserve(number_of_faces);
	conn_type_map[FRAC_FRAC].reserve(number_of_faces);
	conn_type_map[FRAC_BOUND].reserve(number_of_faces);

	for (index_t c = 0; c < num_of_cells; ++c) {

		//cout << "Cell " << c << "\n";
		std::set<index_t> cell_nodes_set; // use set to make nodes from faces unique
		std::map<int, Vector3> local_face_centers;

		// fill local_face_nodes node vector for each face_tag
		std::map<int, std::vector<int>> local_face_nodes;
		bool first_face = true;
		// for each cell go through it's faces
		for (int f = cell_facepos[c]; f < cell_facepos[c + 1]; ++f) {
			int face = cell_faces[f];
			int face_tag = cell_facetag[f];

			//for each face go through it's nodes and fill local_face_nodes
			for (int k = face_nodepos[face]; k < face_nodepos[face + 1]; ++k) {
				local_face_nodes[face_tag].push_back(face_nodes[k]);
			}
		}

		// fill node coordinates and calc center for 'original' non-splitted face
		for (auto& faces_dir : local_face_nodes) { // loop by face sides (X+, X-, Y+,..)
			x.clear(); y.clear(); z.clear();
			// collect all nodes from faces with the same side (face_tag)
			int cc = 0;
			int c_nodes = faces_dir.second.size();
			for (auto& node_idx : faces_dir.second) { // loop by nodes of face
				// for cell sides which have more than 4 nodes (cell side with more that 1 neighbour)
				// use only first 2 (top) and last 2 (bottom) nodes to make original face
				if (c_nodes > 4 && (cc > 1 && cc < c_nodes - 2)) {
					cc++;
					continue;
				}
				else
					cc++;
				x.push_back(nodes[node_idx].x);
				y.push_back(nodes[node_idx].y);
				z.push_back(nodes[node_idx].z);
			}

			// take two top and two bottom points
			// calculate the 'original' face center as arithmetic mean of these 4 nodes
			Vector3 center_orig_face;
			for (int ii = 0; ii < x.size(); ii++) {
				center_orig_face.x += x[ii];
				center_orig_face.y += y[ii];
				center_orig_face.z += z[ii];
			}
			center_orig_face /= x.size();

			local_face_centers[faces_dir.first] = center_orig_face;
		} // face dirs loop
	//}// faces loop

	// for each cell go through its faces
		for (int f = cell_facepos[c]; f < cell_facepos[c + 1]; ++f) {
			int face = cell_faces[f];
			int face_tag = cell_facetag[f];
			bool face_processed = face_nodes_set.find(face) != face_nodes_set.end(); // process the face only once

				node_id = face_cells[2 * face];
				nebr_id = face_cells[2 * face + 1];
				bool face_is_boundary = node_id < 0 || nebr_id < 0;

				WenoFaceSide weno_side;
				weno_side.cell = c;
				weno_side.logical_face = face_tag;
				weno_side.node_offset = static_cast<index_t>(weno_face_nodes.size());
				weno_side.node_count = face_nodepos[face + 1] - face_nodepos[face];
				// Use the target-cell 'original' (un-split) face centre, matching
				// conn.c/c_2 and the discretizer TPFA quadrature point, rather than
				// the shared OPM subface centroid which differs on split/fault faces.
				weno_side.centroid = local_face_centers[face_tag];
				weno_side.area = std::abs(face_areas[face]);
				for (int k = face_nodepos[face]; k < face_nodepos[face + 1]; ++k)
					weno_face_nodes.push_back(face_nodes[k]);
				if (face_is_boundary)
				{
					weno_side.neighbour = -1;
					weno_side.connection_id = -1;
				}
				else
				{
					weno_side.neighbour = node_id == c ? nebr_id : node_id;
					weno_side.connection_id = face_processed ? face_nodes_set[face] : counter;
				}
				weno_face_sides.push_back(weno_side);

				if (face_is_boundary)
					continue;

			// skip the same element
			if (nebr_id == node_id)
				continue;

			if (face_processed) // if we have already met this face from the neighbour cell:
				// 1) no need to add new connection
				// 2) put the second face center to existing connection
			{
				// `face` is global face index, so use `face_nodes_set[face]` to find the Connection which has been already added by the neighbour cell
				Connection& conn_tmp = conns[face_nodes_set[face]];
				conn_tmp.c_2 = local_face_centers[face_tag]; // `conn_tmp.c` has been already added because `face_processed=true`)
				index_t c_id = conn_tmp.elem_id1 == c ? conn_tmp.elem_id2 : conn_tmp.elem_id1;
				if (c_id > c) // take care on which of two neighbour cells uses `c`, and which one uses `c_2` (should be consistend with discretizer calc_trans)
					std::swap(conn_tmp.c, conn_tmp.c_2); // cell with greater index uses c_2
				continue;
			}

			face_nodes_set[face] = counter; // add face to dict
			Connection conn;
			conn.n_pts = static_cast<uint8_t>(face_nodepos[face + 1] - face_nodepos[face]);
			conn.conn_id = counter;
			conn.pts_offset = offset;
			offset += conn.n_pts;
			counter++;

			//for each face go through it's nodes
			for (int k = face_nodepos[face]; k < face_nodepos[face + 1]; ++k) {
				conn_nodes.push_back(face_nodes[k]);
			}
			conn.elem_id1 = node_id;
			conn.elem_id2 = nebr_id;

			conn.c = local_face_centers[face_tag];
			conn.n = Vector3(face_normals[3 * face], face_normals[3 * face + 1], face_normals[3 * face + 2]);
			conn.n /= conn.n.norm();

			conn.area = face_areas[face];

			//cout << "Conn " << conn.elem_id1 << " " << conn.elem_id2 << " Face " << f << " area= " << conn.area << "\n";


			// Find connection type
			conn.type = face_is_boundary ? MAT_BOUND : MAT_MAT;
			conn_type_map[conn.type].push_back(counter - 1);

			conns.push_back(conn);
		}//faces loop

			weno_cell_face_offset.push_back(static_cast<index_t>(weno_face_sides.size()));
		}//cells loop

	// erase absent
	for (auto it = conn_type_map.begin(); it != conn_type_map.end();)
	{
	  if (it->second.size())
		it++;
	  else
		it = conn_type_map.erase(it);
	}

	cout << conns.size() << " connections:\n";
}
