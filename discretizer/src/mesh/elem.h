#ifndef ELEM_H_
#define ELEM_H_

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <set>
#include "linalg/matrix.h"
#include "linalg/vector3.h"

namespace mesh
{
	using linalg::index_t;
	using linalg::value_t;
	using linalg::Vector3;
	typedef linalg::Matrix<value_t> Matrix;

	// spatial dimension
	static const uint8_t ND = linalg::ND;

	Vector3 triangle_centroid_area(const std::vector<Vector3>& nodes, double *cell_area);
	Vector3 tetra_centroid(const std::vector<Vector3>& nodes);
	double tetra_volume(const std::vector<Vector3>& nodes);


	/***********	Elements	***************/
	const index_t MAX_PTS_PER_3D_ELEM = 8; // used to reserve memory. Normally is 8 but for juxtaposed faults (CPG) might be greater in case there are non-neighbour connections
	const index_t MAX_PTS_NUM_PER_2D_ELEM = 4; // not used
	const index_t MAX_PTS_PER_3D_ELEM_GMSH = 8; // used to reserve memory
	const index_t MIN_CONNS_PER_ELEM = 1;
	const index_t MAX_CONNS_PER_ELEM_GMSH = 8;
	const index_t MAX_CONNS_PER_ELEM = 6; //normally is 6 but for the faults (CPG) might be greater 
	const index_t PTS_NUM_1D_ELEM = 2;
	// gmsh element types
	enum ElemType { LINE = 1, TRI = 2, QUAD = 3, TETRA = 4, HEX = 5, PRISM = 6, PYRAMID = 7 };
	// element locations
	enum ElemLoc { FRACTURE_BOUNDARY = 0, BOUNDARY = 1, FRACTURE = 2, MATRIX = 3, WELL = 4 };
	// number of points belonging to certain element type 
	const std::unordered_map<ElemType, uint8_t> Etype_PTS = { {LINE, 2}, {TRI, 3}, {QUAD, 4}, {TETRA, 4}, {HEX, 8}, {PRISM, 6}, {PYRAMID, 5} };
	const std::unordered_map<uint8_t, ElemType> PTS_Etype_2D = { {2, LINE }, {3, TRI }, {4, QUAD} };
	const std::unordered_map<uint8_t, ElemType> PTS_Etype_3D = { {4, TETRA}, {8, HEX}, {6, PRISM}, {5, PYRAMID} };
	// storing elements in the following order
	const std::array<ElemLoc, 4> elem_order = { MATRIX, FRACTURE, BOUNDARY, FRACTURE_BOUNDARY };

	// element class
	class Elem
	{
	public:
		// element location
		ElemLoc loc;
		// geometrical type
		ElemType type;
		// number of points
		index_t n_pts;
		// starting position of 'n_pts' nodes in 'elem_nodes' array
		index_t pts_offset;
		// id of element
		index_t elem_id;

		Elem() {};
		Elem(ElemType _type, index_t _elem_id, index_t _pts_offset) : Elem()
		{
			type = _type;
			n_pts = Etype_PTS.at(type);
			pts_offset = _pts_offset;
		};

		void calculate_centroid(const std::vector<Vector3>& nodes, const std::vector<index_t>& elem_nodes, Vector3 &c);
		void calculate_volume_and_centroid(const std::vector<Vector3>& nodes, const std::vector<index_t>& elem_nodes, value_t &volume, Vector3 &c);

		~Elem() { };
	};

	/***********	Connections		************/
	enum ConnType { MAT_MAT = 0, MAT_BOUND = 1, MAT_FRAC = 2, FRAC_MAT = 3, FRAC_FRAC = 4, FRAC_BOUND = 5 };
	class Connection
	{
	public:
		ConnType type;
		uint8_t n_pts;
		index_t conn_id;
		index_t elem_id1;
		index_t elem_id2;
		index_t pts_offset;

		Vector3 n, c;
		Vector3 c_2; // center for the second cell (differs in fault case)
		value_t area;

		Connection() : area(0.0) { };

		void calculate_centroid(const std::vector<Vector3>& nodes, const std::vector<index_t>& conn_nodes);
		void calculate_area(const std::vector<Vector3>& nodes, const std::vector<index_t>& conn_nodes);
		void calculate_normal(const std::vector<Vector3>& nodes, const std::vector<index_t>& conn_nodes, const std::vector<mesh::Elem>& elems, const std::vector<index_t>& elem_nodes);

		~Connection() { };
	};

	/***********	hash & comparator for connections		***************/
	struct pair_xor_hash
	{
		template <class T1, class T2>
		std::size_t operator() (const std::pair<T1, T2> &pair) const 
		{
		  return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
		}
	};
	struct pair_cantor_hash 
	{
	  std::uint64_t operator()(const std::pair<index_t, index_t>& p) const 
	  {
		// Ensure the pair is ordered
		index_t a = std::min(p.first, p.second);
		index_t b = std::max(p.first, p.second);

		// Apply the Cantor pairing function
		return static_cast<std::uint64_t>((0.5 * (a + b) * (a + b + 1)) + b);
	  }
	};
	struct integer_set_hash
	{
		std::size_t operator() (const std::set<index_t>& data) const {
			std::set<index_t>::iterator it = data.begin();
			std::size_t res = *it;
			++it;
			while (it != data.end()) 
			{
				res *= 1000003;
				res += *it;
				++it;
			}
			return res;
		}
	};
	struct one_way_connection_comparator
	{
		template <class T>
		bool operator()(std::pair<T, T> const& x, std::pair<T, T> const& y) const
		{
			return x == y || (x.first == y.second && x.second == y.first);
		}
	};
	struct two_way_connection_comparator
	{
		template <class T>
		bool operator()(std::pair<T, T> const& x, std::pair<T, T> const& y) const
		{
			return x == y;
		}
	};

	// mapping from <ElemLoc, ElemLoc> -> ConnType
	typedef std::unordered_map<std::pair<ElemLoc, ElemLoc>, ConnType, pair_xor_hash, two_way_connection_comparator> ElemConnectionTable;
	const ElemConnectionTable CONN_TYPE_TABLE = { { { MATRIX, MATRIX				}, MAT_MAT },
												  { { FRACTURE, MATRIX				}, FRAC_MAT },
												  { { MATRIX, FRACTURE				}, MAT_FRAC },
												  { { MATRIX, BOUNDARY				}, MAT_BOUND },
												  {	{ FRACTURE, FRACTURE			}, FRAC_FRAC },
												  { { FRACTURE, FRACTURE_BOUNDARY	}, FRAC_BOUND } };
};

#endif /* ELEM_H_ */
