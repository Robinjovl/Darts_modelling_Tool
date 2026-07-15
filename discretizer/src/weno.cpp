#include "discretizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace
{
using dis::Matrix;
using linalg::index_t;
using linalg::value_t;
using linalg::Vector3;

constexpr index_t WENO_DIM = 3;

struct Candidate
{
	std::array<index_t, WENO_DIM> support_cell{};
	std::array<index_t, WENO_DIM> support_kind{};
	std::array<index_t, WENO_DIM> face_key{};
	std::array<value_t, WENO_DIM * WENO_DIM> inverse{};
	value_t gamma = 0.0;
	value_t condition = 0.0;
};

struct CellGeometry
{
	bool transform_valid = false;
	std::array<value_t, WENO_DIM * WENO_DIM> transform{};
	std::vector<Candidate> candidates;
};

struct GeometricConnection
{
	index_t id = -1;
	index_t engine_cell_a = -1;
	index_t engine_cell_b = -1;
	std::array<value_t, WENO_DIM> reference_a{};
	std::array<value_t, WENO_DIM> reference_b{};
	bool valid_a = false;
	bool valid_b = false;
};

value_t determinant3(const std::array<value_t, 9>& a)
{
	return a[0] * (a[4] * a[8] - a[5] * a[7])
		 - a[1] * (a[3] * a[8] - a[5] * a[6])
		 + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

bool inverse3(const std::array<value_t, 9>& a, std::array<value_t, 9>& inverse,
			  value_t& determinant, value_t& condition)
{
	determinant = determinant3(a);
	if (!std::isfinite(determinant) || std::abs(determinant) <= 1.e-13)
		return false;

	const value_t inv_det = 1.0 / determinant;
	inverse = {
		(a[4] * a[8] - a[5] * a[7]) * inv_det,
		(a[2] * a[7] - a[1] * a[8]) * inv_det,
		(a[1] * a[5] - a[2] * a[4]) * inv_det,
		(a[5] * a[6] - a[3] * a[8]) * inv_det,
		(a[0] * a[8] - a[2] * a[6]) * inv_det,
		(a[2] * a[3] - a[0] * a[5]) * inv_det,
		(a[3] * a[7] - a[4] * a[6]) * inv_det,
		(a[1] * a[6] - a[0] * a[7]) * inv_det,
		(a[0] * a[4] - a[1] * a[3]) * inv_det};

	value_t norm_a_sq = 0.0;
	value_t norm_inverse_sq = 0.0;
	for (index_t i = 0; i < 9; ++i)
	{
		norm_a_sq += a[i] * a[i];
		norm_inverse_sq += inverse[i] * inverse[i];
		if (!std::isfinite(inverse[i]))
			return false;
	}
	condition = std::sqrt(norm_a_sq * norm_inverse_sq);
	return std::isfinite(condition);
}

std::array<value_t, WENO_DIM> transform_point(
	const std::array<value_t, WENO_DIM * WENO_DIM>& transform,
	const Vector3& displacement)
{
	return {
		transform[0] * displacement.x + transform[1] * displacement.y + transform[2] * displacement.z,
		transform[3] * displacement.x + transform[4] * displacement.y + transform[5] * displacement.z,
		transform[6] * displacement.x + transform[7] * displacement.y + transform[8] * displacement.z};
}

bool build_reference_transform(const std::vector<Vector3>& displacements,
						   const value_t condition_limit,
						   std::array<value_t, 9>& transform)
{
	if (displacements.size() < WENO_DIM)
		return false;

	Matrix x(static_cast<index_t>(displacements.size()), WENO_DIM);
	for (index_t row = 0; row < static_cast<index_t>(displacements.size()); ++row)
	{
		x(row, 0) = displacements[row].x;
		x(row, 1) = displacements[row].y;
		x(row, 2) = displacements[row].z;
	}
	Matrix v(WENO_DIM, WENO_DIM);
	std::valarray<value_t> singular_values(WENO_DIM);
	if (!x.svd(v, singular_values))
		return false;

	value_t min_singular = std::numeric_limits<value_t>::max();
	value_t max_singular = 0.0;
	for (index_t d = 0; d < WENO_DIM; ++d)
	{
		const value_t singular = std::abs(singular_values[d]);
		if (!std::isfinite(singular) || singular <= 1.e-13)
			return false;
		min_singular = std::min(min_singular, singular);
		max_singular = std::max(max_singular, singular);
	}
	if (max_singular / min_singular > condition_limit)
		return false;

	// X = U Sigma V^T and y = Sigma^-1 V^T (x - x_i).
	for (index_t d = 0; d < WENO_DIM; ++d)
		for (index_t xyz = 0; xyz < WENO_DIM; ++xyz)
			transform[d * WENO_DIM + xyz] = v(xyz, d) / singular_values[d];
	return true;
}

bool candidate_less(const Candidate& left, const Candidate& right)
{
	return std::tie(left.support_cell, left.support_kind, left.face_key)
		 < std::tie(right.support_cell, right.support_kind, right.face_key);
}

// Canonical geometric signature of a candidate: the sorted set of its support
// points and value sources.  For an internal (CELL) support the geometry is
// fixed by the neighbour cell id, so the traversed face is irrelevant; for a
// no-flow (TARGET_COPY) support the point is the boundary-face centroid, so the
// face must be kept.  Two candidates with equal signatures are geometrically
// identical and must not both contribute their linear (volume) weight.
std::array<std::array<index_t, 3>, WENO_DIM> candidate_geometry_key(const Candidate& candidate)
{
	std::array<std::array<index_t, 3>, WENO_DIM> key{};
	for (index_t support = 0; support < WENO_DIM; ++support)
		key[support] = {candidate.support_cell[support], candidate.support_kind[support],
			candidate.support_kind[support] == dis::WENO_SUPPORT_TARGET_COPY
				? candidate.face_key[support] : -1};
	std::sort(key.begin(), key.end());
	return key;
}
}

void dis::Discretizer::prepare_weno_static(
	const std::vector<index_t>& discretizer_to_engine,
	index_t n_engine_cells,
	const std::vector<index_t>& engine_block_m,
	const std::vector<index_t>& engine_block_p,
	value_t condition_limit,
	index_t max_candidates)
{
	if (mesh == nullptr)
		throw std::invalid_argument("A mesh must be assigned before WENO preprocessing");
	if (n_engine_cells <= 0 || engine_block_m.size() != engine_block_p.size())
		throw std::invalid_argument("Invalid WENO engine cell or connection arrays");
	if (static_cast<index_t>(discretizer_to_engine.size()) < mesh->n_cells)
		throw std::invalid_argument("discretizer_to_engine does not cover all discretizer cells");
	if (condition_limit <= 1.0 || max_candidates <= 0)
		throw std::invalid_argument("Invalid WENO condition limit or candidate cap");
	if (mesh->weno_cell_face_offset.size() != static_cast<size_t>(mesh->n_cells + 1))
		throw std::invalid_argument("Mesh does not contain a complete WENO face-side view");

	std::vector<index_t> engine_to_discretizer(n_engine_cells, -1);
	for (index_t cell = 0; cell < mesh->n_cells; ++cell)
	{
		const index_t engine_cell = discretizer_to_engine[cell];
		if (engine_cell < 0)
			continue;
		if (engine_cell >= n_engine_cells || engine_to_discretizer[engine_cell] >= 0)
			throw std::invalid_argument("discretizer_to_engine is not a one-to-one in-range mapping");
		engine_to_discretizer[engine_cell] = cell;
	}

	std::vector<CellGeometry> geometry(mesh->n_cells);
	std::vector<std::vector<Candidate>> candidates_by_engine_cell(n_engine_cells);

	for (index_t cell = 0; cell < mesh->n_cells; ++cell)
	{
		const index_t engine_cell = discretizer_to_engine[cell];
		if (engine_cell < 0 || engine_cell >= n_engine_cells ||
			cell >= static_cast<index_t>(mesh->elems.size()) || mesh->elems[cell].loc != mesh::MATRIX)
			continue;

		// One compact support face per logical direction.  On CPG split faces,
		// deterministically retain the largest-area subface.
		std::map<index_t, index_t> selected_by_logical_face;
		for (index_t side_id = mesh->weno_cell_face_offset[cell];
			 side_id < mesh->weno_cell_face_offset[cell + 1]; ++side_id)
		{
			const auto& side = mesh->weno_face_sides[side_id];
			auto [it, inserted] = selected_by_logical_face.emplace(side.logical_face, side_id);
			if (!inserted)
			{
				const auto& current = mesh->weno_face_sides[it->second];
				if (side.area > current.area ||
					(side.area == current.area && std::tie(side.connection_id, side.node_offset)
					 < std::tie(current.connection_id, current.node_offset)))
					it->second = side_id;
			}
		}

		std::vector<index_t> selected_sides;
		selected_sides.reserve(selected_by_logical_face.size());
		std::vector<Vector3> face_displacements;
		face_displacements.reserve(selected_by_logical_face.size());
		for (const auto& [logical_face, side_id] : selected_by_logical_face)
		{
			(void)logical_face;
			selected_sides.push_back(side_id);
			face_displacements.push_back(mesh->weno_face_sides[side_id].centroid - mesh->centroids[cell]);
		}

		CellGeometry& cell_geometry = geometry[cell];
		cell_geometry.transform_valid = build_reference_transform(
			face_displacements, condition_limit, cell_geometry.transform);
		if (!cell_geometry.transform_valid)
			continue;

		std::map<index_t, std::vector<index_t>> faces_at_node;
		for (index_t local_face = 0; local_face < static_cast<index_t>(selected_sides.size()); ++local_face)
		{
			const auto& side = mesh->weno_face_sides[selected_sides[local_face]];
			for (index_t node_pos = side.node_offset; node_pos < side.node_offset + side.node_count; ++node_pos)
				faces_at_node[mesh->weno_face_nodes[node_pos]].push_back(local_face);
		}

		std::set<std::array<index_t, WENO_DIM>> face_triplets;
		for (auto& [node, incident_faces] : faces_at_node)
		{
			(void)node;
			std::sort(incident_faces.begin(), incident_faces.end());
			incident_faces.erase(std::unique(incident_faces.begin(), incident_faces.end()), incident_faces.end());
			for (size_t a = 0; a < incident_faces.size(); ++a)
				for (size_t b = a + 1; b < incident_faces.size(); ++b)
					for (size_t c = b + 1; c < incident_faces.size(); ++c)
						face_triplets.insert({incident_faces[a], incident_faces[b], incident_faces[c]});
		}

		// General polyhedral/fault fallback when compact selected subfaces no
		// longer share their original vertices.
		if (face_triplets.empty())
		{
			for (index_t a = 0; a < static_cast<index_t>(selected_sides.size()); ++a)
				for (index_t b = a + 1; b < static_cast<index_t>(selected_sides.size()); ++b)
					for (index_t c = b + 1; c < static_cast<index_t>(selected_sides.size()); ++c)
						face_triplets.insert({a, b, c});
		}

		for (const auto& triplet : face_triplets)
		{
			struct Support
			{
				index_t engine_cell;
				index_t kind;
				index_t face;
				Vector3 point;
			};
			std::array<Support, WENO_DIM> supports;
			bool valid = true;
			for (index_t support = 0; support < WENO_DIM; ++support)
			{
				const index_t selected_id = selected_sides[triplet[support]];
				const auto& side = mesh->weno_face_sides[selected_id];
				if (side.neighbour >= 0 && side.neighbour < mesh->n_cells &&
					discretizer_to_engine[side.neighbour] >= 0)
				{
					supports[support] = {discretizer_to_engine[side.neighbour], WENO_SUPPORT_CELL,
									 selected_id, mesh->centroids[side.neighbour]};
				}
				else if (side.neighbour == -1)
				{
					supports[support] = {engine_cell, WENO_SUPPORT_TARGET_COPY,
									 selected_id, side.centroid};
				}
				else
				{
					valid = false;
					break;
				}
			}
			if (!valid)
				continue;

			std::sort(supports.begin(), supports.end(), [](const Support& left, const Support& right)
			{
				return std::tie(left.engine_cell, left.kind, left.face)
					 < std::tie(right.engine_cell, right.kind, right.face);
			});

			std::array<value_t, 9> interpolation_matrix{};
			Candidate candidate;
			for (index_t support = 0; support < WENO_DIM; ++support)
			{
				const auto reference = transform_point(cell_geometry.transform,
					supports[support].point - mesh->centroids[cell]);
				for (index_t d = 0; d < WENO_DIM; ++d)
					interpolation_matrix[support * WENO_DIM + d] = reference[d];
				candidate.support_cell[support] = supports[support].engine_cell;
				candidate.support_kind[support] = supports[support].kind;
				candidate.face_key[support] = supports[support].face;
			}

			value_t determinant = 0.0;
			if (!inverse3(interpolation_matrix, candidate.inverse, determinant, candidate.condition) ||
				candidate.condition > condition_limit)
				continue;
			candidate.gamma = std::abs(determinant) / 6.0;
			if (candidate.gamma > 0.0 && std::isfinite(candidate.gamma))
				cell_geometry.candidates.push_back(candidate);
		}

		// Remove geometrically identical candidates produced by different face
		// triplets so the shared reference tetrahedron is weighted only once.
		std::sort(cell_geometry.candidates.begin(), cell_geometry.candidates.end(),
			[](const Candidate& left, const Candidate& right)
			{return candidate_geometry_key(left) < candidate_geometry_key(right);});
		cell_geometry.candidates.erase(
			std::unique(cell_geometry.candidates.begin(), cell_geometry.candidates.end(),
				[](const Candidate& left, const Candidate& right)
				{return candidate_geometry_key(left) == candidate_geometry_key(right);}),
			cell_geometry.candidates.end());

		if (static_cast<index_t>(cell_geometry.candidates.size()) > max_candidates)
		{
			std::stable_sort(cell_geometry.candidates.begin(), cell_geometry.candidates.end(),
				[](const Candidate& left, const Candidate& right)
				{
					if (left.condition != right.condition)
						return left.condition < right.condition;
					if (left.gamma != right.gamma)
						return left.gamma > right.gamma;
					return candidate_less(left, right);
				});
			cell_geometry.candidates.resize(max_candidates);
		}
		std::sort(cell_geometry.candidates.begin(), cell_geometry.candidates.end(), candidate_less);

		value_t gamma_sum = 0.0;
		for (const Candidate& candidate : cell_geometry.candidates)
			gamma_sum += candidate.gamma;
		if (!(gamma_sum > 0.0) || !std::isfinite(gamma_sum))
		{
			cell_geometry.candidates.clear();
			continue;
		}
		for (Candidate& candidate : cell_geometry.candidates)
			candidate.gamma /= gamma_sum;
		candidates_by_engine_cell[engine_cell] = cell_geometry.candidates;
	}

	weno.clear();
	weno.cell_candidate_offset.reserve(static_cast<size_t>(n_engine_cells) + 1);
	weno.cell_dependency_offset.reserve(static_cast<size_t>(n_engine_cells) + 1);
	weno.cell_status.assign(n_engine_cells, 0);
	weno.cell_candidate_offset.push_back(0);
	weno.cell_dependency_offset.push_back(0);
	for (index_t engine_cell = 0; engine_cell < n_engine_cells; ++engine_cell)
	{
		const auto& cell_candidates = candidates_by_engine_cell[engine_cell];
		std::set<index_t> dependencies;
		if (!cell_candidates.empty())
		{
			weno.cell_status[engine_cell] = 1;
			dependencies.insert(engine_cell);
		}
		for (const Candidate& candidate : cell_candidates)
		{
			for (index_t support = 0; support < WENO_DIM; ++support)
			{
				weno.candidate_support_cell.push_back(candidate.support_cell[support]);
				weno.candidate_support_kind.push_back(candidate.support_kind[support]);
				dependencies.insert(candidate.support_cell[support]);
			}
			weno.candidate_inverse.insert(weno.candidate_inverse.end(),
				candidate.inverse.begin(), candidate.inverse.end());
			weno.candidate_gamma.push_back(candidate.gamma);
			weno.candidate_condition.push_back(candidate.condition);
		}
		weno.cell_candidate_offset.push_back(static_cast<index_t>(weno.candidate_gamma.size()));
		weno.cell_dependency_cell.insert(weno.cell_dependency_cell.end(), dependencies.begin(), dependencies.end());
		weno.cell_dependency_offset.push_back(static_cast<index_t>(weno.cell_dependency_cell.size()));
	}

	// Gather both target-side query vectors for every canonical geometric
	// connection, then align them with the one-way engine connection order.
	std::map<index_t, std::vector<index_t>> sides_by_connection;
	for (index_t side_id = 0; side_id < static_cast<index_t>(mesh->weno_face_sides.size()); ++side_id)
	{
		const auto& side = mesh->weno_face_sides[side_id];
		if (side.connection_id >= 0 && side.neighbour >= 0)
			sides_by_connection[side.connection_id].push_back(side_id);
	}

	std::map<std::pair<index_t, index_t>, std::vector<GeometricConnection>> connections_by_pair;
	for (const auto& [connection_id, side_ids] : sides_by_connection)
	{
		if (side_ids.size() < 2)
			continue;
		for (size_t first = 0; first < side_ids.size(); ++first)
		{
			const auto& side_a = mesh->weno_face_sides[side_ids[first]];
			for (size_t second = first + 1; second < side_ids.size(); ++second)
			{
				const auto& side_b = mesh->weno_face_sides[side_ids[second]];
				if (side_a.cell != side_b.neighbour || side_b.cell != side_a.neighbour)
					continue;
				const index_t engine_a = discretizer_to_engine[side_a.cell];
				const index_t engine_b = discretizer_to_engine[side_b.cell];
				if (engine_a < 0 || engine_b < 0 || engine_a >= n_engine_cells || engine_b >= n_engine_cells)
					continue;

				GeometricConnection connection;
				connection.id = connection_id;
				connection.engine_cell_a = engine_a;
				connection.engine_cell_b = engine_b;
				if (geometry[side_a.cell].transform_valid)
				{
					connection.reference_a = transform_point(geometry[side_a.cell].transform,
						side_a.centroid - mesh->centroids[side_a.cell]);
					connection.valid_a = weno.cell_status[engine_a] != 0;
				}
				if (geometry[side_b.cell].transform_valid)
				{
					connection.reference_b = transform_point(geometry[side_b.cell].transform,
						side_b.centroid - mesh->centroids[side_b.cell]);
					connection.valid_b = weno.cell_status[engine_b] != 0;
				}
				const auto minmax_cells = std::minmax(engine_a, engine_b);
				connections_by_pair[{minmax_cells.first, minmax_cells.second}].push_back(connection);
				first = side_ids.size();
				break;
			}
		}
	}
	for (auto& [pair, connections] : connections_by_pair)
	{
		(void)pair;
		std::sort(connections.begin(), connections.end(),
			[](const GeometricConnection& left, const GeometricConnection& right)
			{return left.id < right.id;});
	}

	const size_t n_engine_connections = engine_block_m.size();
	weno.one_way_face_reference_m.assign(WENO_DIM * n_engine_connections, 0.0);
	weno.one_way_face_reference_p.assign(WENO_DIM * n_engine_connections, 0.0);
	weno.one_way_face_status_m.assign(n_engine_connections, 0);
	weno.one_way_face_status_p.assign(n_engine_connections, 0);
	std::map<std::pair<index_t, index_t>, size_t> pair_cursor;
	for (size_t connection_id = 0; connection_id < n_engine_connections; ++connection_id)
	{
		const index_t block_m = engine_block_m[connection_id];
		const index_t block_p = engine_block_p[connection_id];
		if (block_m < 0 || block_p < 0 || block_m >= n_engine_cells || block_p >= n_engine_cells)
			continue;
		const auto minmax_cells = std::minmax(block_m, block_p);
		const std::pair<index_t, index_t> pair(minmax_cells.first, minmax_cells.second);
		const auto connections_it = connections_by_pair.find(pair);
		if (connections_it == connections_by_pair.end())
			continue;
		size_t& cursor = pair_cursor[pair];
		if (cursor >= connections_it->second.size())
			continue;
		const GeometricConnection& connection = connections_it->second[cursor++];

		const bool m_is_a = connection.engine_cell_a == block_m;
		const auto& reference_m = m_is_a ? connection.reference_a : connection.reference_b;
		const auto& reference_p = m_is_a ? connection.reference_b : connection.reference_a;
		const bool valid_m = m_is_a ? connection.valid_a : connection.valid_b;
		const bool valid_p = m_is_a ? connection.valid_b : connection.valid_a;
		for (index_t d = 0; d < WENO_DIM; ++d)
		{
			weno.one_way_face_reference_m[WENO_DIM * connection_id + d] = reference_m[d];
			weno.one_way_face_reference_p[WENO_DIM * connection_id + d] = reference_p[d];
		}
		weno.one_way_face_status_m[connection_id] = valid_m;
		weno.one_way_face_status_p[connection_id] = valid_p;
	}
}
