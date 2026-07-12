#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "discretizer.h"

namespace
{
using linalg::Vector3;
using linalg::index_t;
using linalg::value_t;

struct Topology
{
  mesh::ElemType type;
  std::string name;
  std::vector<Vector3> nodes;
  std::vector<std::vector<index_t>> faces;
  Vector3 centroid;
  index_t candidates;
};

Vector3 face_centroid(const Topology& topology, const std::vector<index_t>& face)
{
  Vector3 centroid;
  for (const index_t node : face)
    centroid += topology.nodes[node];
  centroid /= static_cast<value_t>(face.size());
  return centroid;
}

Topology tetrahedron()
{
  return {mesh::TETRA, "tetrahedron",
          {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
          {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}},
          {0.25, 0.25, 0.25}, 4};
}

Topology hexahedron()
{
  return {mesh::HEX, "hexahedron",
          {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
           {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}},
          {{0, 1, 2, 3}, {4, 7, 6, 5}, {0, 4, 5, 1},
           {1, 5, 6, 2}, {2, 6, 7, 3}, {3, 7, 4, 0}},
          {0, 0, 0}, 8};
}

Topology wedge()
{
  return {mesh::PRISM, "wedge",
          {{0, 0, -1}, {1, 0, -1}, {0, 1, -1},
           {0, 0, 1}, {1, 0, 1}, {0, 1, 1}},
          {{0, 2, 1}, {3, 4, 5}, {0, 1, 4, 3},
           {1, 2, 5, 4}, {2, 0, 3, 5}},
          {1.0 / 3.0, 1.0 / 3.0, 0}, 6};
}

Topology pyramid()
{
  return {mesh::PYRAMID, "pyramid",
          {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {0, 0, 1}},
          {{0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}},
          {0, 0, 0.25}, 8};
}

void append_side(mesh::Mesh& mesh, index_t cell, index_t neighbour,
                 index_t connection, index_t logical_face,
                 const std::vector<index_t>& face, const Vector3& centroid)
{
  mesh::WenoFaceSide side;
  side.cell = cell;
  side.neighbour = neighbour;
  side.connection_id = connection;
  side.logical_face = logical_face;
  side.node_offset = static_cast<index_t>(mesh.weno_face_nodes.size());
  side.node_count = static_cast<index_t>(face.size());
  side.centroid = centroid;
  side.area = 1.0;
  mesh.weno_face_nodes.insert(mesh.weno_face_nodes.end(), face.begin(), face.end());
  mesh.weno_face_sides.push_back(side);
}

int test_topology(const Topology& topology)
{
  mesh::Mesh mesh;
  const index_t n_faces = static_cast<index_t>(topology.faces.size());
  mesh.n_cells = n_faces + 1;
  mesh.num_of_elements = mesh.n_cells;
  mesh.nodes = topology.nodes;
  mesh.centroids.resize(mesh.n_cells);
  mesh.centroids[0] = topology.centroid;
  mesh.elems.resize(mesh.n_cells);
  mesh.elems[0].type = topology.type;
  mesh.elems[0].loc = mesh::MATRIX;
  for (index_t face = 0; face < n_faces; ++face)
  {
    mesh.elems[face + 1].loc = mesh::FRACTURE;
    const Vector3 centroid = face_centroid(topology, topology.faces[face]);
    mesh.centroids[face + 1] = 2.0 * centroid - topology.centroid;
    append_side(mesh, 0, face + 1, face, face, topology.faces[face], centroid);
  }
  mesh.weno_cell_face_offset.push_back(0);
  mesh.weno_cell_face_offset.push_back(n_faces);
  for (index_t face = 0; face < n_faces; ++face)
  {
    const Vector3 centroid = face_centroid(topology, topology.faces[face]);
    append_side(mesh, face + 1, 0, face, 0, topology.faces[face], centroid);
    mesh.weno_cell_face_offset.push_back(n_faces + face + 1);
  }

  std::vector<index_t> mapping(mesh.n_cells);
  std::vector<index_t> block_m(n_faces, 0);
  std::vector<index_t> block_p(n_faces);
  for (index_t cell = 0; cell < mesh.n_cells; ++cell)
    mapping[cell] = cell;
  for (index_t face = 0; face < n_faces; ++face)
    block_p[face] = face + 1;

  dis::Discretizer discretizer;
  discretizer.set_mesh(&mesh);
  discretizer.prepare_weno_static(mapping, mesh.n_cells, block_m, block_p, 1.e8, 64);
  const auto& weno = discretizer.weno;
  const index_t candidate_count = weno.cell_candidate_offset[1] -
                                  weno.cell_candidate_offset[0];
  if (candidate_count != topology.candidates)
  {
    std::cerr << topology.name << ": expected " << topology.candidates
              << " candidates, got " << candidate_count << '\n';
    return 1;
  }

  const std::array<value_t, 3> gradient = {0.7, -1.1, 0.4};
  std::vector<value_t> scalar(mesh.n_cells);
  for (index_t cell = 0; cell < mesh.n_cells; ++cell)
    scalar[cell] = 2.3 + gradient[0] * mesh.centroids[cell].x +
                   gradient[1] * mesh.centroids[cell].y +
                   gradient[2] * mesh.centroids[cell].z;

  for (index_t face = 0; face < n_faces; ++face)
  {
    if (!weno.one_way_face_status_m[face])
    {
      std::cerr << topology.name << ": missing target-side data for face " << face << '\n';
      return 1;
    }
    const value_t* reference = &weno.one_way_face_reference_m[3 * face];
    value_t alpha_sum = 0.0;
    value_t reconstructed = 0.0;
    std::vector<value_t> candidate_value(candidate_count);
    std::vector<value_t> alpha(candidate_count);
    for (index_t candidate = 0; candidate < candidate_count; ++candidate)
    {
      value_t difference[3];
      for (index_t support = 0; support < 3; ++support)
        difference[support] = scalar[weno.candidate_support_cell[3 * candidate + support]] -
                              scalar[0];
      value_t smoothness = 0.0;
      candidate_value[candidate] = scalar[0];
      for (index_t direction = 0; direction < 3; ++direction)
      {
        value_t slope = 0.0;
        for (index_t support = 0; support < 3; ++support)
          slope += weno.candidate_inverse[9 * candidate + 3 * direction + support] *
                   difference[support];
        smoothness += slope * slope;
        candidate_value[candidate] += reference[direction] * slope;
      }
      alpha[candidate] = weno.candidate_gamma[candidate] /
                         std::pow(1.e-12 + smoothness, 2);
      alpha_sum += alpha[candidate];
    }
    for (index_t candidate = 0; candidate < candidate_count; ++candidate)
      reconstructed += alpha[candidate] / alpha_sum * candidate_value[candidate];

    const Vector3 centroid = face_centroid(topology, topology.faces[face]);
    const value_t exact = 2.3 + gradient[0] * centroid.x + gradient[1] * centroid.y +
                          gradient[2] * centroid.z;
    if (std::abs(reconstructed - exact) > 1.e-11)
    {
      std::cerr << topology.name << ": linear reconstruction error on face " << face
                << " is " << reconstructed - exact << '\n';
      return 1;
    }
  }
  return 0;
}
}

int main()
{
  int errors = 0;
  errors += test_topology(tetrahedron());
  errors += test_topology(hexahedron());
  errors += test_topology(wedge());
  errors += test_topology(pyramid());
  return errors;
}
