#include <numeric>
#include <algorithm>
#include <fstream>
#include "linear_cpu_interpolator_base.hpp"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
typedef linalg::Matrix<double> Matrix;

template <typename index_t, int N_DIMS, int N_OPS>
linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::linear_cpu_interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                                                                                   const std::vector<int> &axes_points,
                                                                                   const std::vector<double> &axes_min, 
                                                                                   const std::vector<double> &axes_max, 
                                                                                   bool _use_barycentric_interpolation)
    : interpolator_base(supporting_point_evaluator, axes_points, axes_min, axes_max), 
      use_barycentric_interpolation(_use_barycentric_interpolation)
{

    axes_mult[N_DIMS - 1] = 1;
    for (int dim{N_DIMS - 2}; dim >= 0; dim--)
        axes_mult[dim] = axes_mult[dim + 1] * axes_points[dim + 1];

    // initialize the values with 0
    standard_simplex = {};
    // and then set some to 1
    for (int vertex_i = 0; vertex_i < N_DIMS; vertex_i++)
        for (int dim_i = vertex_i; dim_i < N_DIMS; dim_i++)
            standard_simplex[vertex_i][dim_i] = 1;

    double int_type_max = static_cast<double>(std::numeric_limits<index_t>::max());
    if (n_points_total_fp > int_type_max)
    {
        std::string error = "Error: The total requested amount of points (" + std::to_string(n_points_total_fp) +
                            ") exceeds the limit in index type (" + std::to_string(int_type_max) + ")\n";
        throw std::range_error(error);
    }

    transform_last_axis = 1;

    if (use_barycentric_interpolation)
    {
      find_delaunay_and_barycentric();
      // load_delaunay_triangulation("c:\\work\\packages\\open-darts\\engines\\src\\interpolation\\hypercube_delaunay_7_new.bin");
    }
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::load_delaunay_triangulation(const std::string filename)
{
  /*std::ifstream file(filename, std::ios::binary);

  if (!file)
  {
    printf("Unable to open file\n");
    exit(-1);
  }

  int n_simplices, n_simplex_size, n_dim, n_mappings, n_mapping_size;
  //int n_located_simplices;
  while (file.peek() != EOF)
  {
    file.read(reinterpret_cast<char*>(&n_simplices), sizeof(n_simplices));
    file.read(reinterpret_cast<char*>(&n_simplex_size), sizeof(n_simplex_size));
    file.read(reinterpret_cast<char*>(&n_mappings), sizeof(n_mappings));
    file.read(reinterpret_cast<char*>(&n_mapping_size), sizeof(n_mapping_size));
    n_dim = n_simplex_size - 1;

    Delaunay cur_tri(n_simplices, n_simplex_size, n_mappings, n_mapping_size);

    // read simplices
    for (auto& simplex : cur_tri.simplices)
    {
      file.read(reinterpret_cast<char*>(simplex.data()), n_simplex_size * sizeof(int));
    }

    // read points
    for (auto& pt : cur_tri.points)
    {
      file.read(reinterpret_cast<char*>(pt.data()), (n_simplex_size - 1) * sizeof(int));
    }

    // read barycentric transformation
    for (auto& mat : cur_tri.barycentric_matrices)
    {
      file.read(reinterpret_cast<char*>(mat.data()), mat.size() * sizeof(double));
    }

    // read spatial maps
    //n_located_simplices = 0;
    for (auto& map : cur_tri.spatial_map)
    {
      file.read(reinterpret_cast<char*>(map.data()), map.size() * sizeof(int));
      auto pos = std::find_if(map.rbegin(), map.rend(), [](int value) { return value != -1; }).base();
      map.erase(pos, map.end());
      //n_located_simplices += map.size();
    }
    //printf("Ndim = %d\tAvg .= %2.3f\n", n_dim, (double)n_located_simplices / (double)cur_tri.spatial_map.size());

    delaunay_spatial_map_n_points = std::round(std::pow(cur_tri.spatial_map.size(), 1.0 / (double)n_dim));
    delaunay_spatial_map_step = 1.0 / (double)delaunay_spatial_map_n_points;
    tri_info.emplace(std::make_pair(n_dim, std::move(cur_tri)));
  }

  file.close();*/
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::find_delaunay_and_barycentric()
{
  py::gil_scoped_acquire acquire;

  // generate vertices
  int n_points = 1 << N_DIMS;
  std::vector<double> points;
  points.reserve(n_points * N_DIMS);
  for (int i = 0; i < n_points; ++i)
    for (int j = 0; j < N_DIMS; ++j)
      points.push_back((i >> (N_DIMS - 1 - j)) & 1);

  // cast to 2D numpy array
  std::vector<py::ssize_t> shape = { n_points, N_DIMS };
  std::vector<py::ssize_t> strides = { sizeof(double) * N_DIMS, sizeof(double) };
  py::array_t<double> numpy_points = py::array(py::buffer_info(
    const_cast<double*>(points.data()), /* Pointer to data (non-const qualifier is a workaround) */
    sizeof(double),                    /* Size of one scalar */
    py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
    2,                                  /* Number of dimensions */
    shape,                              /* Shape of the matrix */
    strides                             /* Strides for each dimension */
  ));

  // calculate Delaunay triangulation
  py::object scipy = py::module_::import("scipy.spatial");
  py::object Delaunay = scipy.attr("Delaunay");
  tri_info.tri = Delaunay(numpy_points);
  py::array_t<int> simplices = tri_info.tri.attr("simplices").template cast<py::array_t<int>>();
  auto s_info = simplices.request();

  // calculate barycentric transformations
  int n_simplices = s_info.shape[0];
  int n_simplex_size = s_info.shape[1]; // N_DIMS + 1
  tri_info.barycentric_matrices.resize(n_simplices, Matrix(n_simplex_size, n_simplex_size));
  int* simplices_data = static_cast<int*>(s_info.ptr);
  int* cur_simplex;
  for (int sim_i = 0; sim_i < n_simplices; sim_i++)
  {
    auto& mat = tri_info.barycentric_matrices[sim_i];
    cur_simplex = &simplices_data[sim_i * n_simplex_size];
    for (int pt_i = 0; pt_i < N_DIMS + 1; pt_i++)
    {
      std::copy_n(points.data() + cur_simplex[pt_i] * N_DIMS, N_DIMS, &mat.values[0] + pt_i * n_simplex_size);
      mat(pt_i, N_DIMS) = 1.0;
    }

    mat.transposeInplace();
    // invert the system
    bool res = mat.inv();
    if (!res)
    {
      printf("Inversion failed!\n");
      exit(-1);
    }
  }
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::find_hypercube(const std::vector<double> &points,
                                                                          std::array<index_t, N_DIMS> &hypercube,
                                                                          std::array<double, N_DIMS> &scaled_point,
                                                                          const int point_index)
{

    for (int i = 0; i < N_DIMS; i++)
    {
        double point = points[point_index + i];
        if (transform_last_axis && i == (N_DIMS - 1))
        {
            point = axes_max[i] - (point - axes_min[i]);
        }
        scaled_point[i] = (point - axes_min[i]) * axes_step_inv[i];
        hypercube[i] = (int)scaled_point[i];
        scaled_point[i] -= hypercube[i];
    }
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::find_simplex(const std::array<index_t, N_DIMS> &hypercube,
                                                                        const std::array<double, N_DIMS> &scaled_point,
                                                                        std::array<int, N_DIMS> &tri_order,
                                                                        std::array<std::array<index_t, N_DIMS>, N_DIMS + 1> &simplex)
{
    std::iota(tri_order.begin(), tri_order.end(), 0);
    std::sort(tri_order.begin(), tri_order.end(),
              [&scaled_point](int i1, int i2) { return scaled_point[i1] < scaled_point[i2]; });

    for (int vertex_i = 0; vertex_i <= N_DIMS; vertex_i++)
    {
        for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
            simplex[vertex_i][dim_i] = hypercube[dim_i];
        for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
            simplex[vertex_i][tri_order[dim_i]] += standard_simplex[vertex_i][dim_i];
    }
}

template <typename index_t, int N_DIMS, int N_OPS>
int linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::interpolate(const std::vector<value_t> &point, std::vector<value_t> &values)
{
    std::array<index_t, N_DIMS> hypercube;
    std::array<double, N_DIMS> scaled_point;
    find_hypercube(point, hypercube, scaled_point);
    std::array<double, N_DIMS + 1> weights;
    std::array<std::array<index_t, N_DIMS>, N_DIMS + 1> simplex;
    std::array<double, N_OPS> supp_values;

    if (use_barycentric_interpolation)
    {
      py::gil_scoped_acquire guard;
      
      // size of n-simplex
      constexpr int n_size = N_DIMS + 1;

      // find Delaunay simplex
      auto numpy_point = py::array_t<double>(N_DIMS, scaled_point.data());
      int simplex_id = tri_info.tri.attr("find_simplex")(numpy_point).template cast<int>();

      const auto& mat = tri_info.barycentric_matrices[simplex_id];

      // barycentric coordinates
      for (int i = 0; i < n_size; i++)
      {
        weights[i] = 0.0;
        for (int j = 0; j < N_DIMS; j++)
          weights[i] += mat(i, j) * scaled_point[j];

        weights[i] += mat(i, N_DIMS);

        // check consistency
        if (weights[i] < -EQUALITY_TOLERANCE || weights[i] > 1.0 + EQUALITY_TOLERANCE)
          printf("%d-th barycentric coordinate %f lies outside unit interval", i, weights[i]);

        assert(weights[i] > -EQUALITY_TOLERANCE && weights[i] < 1.0 + EQUALITY_TOLERANCE);
      }

      // estimate points comprising the simplex
      double* vertices = static_cast<double*>(tri_info.tri.attr("points").template cast<py::array_t<double>>().request().ptr);
      int* simplices = &(static_cast<int*>(tri_info.tri.attr("simplices").template cast<py::array_t<int>>().request().ptr))[n_size * simplex_id];
      double* vertex;
      for (int vertex_i = 0; vertex_i <= N_DIMS; vertex_i++)
      {
        vertex = &vertices[simplices[vertex_i] * N_DIMS];
        for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
          simplex[vertex_i][dim_i] = hypercube[dim_i] + static_cast<index_t>(vertex[dim_i]);
      }
    }
    else
    {
      std::array<int, N_DIMS> tri_order;
      find_simplex(hypercube, scaled_point, tri_order, simplex);

      weights[0] = scaled_point[tri_order[0]];
      weights[N_DIMS] = 1 - scaled_point[tri_order[N_DIMS - 1]];

      for (int dim_i = 1; dim_i < N_DIMS; dim_i++)
        weights[dim_i] = scaled_point[tri_order[dim_i]] - scaled_point[tri_order[dim_i - 1]];
    }

    // interpolate values
    values.assign(N_OPS, 0);
    for (int dim_i = 0; dim_i <= N_DIMS; dim_i++)
    {
      get_supporting_point(simplex[dim_i], supp_values);
      for (int op = 0; op < N_OPS; op++)
      {
        values[op] += weights[dim_i] * supp_values[op];
      }
    }

    return 0;
}

template <typename index_t, int N_DIMS, int N_OPS>
int linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::interpolate_with_derivatives(const std::vector<double> &points,
                                                                                       const std::vector<int> &points_idxs,
                                                                                       std::vector<double> &values,
                                                                                       std::vector<double> &derivatives)
{
    if (use_barycentric_interpolation)
      py::gil_scoped_acquire guard;

    for (std::size_t point_i = 0; point_i < points_idxs.size(); point_i++)
    {
        int point_offset = points_idxs[point_i];

        std::array<index_t, N_DIMS> hypercube;
        std::array<double, N_DIMS> scaled_point;
        find_hypercube(points, hypercube, scaled_point, point_offset * N_DIMS);
        std::array<std::array<index_t, N_DIMS>, N_DIMS + 1> simplex;

        if (use_barycentric_interpolation)
        {
          // size of n-simplex
          constexpr int n_size = N_DIMS + 1;
          std::array<double, N_DIMS + 1> weights;

          // Python code necessitating GIL 
          py::gil_scoped_acquire acquire;

          // find Delaunay simplex
          auto numpy_point = py::array_t<double>(N_DIMS, scaled_point.data());
          int simplex_id = tri_info.tri.attr("find_simplex")(numpy_point).template cast<int>();
          const auto& mat = tri_info.barycentric_matrices[simplex_id];

          // barycentric coordinates
          for (int i = 0; i < n_size; i++)
          {
            weights[i] = 0.0;
            for (int j = 0; j < N_DIMS; j++)
              weights[i] += mat(i, j) * scaled_point[j];

            weights[i] += mat(i, N_DIMS);

            // check consistency
            assert(weights[i] >= 0.0 && weights[i] < 1.0 + EQUALITY_TOLERANCE);
          }

          // estimate points comprising the simplex
          double* vertices = static_cast<double*>(tri_info.tri.attr("points").template cast<py::array_t<double>>().request().ptr);
          int* simplices = &(static_cast<int*>(tri_info.tri.attr("simplices").template cast<py::array_t<int>>().request().ptr))[n_size * simplex_id];
          double* vertex;
          for (int vertex_i = 0; vertex_i <= N_DIMS; vertex_i++)
          {
            vertex = &vertices[simplices[vertex_i] * N_DIMS];
            for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
            {
              simplex[vertex_i][dim_i] = hypercube[dim_i] + static_cast<index_t>(vertex[dim_i]);
            }
          }

          // retrieve values at supporting points
          std::array<std::array<double, N_OPS>, N_DIMS + 1> simplex_vertex_values;
          for (int i = 0; i <= N_DIMS; i++)
            get_supporting_point(simplex[i], simplex_vertex_values[i]);

          int idx;
          for (int op_i = 0; op_i < N_OPS; op_i++)
          {
            values[point_offset * N_OPS + op_i] = 0.0;
            // calculate values
            for (int pt_i = 0; pt_i <= N_DIMS; pt_i++)
              values[point_offset * N_OPS + op_i] += weights[pt_i] * simplex_vertex_values[pt_i][op_i];


            for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
            {
              idx = (point_offset * N_OPS + op_i) * N_DIMS + dim_i;
              derivatives[idx] = 0.0;
              // calculate derivatives
              for (int pt_i = 0; pt_i <= N_DIMS; pt_i++)
                derivatives[idx] += mat(pt_i, dim_i) * simplex_vertex_values[pt_i][op_i];

              if (transform_last_axis && dim_i == (N_DIMS - 1))
                derivatives[idx] *= -axes_step_inv[dim_i];
              else
                derivatives[idx] *= axes_step_inv[dim_i];
            }
          }
        }
        else
        {
          std::array<int, N_DIMS> tri_order;
          find_simplex(hypercube, scaled_point, tri_order, simplex);

          std::array<std::array<double, N_OPS>, N_DIMS + 1> simplex_vertex_values;
          for (int i = 0; i <= N_DIMS; i++)
            get_supporting_point(simplex[i], simplex_vertex_values[i]);

          for (int i = 0; i < N_OPS; i++)
          {
            values[point_offset * N_OPS + i] = simplex_vertex_values[0][i];
          }
          double simplex_vertex_values_gap;
          for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
          {
            for (int op_i = 0; op_i < N_OPS; op_i++)
            {
              simplex_vertex_values_gap = simplex_vertex_values[dim_i + 1][op_i] - simplex_vertex_values[dim_i][op_i];
              values[point_offset * N_OPS + op_i] += (1 - scaled_point[tri_order[dim_i]]) * simplex_vertex_values_gap;
              if (transform_last_axis && tri_order[dim_i] == (N_DIMS - 1))
                simplex_vertex_values_gap *= -1;
              derivatives[(point_offset * N_OPS + op_i) * N_DIMS + tri_order[dim_i]] = -simplex_vertex_values_gap * axes_step_inv[tri_order[dim_i]];
            }
          }
        }
    }
    return 0;
}

template <typename index_t, int N_DIMS, int N_OPS>
index_t linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::get_index_from_vertex(const std::array<index_t, N_DIMS> &vertex)
{
    index_t index = 0;
    for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
        index += vertex[dim_i] * this->axes_mult[dim_i];
    return index;
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::get_point_from_vertex(const std::array<index_t, N_DIMS> &vertex,
                                                                                 std::vector<double> &point)
{
    for (int i = 0; i < N_DIMS; i++)
        point[i] = static_cast<double>(vertex[i]) * axes_step[i] + axes_min[i];
    if (transform_last_axis)
        point[N_DIMS - 1] = axes_max[N_DIMS - 1] - (point[N_DIMS - 1] - axes_min[N_DIMS - 1]);
}