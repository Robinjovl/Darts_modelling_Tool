#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <algorithm>

#include "multilinear_static_cpu_interpolator.hpp"

using namespace std;

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
multilinear_static_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::multilinear_static_cpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                                                                                          const std::vector<int> &axes_points,
                                                                                                          const std::vector<double> &axes_min,
                                                                                                          const std::vector<double> &axes_max)
    : multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>(supporting_point_evaluator, axes_points, axes_min, axes_max)

{
  this->n_points_used = this->n_points_total;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_static_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::init()
{
  // initialize base class first
  interpolator_base::init();

  // evaluate supporting point data unless it was already assigned via Python
  if (point_data.size() == 0)
  {
    cout << "Computing " << this->n_points_total << " supporting points for static storage..." << std::endl;
    point_data.resize(this->n_points_total);

    for (auto i = 0; i < this->n_points_total; i++)
    {
      // let generator fill the vector
      this->get_point_coordinates(i, this->new_point_coords);
      this->supporting_point_evaluator->evaluate(this->new_point_coords, this->new_operator_values);
      // and move the data to the new place
      std::copy_n(std::make_move_iterator(this->new_operator_values.begin()), N_OPS, point_data[i].begin());
    }
  }
  // evaluate hypercube data unless it was already assigned via Python
  if (hypercube_data.size() == 0)
  {
    hypercube_points_index_t points;

    // compute the total number of hypercubes
    uint64_t n_hypercubes_total = 1;
    for (int i = 0; i < N_DIMS; i++)
    {
      n_hypercubes_total *= this->axes_points[i] - 1;
    }
    hypercube_data.resize(n_hypercubes_total);
    // fill each hypercube with corresponding data from point_data
    for (auto i = 0; i < n_hypercubes_total; i++)
    {
      this->get_hypercube_points(i, points);

      for (auto j = 0; j < this->N_VERTS; ++j)
      {
        std::copy_n(point_data[points[j]].begin(), N_OPS, hypercube_data[i].begin() + j * N_OPS);
      }
    }
  }
  return 0;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
const typename multilinear_static_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::hypercube_data_t &multilinear_static_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_hypercube_data(const index_t hypercube_index)
{

  return hypercube_data[hypercube_index];
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_static_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::write_to_file(const std::string filename)
{
  std::ofstream txtFile;

  txtFile.open (filename.c_str ());
  txtFile.precision(std::numeric_limits<value_t>::digits10 + 2);

  if (txtFile.is_open ())
  {
    txtFile << this->get_n_dims() << " " << this->get_n_ops() << endl;
    for (int k = 0; k < N_DIMS; k++)
    {
      txtFile << this->axes_points[k] << " " << this->axes_min[k] << " " << this->axes_max[k] << endl;
    }

    for (index_t k = 0; k < point_data.size(); ++k)
      {
        for (int i = 0; i < point_data[k].size(); ++i)
          txtFile << point_data[k][i] << " ";
        txtFile << endl;
      }

  }
  else
  {
    printf ("Can`t open %s for writing!\n", filename.c_str());
    return -1;
  }
  txtFile.close ();

  return 0;
}
