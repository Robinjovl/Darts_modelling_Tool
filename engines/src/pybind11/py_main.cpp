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

#include "py_globals.h"
namespace py = pybind11;

void pybind_pm_discretizer(py::module &);
void pybind_mesh_conn(py::module &);
void pybind_globals(py::module &);
void pybind_engine_base(py::module &);
void pybind_engine_nc_gpu(py::module &);
void pybind_engine_nc_nl_cpu(py::module &);
void pybind_engine_elasticity_cpu(py::module &);
void pybind_engine_pm_cpu(py::module &);
void pybind_mech_operators(py::module &);
void pybind_contact(py::module &);
void pybind_engine_nce_gpu(py::module &);
void pybind_engine_nce_g_cpu(py::module &);
void pybind_engine_nc_cg_cpu(py::module &);
void pybind_engine_nc_cg_gpu(py::module &);
void pybind_engine_nc_nonnewton_cpu(py::module &);
void pybind_engine_nc_cg_dif_gpu(py::module &);
void pybind_engine_super_cpu(py::module &);
#ifndef WITH_GPU
void pybind_engine_super_mp_cpu(py::module &);
#endif //WITH_GPU
void pybind_engine_super_elastic_cpu(py::module &);
void pybind_engine_super_gpu(py::module &);
void pybind_well_controls(py::module &);
void pybind_ms_well(py::module &);
void pybind_evaluator_iface(py::module &);
void pybind_operator_set_from_files(py::module &);

void pybind_operator_set_interpolator_rates(py::module &);
void pybind_operator_set_interpolator_super(py::module &);
void pybind_operator_set_interpolator_pz_cap_gra(py::module &);
void pybind_operator_set_interpolator_super_elastic(py::module &);

class ms_well;
class operator_set_gradient_evaluator_iface;

PYBIND11_MODULE(engines, m)
{
  m.doc() = "Delft Advanced Research Terra Simulator";
  //auto m1 = m.def_submodule("engines", "Collection of DARTS simulators based on OBL approach");
  py::bind_vector<std::vector<index_t>>(m, "index_vector", py::module_local(true), py::buffer_protocol())
      .def(py::pickle(
          [](const std::vector<index_t>& p) { // __getstate__
              py::tuple t(p.size());
              for (int i = 0; i < p.size(); i++)
                  t[i] = p[i];

              return t;
          },
          [](py::tuple t) { // __setstate__
              std::vector<index_t> p(t.size());

              for (int i = 0; i < p.size(); i++)
                  p[i] = t[i].cast<index_t>();

              //p.setExtra(t[1].cast<int>());

              return p;
          })) \
      .def("resize",
          (void (std::vector<index_t>::*) (size_t count)) & std::vector<index_t>::resize,
          "changes the number of elements stored");
  py::bind_vector<std::vector<value_t>>(m, "value_vector", py::module_local(true), py::buffer_protocol())
      .def(py::pickle(
          [](const std::vector<value_t> &p) { // __getstate__
            py::tuple t(p.size());
            for (int i = 0; i < p.size(); i++)
              t[i] = p[i];

            return t;
          },
          [](py::tuple t) { // __setstate__
            std::vector<value_t> p(t.size());

            for (int i = 0; i < p.size(); i++)
              p[i] = t[i].cast<value_t>();

            //p.setExtra(t[1].cast<int>());

            return p;
          })) \
      .def("resize",
          (void (std::vector<value_t>::*) (size_t count)) &std::vector<value_t>::resize,
          "changes the number of elements stored");
  py::bind_vector<std::vector<ms_well *>>(m, "ms_well_vector");
  py::bind_vector<std::vector<operator_set_gradient_evaluator_iface *>>(m, "op_vector");
  py::bind_map<std::map<std::string, timer_node>>(m, "timer_map");

  pybind_pm_discretizer(m);
  pybind_mesh_conn(m);
  pybind_globals(m);
  pybind_engine_base(m);
  pybind_engine_nc_nl_cpu(m);
  pybind_engine_elasticity_cpu(m);
  pybind_engine_pm_cpu(m);
  pybind_mech_operators(m);
  pybind_contact(m);
  pybind_engine_nce_g_cpu(m);
  pybind_engine_nc_cg_cpu(m);
  pybind_engine_nc_nonnewton_cpu(m);
  pybind_engine_super_cpu(m);
#ifndef WITH_GPU
  pybind_engine_super_mp_cpu(m);
#endif //WITH_GPU
  pybind_engine_super_elastic_cpu(m);

  pybind_well_controls(m);
  pybind_ms_well(m);
  pybind_evaluator_iface(m);

  pybind_operator_set_interpolator_rates(m);
  pybind_operator_set_interpolator_super(m);
  pybind_operator_set_interpolator_pz_cap_gra(m);

#ifdef WITH_GPU
  pybind_engine_nc_cg_gpu(m);
  pybind_engine_super_gpu(m);
#endif
}
