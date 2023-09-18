#include "py_mech_discretizer.h"

namespace py = pybind11;
using dis::Matrix;
using dis::Matrix33;
using dis::Discretizer;
using dis::MechDiscretizer;
using dis::MechDiscretizerMode;
using dis::GenericBoundaryCondition;
using dis::THMBoundaryCondition;
using dis::Stiffness;

PYBIND11_MAKE_OPAQUE(std::vector<Stiffness>);
PYBIND11_MAKE_OPAQUE(std::vector<Matrix>);
PYBIND11_MAKE_OPAQUE(std::vector<Matrix33>);

template <MechDiscretizerMode MODE>
struct mech_discretizer_exposer
{
  static const std::string class_name;

  static void expose(py::module& m)
  {
	py::class_<MechDiscretizer<MODE>, Discretizer>(m, class_name.c_str(), py::module_local())
	  .def(py::init<>())
	  .def_readwrite("stfs", &MechDiscretizer<MODE>::stfs)
	  .def_readwrite("biots", &MechDiscretizer<MODE>::biots)
	  .def_readwrite("thermal_expansions", &MechDiscretizer<MODE>::th_exps)
	  .def_readwrite("u_grads", &MechDiscretizer<MODE>::u_grads)
	  .def_readwrite("neumann_boundaries_grad_reconstruction", &MechDiscretizer<MODE>::NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION)
	  .def("reconstruct_displacement_gradients_per_cell", &MechDiscretizer<MODE>::reconstruct_displacement_gradients_per_cell)
	  .def("calc_mpfa_mpsa_transmissibilities", &MechDiscretizer<MODE>::calc_mpfa_mpsa_transmissibilities)
	  ;
  }
};

template<> const std::string 
mech_discretizer_exposer<MechDiscretizerMode::POROELASTIC>::class_name = "poro_mech_discretizer";

template<> const std::string
mech_discretizer_exposer<MechDiscretizerMode::THERMOPOROELASTIC>::class_name = "thermoporo_mech_discretizer";


void pybind_mech_discretizer(py::module& m)
{
  mech_discretizer_exposer<MechDiscretizerMode::POROELASTIC> dis_poro;
  dis_poro.expose(m);

  mech_discretizer_exposer<MechDiscretizerMode::THERMOPOROELASTIC> dis_thermoporo;
  dis_thermoporo.expose(m);

  py::class_<Stiffness, Matrix>(m, "Stiffness") \
	.def(py::init<>())
	.def(py::init<value_t, value_t>())
	.def(py::init<std::valarray<value_t> &>())
	.def_readwrite("values", &Stiffness::values)
	.def(py::pickle(
	  [](const Stiffness& p) { // __getstate__
		py::tuple t(p.values.size());
		for (int i = 0; i < p.values.size(); i++)
		  t[i] = p.values[i];

		return t;
	  },
	  [](py::tuple t) { // __setstate__
		Stiffness p;

		for (int i = 0; i < t.size(); i++)
		  p.values[i] = t[i].cast<value_t>();

		return p;
	  }));
  py::bind_vector<std::vector<Stiffness>>(m, "stf_vector")
	.def(py::pickle(
	  [](const std::vector<Stiffness>& p) { // __getstate__
		py::tuple t(p.size());
		for (int i = 0; i < p.size(); i++)
		  t[i] = p[i];

		return t;
	  },
	  [](py::tuple t) { // __setstate__
		std::vector<Stiffness> p(t.size());

		for (int i = 0; i < p.size(); i++)
		  p[i] = t[i].cast<Stiffness>();

		return p;
	  }));

  // expose the rest

	py::class_<GenericBoundaryCondition>(m, "GenericBoundaryCondition", py::module_local())
		.def(py::init<>())

		.def_readwrite("a", &GenericBoundaryCondition::a)
		.def_readwrite("b", &GenericBoundaryCondition::b)
		;

  py::class_<THMBoundaryCondition>(m, "THMBoundaryCondition", py::module_local())
	.def(py::init<>())

	.def_readwrite("mech_normal", &THMBoundaryCondition::mech_normal)
	.def_readwrite("mech_tangen", &THMBoundaryCondition::mech_tangen)
  .def_readwrite("flow", &THMBoundaryCondition::flow)
	.def_readwrite("thermal", &THMBoundaryCondition::thermal)
	;

};