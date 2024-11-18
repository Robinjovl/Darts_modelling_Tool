#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

PYBIND11_MAKE_OPAQUE(std::vector<std::vector<value_t>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::vector<value_t>>>);

namespace py = pybind11;
#include "mech/mech_operators.hpp"
using namespace pm;

void pybind_mech_operators(py::module &m)
{
	py::class_<mech_operators>(m, "mech_operators") \
		.def(py::init<>()) \
		.def("init", (void (mech_operators::*)(conn_mesh *, pm_discretizer*, uint8_t, uint8_t, uint8_t,
			uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t)) &mech_operators::init) \
		.def("init", (void (mech_operators::*)(conn_mesh *, pm_discretizer*, uint8_t, uint8_t, uint8_t,
			uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t)) &mech_operators::init) \
		.def("prepare", &mech_operators::prepare) \
		.def("eval_stresses", (void(mech_operators::*)(std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, const std::vector<value_t>&)) &mech_operators::eval_stresses) \
		.def("eval_porosities", (void(mech_operators::*)(std::vector<value_t>&, std::vector<value_t>&)) &mech_operators::eval_porosities) \
		.def("eval_unknowns_on_faces", (void(mech_operators::*)(std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&)) &mech_operators::eval_unknowns_on_faces) \
		.def_readwrite("p_faces", &mech_operators::pressures) \
		.def_readwrite("stresses", &mech_operators::stresses) \
		.def_readwrite("total_stresses", &mech_operators::total_stresses) \
		.def_readwrite("eps_vol", &mech_operators::eps_vol) \
		.def_readwrite("porosities", &mech_operators::porosities) \
		.def_readwrite("velocities", &mech_operators::velocity) \
		.def_readwrite("face_unknowns", &mech_operators::face_unknowns);
	py::bind_vector<std::vector<std::vector<value_t>>>(m, "vector_value_vector");
	py::bind_vector<std::vector<std::vector<std::vector<value_t>>>>(m, "vector_vector_value_vector");
};

#endif /* PYBIND11_ENABLED */

