#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_super(py::module &m)
{
	// N_OPT = NC * (2 * NP + 2) + 3 * NP + 4
	// NP = 1: A =  4, B =  7 (th)
	// NP = 2: A =  6, B = 10 (th)
	// NP = 3: A =  8, B = 12 (th)
	// NP = 4: A = 10, B = 16 (th)

	// N_DIMS = 1, 2, ..., N_DIMS_MAX
	const int N_DIMS_MAX = MAX_NC;

	// thermal problem: N_OPS = A * N_DIMS + B
	const int A1 = 4;
	const int B1 = 7;

	// thermal problem: N_OPS = A * N_DIMS + B, two phase
	const int A2 = 6;
	const int B2 = 10;

	// single phase thermal: N_OPS = A * N_DIMS + B
	const int A3 = 8;
	const int B3 = 13;

	// isothermal problem: N_OPS = A * N_DIMS + B, three phases
	const int A4 = 10;
	const int B4 = 16;

	// geothermal problem: N_OPS = A * N_DIMS + B, three phases
	const int A5 = 4;
	const int B5 = 4;

  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A1, B1> e1;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A2, B2> e2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A3, B3> e3;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A4, B4> e4;
	recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A5, B5> e5;

  e1.expose(m);
  e2.expose(m);
  e3.expose(m);
  e4.expose(m);
  e5.expose(m);
}


//==========================================================

void pybind_operator_set_interpolator_super_dynamic(py::module& m)
{
	//std::string long_name = "Operator set interpolator with " + i_typename + " index type and " + f_typename + " value type for " + std::to_string(N_OPS) + " operators in " + std::to_string(N_DIMS) + "-dimensional parameter space";

	py::class_<multilinear_adaptive_cpu_interpolator_dynamic<uint32_t, double>,
		operator_set_gradient_evaluator_iface>(m, "multilinear_adaptive_cpu_interpolator_dynamic_i_d", "") 
		.def(py::init<operator_set_evaluator_iface*, std::vector<index_t>&, std::vector<value_t>&, std::vector<value_t>&, int, int >(), py::keep_alive<1, 2>()) /*.def("benchmark", &interpolator_class::benchmark, "Init by nc and rate operators") \*/
		//.def("evaluate_with_derivatives", &multilinear_adaptive_cpu_interpolator_dynamic<uint32_t, double>::evaluate_with_derivatives,
		//	"Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a)
		//.def("init_timer_node", &multilinear_adaptive_cpu_interpolator_dynamic<uint32_t, double>::init_timer_node,
		//	"Initialize timer", "timer_node"_a)
		.def("init", &multilinear_adaptive_cpu_interpolator_dynamic<uint32_t, double>::init, "Initialize interpolator")
		.def("write_to_file", &multilinear_adaptive_cpu_interpolator_dynamic<uint32_t, double>::write_to_file, "Write interpolator data to file")
		//.def("evaluate", &multilinear_adaptive_cpu_interpolator_dynamic<uint32_t, double>::evaluate,
		//	"Evaluate operators", "state"_a, "values"_a)
		.def_readwrite("point_data", &multilinear_adaptive_cpu_interpolator_dynamic<uint32_t, double>::point_data);
}
#endif //PYBIND11_ENABLED