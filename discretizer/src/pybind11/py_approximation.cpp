#include <string>
#include "py_global.h"
#include "approximation.h"

namespace py = pybind11;
using dis::VarName;
using dis::Matrix;
using dis::index_t;
using dis::LinearApproximation;

#define MAKE_OPAQUE_VARIADIC(TYPE, ...) \
    PYBIND11_MAKE_OPAQUE(std::vector<TYPE<__VA_ARGS__>>)

MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Uvar);
MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Pvar);
MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Tvar);

MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Uvar, VarName::Pvar);
MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Uvar, VarName::Tvar);
MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Pvar, VarName::Uvar);
MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Pvar, VarName::Tvar);
MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Tvar, VarName::Uvar);
MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Tvar, VarName::Pvar);

MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Uvar, VarName::Pvar, VarName::Tvar);
MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Uvar, VarName::Tvar, VarName::Pvar);
MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Pvar, VarName::Uvar, VarName::Tvar);
MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Pvar, VarName::Tvar, VarName::Uvar);
MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Tvar, VarName::Uvar, VarName::Pvar);
MAKE_OPAQUE_VARIADIC(LinearApproximation, VarName::Tvar, VarName::Pvar, VarName::Uvar);

constexpr std::string_view var_to_string(VarName v)
{
  switch (v)
  {
  case VarName::Uvar: return "u";
  case VarName::Pvar: return "p";
  case VarName::Tvar: return "t";
  default: return "";
  }
};

template <VarName... Names>
struct VarNameHelper;

template <VarName V>
struct VarNameHelper<V> 
{
  static inline const std::string value = std::string(var_to_string(V));
};

template <VarName First, VarName... Rest>
struct VarNameHelper<First, Rest...> 
{
  static inline const std::string value = std::string(var_to_string(First)) + VarNameHelper<Rest...>::value;
};

template <VarName... VarNames>
struct linear_approximation_exposer
{
  static inline std::string class_name = "linear_approximation_" + std::string(VarNameHelper<VarNames...>::value);

  static void expose(py::module& m)
  {
	py::class_<LinearApproximation<VarNames...>>(m, class_name.c_str(), py::module_local()) \
	  .def(py::init<>())
	  .def(py::init<index_t,index_t>())
	  .def_readwrite("a", &LinearApproximation<VarNames...>::a)
	  .def_readwrite("rhs", &LinearApproximation<VarNames...>::rhs)
	  .def_readwrite("stencil", &LinearApproximation<VarNames...>::stencil)
	  .def_property_readonly_static("var_names", [](py::object) {return LinearApproximation<VarNames...>::var_names; })
	  .def(py::pickle(
		[](const LinearApproximation<VarNames...>& ap) { // __getstate__
		  py::tuple t(4);
		  t[0] = ap.a;
		  t[1] = ap.rhs;
		  t[2] = ap.stencil;
		  t[3] = ap.var_names;
		  return t;
		},
		[](py::tuple t) { // __setstate__
		  LinearApproximation<VarNames...> ap;
		  ap.a = t[0].cast<Matrix>();
		  ap.rhs = t[1].cast<Matrix>();
		  ap.stencil = t[2].cast<std::vector<index_t>>();
		  return ap;
		}));
	py::bind_vector<std::vector<LinearApproximation<VarNames...>>>(m, "vector_" + class_name)
	  .def(py::pickle(
		[](const std::vector<LinearApproximation<VarNames...>>& ap) { // __getstate__
		  py::tuple t(ap.size());
		  for (int i = 0; i < ap.size(); i++)
			t[i] = ap[i];

		  return t;
		},
		[](py::tuple t) { // __setstate__
		  std::vector<LinearApproximation<VarNames...>> ap(t.size());

		  for (int i = 0; i < ap.size(); i++)
			ap[i] = t[i].cast<LinearApproximation<VarNames...>>();

		  return ap;
		}));
  }
};

void pybind_approximation(py::module& m)
{
  py::enum_<VarName>(m, "var_name")
	.value("displacements", VarName::Uvar)
	.value("pressure", VarName::Pvar)
	.value("temperature", VarName::Tvar)
	.export_values();

  // all possible permutations/combinations of parameter pack
  // single parameter
  linear_approximation_exposer<VarName::Uvar>::expose(m);
  linear_approximation_exposer<VarName::Pvar>::expose(m);
  linear_approximation_exposer<VarName::Tvar>::expose(m);
  // two parameters
  linear_approximation_exposer<VarName::Uvar, VarName::Pvar>::expose(m);
  linear_approximation_exposer<VarName::Uvar, VarName::Tvar>::expose(m);
  linear_approximation_exposer<VarName::Pvar, VarName::Uvar>::expose(m);
  linear_approximation_exposer<VarName::Pvar, VarName::Tvar>::expose(m);
  linear_approximation_exposer<VarName::Tvar, VarName::Uvar>::expose(m);
  linear_approximation_exposer<VarName::Tvar, VarName::Pvar>::expose(m);
  // three parameters
  linear_approximation_exposer<VarName::Uvar, VarName::Pvar, VarName::Tvar>::expose(m);
  linear_approximation_exposer<VarName::Uvar, VarName::Tvar, VarName::Pvar>::expose(m);
  linear_approximation_exposer<VarName::Pvar, VarName::Uvar, VarName::Tvar>::expose(m);
  linear_approximation_exposer<VarName::Pvar, VarName::Tvar, VarName::Uvar>::expose(m);
  linear_approximation_exposer<VarName::Tvar, VarName::Uvar, VarName::Pvar>::expose(m);
  linear_approximation_exposer<VarName::Tvar, VarName::Pvar, VarName::Uvar>::expose(m);
};