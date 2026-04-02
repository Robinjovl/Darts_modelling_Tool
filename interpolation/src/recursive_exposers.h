#ifndef RECURSIVE_EXPOSERS_H
#define RECURSIVE_EXPOSERS_H

// Recursive template helpers for exposing interpolators parametrized by <N_DIMS, N_OPS>.
// Used by pybind11 modules to instantiate interpolator bindings for all required combinations.

// exposer helper class for <N_DIMS, N_OPS> template: N_OPS = N_DIMS * N_OPS_A + N_OPS_B

template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t, uint8_t N_DIMS, uint8_t N_OPS_A, uint8_t N_OPS_B>
struct recursive_exposer_ndims_nops
{
  static void expose(pymodule_t &m)
  {
    exposer_t<N_DIMS, N_DIMS * N_OPS_A + N_OPS_B> e;

    e.expose(m);

    recursive_exposer_ndims_nops<exposer_t, pymodule_t, N_DIMS - 1, N_OPS_A, N_OPS_B>::expose(m);
  }
};

// partial specialization to stop recursion at N_DIMS == 1
template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t, uint8_t N_OPS_A, uint8_t N_OPS_B>
struct recursive_exposer_ndims_nops<exposer_t, pymodule_t, 1, N_OPS_A, N_OPS_B>
{
  static void expose(pymodule_t &m)
  {
    exposer_t<1, 1 * N_OPS_A + N_OPS_B> e;

    e.expose(m);
  }
};

// double-recursive exposer for all (N_DIMS, N_OPS) combinations up to given maxima

template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t, uint8_t N_DIMS, uint8_t N_OPS>
struct recursive_exposer_ndims_nops2
{
    static void expose(pymodule_t& m)
    {
        exposer_t<N_DIMS, N_OPS> e;

        e.expose(m);

        recursive_exposer_ndims_nops2<exposer_t, pymodule_t, N_DIMS - 1, N_OPS>::expose(m);
        recursive_exposer_ndims_nops2<exposer_t, pymodule_t, N_DIMS, N_OPS - 1>::expose(m);
    }
};

// partial specializations to stop recursion
template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t, uint8_t N_OPS>
struct recursive_exposer_ndims_nops2<exposer_t, pymodule_t, 1, N_OPS>
{
    static void expose(pymodule_t& m)
    {
        exposer_t<1, N_OPS> e;

        e.expose(m);
    }
};

template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t, uint8_t N_DIMS>
struct recursive_exposer_ndims_nops2<exposer_t, pymodule_t, N_DIMS, 1>
{
    static void expose(pymodule_t& m)
    {
        exposer_t<N_DIMS, 1> e;

        e.expose(m);
    }
};

template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t>
struct recursive_exposer_ndims_nops2<exposer_t, pymodule_t, 1, 1>
{
    static void expose(pymodule_t& m)
    {
        exposer_t<1, 1> e;

        e.expose(m);
    }
};

// single-dimension exposer: fixed N_DIMS, recurse N_OPS from N_OPS down to 1
template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t, uint8_t N_DIMS, uint8_t N_OPS>
struct recursive_exposer_nops
{
    static void expose(pymodule_t& m)
    {
        exposer_t<N_DIMS, N_OPS> e;
        e.expose(m);
        recursive_exposer_nops<exposer_t, pymodule_t, N_DIMS, N_OPS - 1>::expose(m);
    }
};

template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t, uint8_t N_DIMS>
struct recursive_exposer_nops<exposer_t, pymodule_t, N_DIMS, 1>
{
    static void expose(pymodule_t& m)
    {
        exposer_t<N_DIMS, 1> e;
        e.expose(m);
    }
};

#endif /* RECURSIVE_EXPOSERS_H */
