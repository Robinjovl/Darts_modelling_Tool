#ifndef APPROXIMATION_H_
#define APPROXIMATION_H_

#include <type_traits>
#include <array>
#include "mesh/mesh.h"

namespace dis
{
  using mesh::index_t;
  using mesh::value_t;
  using mesh::Matrix;

  // variable's names we perform approxiamtion over: 
  // 'Uvar' - vector of displacements, 
  // 'Pvar' - pressure 
  // 'Tvar' - temperature
  enum VarName { Uvar, Pvar, Tvar };
  
  // class template that represents a linear approximation
  // template parameters are variables used in approximation
  template <VarName... VarNames>
  class LinearApproximation
  {
  public:
    static inline const std::array<VarName, sizeof...(VarNames)> var_names = { VarNames... };
    static inline const index_t n_block = sizeof...(VarNames);

    LinearApproximation() {};
    LinearApproximation(index_t apprx_size, index_t stencil_size)
    {
      a = Matrix(apprx_size, stencil_size * var_names.size());
      rhs = Matrix(apprx_size, 1);
      stencil.reserve(stencil_size);
    };
    LinearApproximation(const LinearApproximation<VarNames...>& ap)
    {
      (*this) = ap;
    };
    LinearApproximation<VarNames...>& operator=(const LinearApproximation<VarNames...>& ap)
    {
      a = ap.a;
      rhs = ap.rhs;
      stencil = ap.stencil;
      return *this;
    };
    ~LinearApproximation() {};

    Matrix a, rhs;
    std::vector<index_t> stencil;

    // sort stencil and swapping columns in 'a' accordingly
    void sort()
    {
      std::vector<index_t> sorted_ids(stencil.size());
      std::iota(sorted_ids.begin(), sorted_ids.end(), 0);
      std::sort(sorted_ids.begin(), sorted_ids.end(),
        [&stencil = this->stencil](index_t i1, index_t i2) {return stencil[i1] < stencil[i2]; });

      std::vector<index_t> old_stencil(stencil);
      Matrix old_a(a);
      for (index_t i = 0; i < stencil.size(); i++)
      {
        stencil[i] = old_stencil[sorted_ids[i]];

        for (index_t row = 0; row < a.M; row++)
          std::copy_n(std::begin(old_a.values) + row * a.N + sorted_ids[i] * n_block, n_block,
                      std::begin(a.values) + row * a.N + i * n_block);
      }
    };
  };

  // utilities for compile-time estimate of the type of approximation merged from two elements
  template <VarName...>
  struct TypeList {};

  template <VarName V, typename TypeList>
  struct Contains;

  template <VarName V, VarName... Others>
  struct Contains<V, TypeList<Others...>> : std::disjunction<std::bool_constant<(V == Others)>...> {};

  template <VarName V, typename List, bool isPresent>
  struct AddIfNotPresent;

  template <VarName V, VarName... Items>
  struct AddIfNotPresent<V, TypeList<Items...>, true> 
  {
    using type = TypeList<Items...>;
  };

  template <VarName V, VarName... Items>
  struct AddIfNotPresent<V, TypeList<Items...>, false> 
  {
    using type = TypeList<Items..., V>;
  };

  template <typename List1, typename List2>
  struct MergeTypeLists;

  template <VarName... Items1>
  struct MergeTypeLists<TypeList<Items1...>, TypeList<>> 
  {
    using type = TypeList<Items1...>;
  };

  template <VarName... Items1, VarName First, VarName... Rest>
  struct MergeTypeLists<TypeList<Items1...>, TypeList<First, Rest...>> 
  {
    using NewList = typename AddIfNotPresent<First, TypeList<Items1...>, Contains<First, TypeList<Items1...>>::value>::type;
    using type = typename MergeTypeLists<NewList, TypeList<Rest...>>::type;
  };

  template<typename T>
  struct ExtractTypes;

  template<VarName... Vs>
  struct ExtractTypes<TypeList<Vs...>> 
  {
    using type = LinearApproximation<Vs...>;
  };

  template <VarName V, typename List>
  struct IndexOf;

  template <VarName V, VarName... Others>
  struct IndexOf<V, TypeList<Others...>> {
    static constexpr int value = [] {
      constexpr std::array<VarName, sizeof...(Others)> arr = { Others... };
      for (int i = 0; i < sizeof...(Others); ++i) {
        if (arr[i] == V) return i;
      }
      return -1;
    }();
  };

  template <VarName V, VarName... Others>
  constexpr int CalculateMapping() {
    constexpr VarName arr[] = { Others... };
    for (std::size_t i = 0; i < sizeof...(Others); ++i) {
      if (arr[i] == V) return static_cast<int>(i);
    }
    return -1;
  }

  // merge stencils
  static void merge_stencils(const std::vector<index_t>& st1, const std::vector<index_t>& st2, std::vector<index_t>& st)
  {
    index_t i = 0, j = 0;

    while (i != st1.size() && j != st2.size())
    {
      if (st1[i] == st2[j]) 
      {
        st.push_back(st1[i]);
        i++; j++;
      }
      else if (st1[i] < st2[j]) 
      {
        st.push_back(st1[i]);
        i++;
      }
      else 
      {
        st.push_back(st2[j]);
        j++;
      }
    }

    while (i < st1.size()) 
    {
      st.push_back(st1[i]);
      i++;
    }

    while (j < st2.size()) 
    {
      st.push_back(st2[j]);
      j++;
    }
  }

  // merge function
  template <VarName... VarNames1, VarName... VarNames2>
  auto merge(const LinearApproximation<VarNames1...>& ap1, const LinearApproximation<VarNames2...>& ap2, value_t mult2)
  {
    // check if approximations are of the same size
    assert(ap1.a.M == ap2.a.M);
    
    // calculate output type
    using MergedType = typename MergeTypeLists<TypeList<VarNames1...>, TypeList<VarNames2...>>::type;
    using ResultType = typename ExtractTypes<MergedType>::type;
    ResultType res;
    merge_stencils(ap1.stencil, ap2.stencil, res.stencil); // merge stencils first
    res.a = Matrix(ap1.a.M, res.n_block * res.stencil.size());
    res.rhs = Matrix(ap1.a.M, 1);

    // compile-time evaluation of output columns where we need to add the second matrix
    constexpr std::array<int, sizeof...(VarNames2)> mapping2 = { []() -> int
      {
        if (IndexOf<VarNames2, TypeList<VarNames1...>>::value != -1)
        { return IndexOf<VarNames2, TypeList<VarNames1...>>::value; }
        else { return CalculateMapping<VarNames2, VarNames1..., VarNames2>(); }
      }()...
    };

    constexpr index_t nb1 = ap1.n_block;
    constexpr index_t nb2 = ap2.n_block;
    constexpr index_t nb = res.n_block;

    index_t i = 0, j = 0, k = 0;
    while (i != ap1.stencil.size() && j != ap2.stencil.size())
    {
      if (ap1.stencil[i] == ap2.stencil[j])
      {
        for (index_t it = 0; it < ap1.a.M; ++it)
        {
          // 1st contribution
          for (index_t jt = 0; jt < nb1; ++jt)
          {
            res.a(it, k * nb + jt) += ap1.a(it, i * nb1 + jt);
          }
          // 2nd contribution
          for (index_t jt = 0; jt < nb2; ++jt)
          {
            res.a(it, k * nb + mapping2[jt]) += mult2 * ap2.a(it, j * nb2 + jt);
          }
        }
        i++; j++; k++;
      }
      else if (ap1.stencil[i] < ap2.stencil[j]) {
        for (index_t it = 0; it < ap1.a.M; ++it)
        {
          // 1st contribution
          for (index_t jt = 0; jt < nb1; ++jt)
          {
            res.a(it, k * nb + jt) += ap1.a(it, i * nb1 + jt);
          }
        }
        i++; k++;
      }
      else {
        for (index_t it = 0; it < ap1.a.M; ++it)
        {
          // 2nd contribution
          for (index_t jt = 0; jt < nb2; ++jt)
          {
            res.a(it, k * nb + mapping2[jt]) += mult2 * ap2.a(it, j * nb2 + jt);
          }
        }
        k++; j++;
      }
    }
    // remaining stencil from 1st
    while (i < ap1.stencil.size())
    {
      for (index_t it = 0; it < ap1.a.M; ++it)
      {
        // 1st contribution
        for (index_t jt = 0; jt < nb1; ++jt)
        {
          res.a(it, k * nb + jt) += ap1.a(it, i * nb1 + jt);
        }
      }
      k++; i++;
    }
    // remaining stencil from 2nd
    while (j < ap2.stencil.size())
    {
      for (index_t it = 0; it < ap1.a.M; ++it)
      {
        // 2nd contribution
        for (index_t jt = 0; jt < nb2; ++jt)
        {
          res.a(it, k * nb + mapping2[jt]) += mult2 * ap2.a(it, j * nb2 + jt);
        }
      }
      k++; j++;
    }

    res.rhs = ap1.rhs + mult2 * ap2.rhs;

    return res;
  }

  template <VarName... VarNames1, VarName... VarNames2>
  auto operator+(const LinearApproximation<VarNames1...>& ap1, const LinearApproximation<VarNames2...>& ap2)
  {
    return merge(ap1, ap2, 1.0);
  }
  template <VarName... VarNames1, VarName... VarNames2>
  auto operator-(const LinearApproximation<VarNames1...>& ap1, const LinearApproximation<VarNames2...>& ap2)
  {
    return merge(ap1, ap2, -1.0);
  }
  template <VarName... VarNames>
  LinearApproximation<VarNames...> operator*(const LinearApproximation<VarNames...>& ap, const value_t val)
  {
    LinearApproximation<VarNames...> result(ap);
    result.a = val * result.a;
    result.rhs = val * result.rhs;
    return result;
  }
  template <VarName... VarNames>
  LinearApproximation<VarNames...> operator*(const value_t val, const LinearApproximation<VarNames...>& ap)
  {
    return ap * val;
  }
  template <VarName... VarNames>
  LinearApproximation<VarNames...> operator/(const LinearApproximation<VarNames...>& ap, const value_t val)
  {
    return ap * (1 / val);
  }

}

#endif /* APPROXIMATION_H_ */
