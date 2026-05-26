//*************************************************************************
//    Copyright (c) 2022
//    Delft University of Technology, the Netherlands
//    Netherlands eScience Center
//
//    This file is part of the open Delft Advanced Research Terra Simulator (opendarts)
//
//    opendarts is free software: you can redistribute it and/or modify
//    it under the terms of the Apache License.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_DUAL_ARRAY_HPP
#define OPENDARTS_LINEAR_SOLVERS_DUAL_ARRAY_HPP
//--------------------------------------------------------------------------

// dual_array<T> -- the host/device storage primitive of the unified matrix
// layout (see SOLVER_REFACTORING_PLAN.md section 12.5).
//
// It owns a contiguous host buffer and, in a CUDA build, a lazily allocated
// device buffer mirroring it. It tracks which side was last written and
// copies on demand. dual_array is the single home of every cudaMalloc /
// cudaMemcpy in the matrix layer; all other matrix code is device-agnostic.
//
// In a non-CUDA build the type degrades to the host buffer alone, with no
// device members and no overhead.
//
// The host/device-pair-with-dirty-flags shape is intentionally that of a
// Kokkos::DualView / CHAI ManagedArray: should a performance-portability
// layer ever be adopted, it is swapped in behind this type without touching
// the matrix public API or the backend adapters.

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef WITH_GPU
#include <cassert>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#endif

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Contiguous array mirrored between host and (CUDA) device memory.

        Move-only: an implicit copy would silently duplicate device memory.
        Use clone() for an explicit deep copy.

        @tparam T trivially copyable element type (e.g. mat_float, index_t).
    */
    template <class T>
    class dual_array
    {
      static_assert(std::is_trivially_copyable_v<T>,
        "dual_array<T> requires a trivially copyable element type");

    public:
      using value_type = T;
      using size_type = std::size_t;

      dual_array() noexcept = default;

      /** Allocates a host buffer of @p n value-initialised elements. */
      explicit dual_array(size_type n) : host_(n) {}

      ~dual_array() { free_device(); }

      // Move-only.
      dual_array(dual_array &&other) noexcept { move_from(other); }
      dual_array &operator=(dual_array &&other) noexcept
      {
        if (this != &other)
        {
          free_device();
          move_from(other);
        }
        return *this;
      }
      dual_array(const dual_array &) = delete;
      dual_array &operator=(const dual_array &) = delete;

      /** Explicit deep copy of the host buffer. The returned array has no
          device buffer allocated; it is created on first device use. */
      [[nodiscard]] dual_array clone() const
      {
        dual_array copy;
        copy.host_ = host_;
        copy.host_modified_ = true;
        return copy;
      }

      // --- size --------------------------------------------------------------
      [[nodiscard]] size_type size() const noexcept { return host_.size(); }
      [[nodiscard]] bool empty() const noexcept { return host_.empty(); }

      /** Resizes the host buffer (existing elements preserved, new ones
          value-initialised). Any device buffer is invalidated and freed: a
          resize changes the layout, so a stale device copy must not survive. */
      void resize(size_type n)
      {
        host_.resize(n);
        free_device();
        host_modified_ = true;
        device_modified_ = false;
      }

      // --- host access -------------------------------------------------------
      // The non-const overload conservatively records a host-side write, so a
      // later sync_to_device() refreshes the device copy. Use the const
      // overload for read-only access to avoid a needless re-sync.
      [[nodiscard]] T *host_data() noexcept
      {
        host_modified_ = true;
        return host_.data();
      }
      [[nodiscard]] const T *host_data() const noexcept { return host_.data(); }

#ifdef WITH_GPU
      // --- device access -----------------------------------------------------
      /** Returns the device buffer, allocating it on first use, and records a
          device-side write. Does NOT copy host data -- call sync_to_device()
          for that. Use the const overload for read-only device access. */
      [[nodiscard]] T *device_data()
      {
        ensure_device_allocated();
        device_modified_ = true;
        return device_;
      }
      [[nodiscard]] const T *device_data() const
      {
        ensure_device_allocated();
        return device_;
      }

      /** Copies host -> device when the host side holds newer data (or the
          device buffer has not been populated since the last resize). */
      void sync_to_device() const
      {
        assert(!(host_modified_ && device_modified_)
          && "dual_array: host and device both modified -- lost update");
        ensure_device_allocated();
        if (host_modified_ || !device_populated_)
        {
          if (!host_.empty())
            cuda_check(cudaMemcpy(device_, host_.data(), bytes(), cudaMemcpyHostToDevice),
              "dual_array::sync_to_device");
          host_modified_ = false;
          device_populated_ = true;
        }
      }

      /** Copies device -> host when the device side holds newer data. */
      void sync_to_host()
      {
        assert(!(host_modified_ && device_modified_)
          && "dual_array: host and device both modified -- lost update");
        if (device_modified_)
        {
          if (!host_.empty())
            cuda_check(cudaMemcpy(host_.data(), device_, bytes(), cudaMemcpyDeviceToHost),
              "dual_array::sync_to_host");
          device_modified_ = false;
        }
      }

      [[nodiscard]] bool device_allocated() const noexcept { return device_ != nullptr; }
#endif // WITH_GPU

    private:
      void move_from(dual_array &other) noexcept
      {
        host_ = std::move(other.host_);
        host_modified_ = other.host_modified_;
        device_modified_ = other.device_modified_;
#ifdef WITH_GPU
        device_ = other.device_;
        device_capacity_ = other.device_capacity_;
        device_populated_ = other.device_populated_;
        other.device_ = nullptr;
        other.device_capacity_ = 0;
        other.device_populated_ = false;
#endif
      }

#ifdef WITH_GPU
      [[nodiscard]] size_type bytes() const noexcept { return host_.size() * sizeof(T); }

      void ensure_device_allocated() const
      {
        if (device_ != nullptr && device_capacity_ >= host_.size())
          return;
        free_device();
        if (host_.empty())
          return;
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&device_), host_.size() * sizeof(T)),
          "dual_array::ensure_device_allocated");
        device_capacity_ = host_.size();
      }

      void free_device() const noexcept
      {
        if (device_ != nullptr)
        {
          cudaFree(device_);
          device_ = nullptr;
        }
        device_capacity_ = 0;
        device_populated_ = false;
      }

      static void cuda_check(cudaError_t err, const char *what)
      {
        if (err != cudaSuccess)
          throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
      }

      mutable T *device_ = nullptr;
      mutable size_type device_capacity_ = 0; // elements allocated on device
      mutable bool device_populated_ = false; // device buffer holds a current copy
#else
      void free_device() const noexcept {}
#endif // WITH_GPU

      std::vector<T> host_;
      mutable bool host_modified_ = false;
      mutable bool device_modified_ = false;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_DUAL_ARRAY_HPP
//--------------------------------------------------------------------------
