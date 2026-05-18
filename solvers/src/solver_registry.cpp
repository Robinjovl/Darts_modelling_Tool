//*************************************************************************
//    Copyright (c) 2026
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

#include "solver_registry.hpp"

#include <map>
#include <stdexcept>
#include <utility>

#include "linsolv_iface.hpp"
#include "solver_config.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    namespace
    {
      // Process-wide name -> factory map.
      //
      // A function-local static is used so the map is constructed on first use.
      // Solvers self-register from static registrar objects living in other
      // translation units; routing every registration through this accessor
      // avoids the static initialization order fiasco.
      std::map<std::string, opendarts::linear_solvers::solver_factory> &solver_registry_map()
      {
        static std::map<std::string, opendarts::linear_solvers::solver_factory> registry;
        return registry;
      }
    } // anonymous namespace

    bool register_solver(const std::string &name,
        opendarts::linear_solvers::solver_factory factory)
    {
      if (!factory)
        return false;
      // emplace().second is false if the name was already registered.
      return solver_registry_map().emplace(name, std::move(factory)).second;
    }

    std::shared_ptr<opendarts::linear_solvers::linsolv_iface> create_linear_solver(
        const std::string &name,
        const opendarts::linear_solvers::solver_config &config,
        int block_size)
    {
      auto registry_entry = solver_registry_map().find(name);
      if (registry_entry == solver_registry_map().end())
        throw std::runtime_error("openDARTS linear solver registry: solver '" + name +
                                 "' is not registered in this build.");
      return registry_entry->second(config, block_size);
    }

    bool is_solver_registered(const std::string &name)
    {
      return solver_registry_map().count(name) != 0;
    }

    std::vector<std::string> registered_solvers()
    {
      std::vector<std::string> names;
      names.reserve(solver_registry_map().size());
      for (const auto &registry_entry : solver_registry_map())
        names.push_back(registry_entry.first);  // std::map iterates in sorted key order
      return names;
    }
  } // namespace linear_solvers
} // namespace opendarts
