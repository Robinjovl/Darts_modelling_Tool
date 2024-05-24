---
title: 'openDARTS: open Delft Advanced Research Terra Simulator'
tags:
  - energy transition
  - geothermal
  - CO2 sequestration
  - multi-physics
  - geo-engineering
authors:
  - name: Denis Voskov
    orcid: 0000-0002-5399-1755
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Xiaocong Lyu
    affiliation: 1
  - name: Stephan de Hoop
    affiliation: 1
  - name: Mark Khait
    affiliation: 1
  - name: Aleks Novikov
    affiliation: 1
  - name: Michiel Wapperom
    affiliation: 1
    orcid: 0000-0003-3432-4233
  - name: Ilshat Saifullin
    orcid: 0009-0001-0089-8629 
    affiliation: 1
  - name: Xiaoming Tian
    affiliation: "1, 5"
    orcid: 0000-0003-0642-6064
  - name: Gabriel Serrão Seabra
    orcid: 0009-0002-0558-8117
    affiliation: "1, 6"
  - name: Luisa Orozco
    orcid: 0000-0002-9153-650X
    affiliation: 3
    corresponding: true # (This is how to denote the corresponding author)
  - name: Artur Palha
    orcid: 0000-0002-3217-0747
    affiliation: 4
affiliations:
 - name: Department of Geoscience and Engineering, TU Delft, Delft, Netherlands
   index: 1
 - name: Energy Science and Engineering Department, Stanford, CA, USA
   index: 2
 - name: Netherlands eScience Center, Amsterdam, The Netherlands
   index: 3
 - name: Delft Institute of Applied Mathematics, Delft University of Technology
   index: 4
 - name: Guangzhou Institute of Energy Conversion, Chinese Academy of Sciences, Guangzhou, China
   index: 5
 - name: Petrobras, Petr\'oleo Brasileiro S.A., Rio de Janeiro, Brazil
   index: 6
date: 1 April 2024
bibliography: paper.bib
---
# Summary

Open Delft Advanced Research Terra Simulator [@openDARTS_2023] is a simulation framework for forward and inverse modelling and uncertainty quantification of multi-physics processes in geo-engineering applications such as geothermal, CO2 sequestration, water pumping, and hydrogen storage. To efficiently achieve high levels of accuracy on complex geometries, it utilizes advanced numerical methods such as fully implicit thermo-hydro-mechanical-chemical formulation, a highly flexible finite-volume spatial approximation, operator-based linearization for nonlinear terms, and efficient physics-based preconditioners. openDARTS goals are computational efficiency, expandability, and easiness of use. For this reason, openDARTS is based on a hybrid design with an efficient core C++ implementation wrapped around a highly customizable and easy to use Python code.

# Statement of need

openDARTS is designed to use Python as its user interface, which makes it widely used in education and research for both introductory and advanced programming. It is a reservoir simulator with advanced capabilities that are not reliant on proprietary nor licensed software, reducing significantly the entry barrier for researchers and students interested in energy transition applications for the subsurface. The modules discretizer and [darts-flash](https://gitlab.com/open-darts/darts-flash) allow efficient processing of Corner Point Geometry meshes and advanced multiphase equilibrium evaluation for complex fluids respectively.

The openDARTS framework is fully validated and benchmarked for geothermal showing similar accuracy as state of the art simulators TOUGH2 and AD-GPRS while providing a noticeable reduction on CPU time mainly due to the OBL approach [@Wang2020]; CO2 sequestration compared with experiments and other simulators [@Wapperom2023; @fluidflower2023]; hydrocarbon production [@Lyu2021] and induced seismicity applications [@Novikov2022]. The framework's design and parallel implementations provide an exceptional level of flexibility, modularity and performance [@Khait2021]. This versatility is evidenced by its capability to cater a wide range of applications.

Furthermore, advanced inverse capabilities based on adjoint gradients allow openDARTS to effectively address data assimilation [@Tian2024] and uncertainty quantification [@Wang2023] for energy transition applications.

# Key features

## Unified thermal-compositional PDE formulation

openDARTS has a generic PDE formulation for thermal compositional flow in porous media [@Khait2018]. This makes possible to adjust terms in PDE to account for various multi-physical phenomena such as: darcy flow, gravity, multi-component & multiphase flows, thermal flows, chemical and kinetic reactions, etc.

## Geomechanics

openDARTS has a linear poro-elastic (small deformations) formulation that via multi-point approximation used for both fluid and stress fluxes allows for calculating displacements and stressed in a single collocated grid for all physics phenomena on complex meshes [@Novikov2022]. The framework is suitable for solving multi-scale hydro-mechanical, discrete fracture networks [@Hoop2022], friction contact mechanics (slip-fault) problems.

## Discretization

openDARTS employs the finite volume method for spatial discretization and the fully implicit backward Euler method for time discretization. This approach supports arbitrary star-shaped polyhedral cells, offering high flexibility. Additionally, openDARTS implements both two-point and multi-point flux approximations.

Different grid types supported by openDARTS are *a) structured grid* for teaching and basic modelling, *b) radial grid* for near-well and core scale laboratory experiments, *c) corner-point geometry* for industry related applications, *d) unstructured grid* for modelling of flow with complex geometries and discrete fracture networks.

## Operator-Based Linearization

One of the most computationally complex and expensive parts is a calculation of partial derivatives to construct the Jacobian. openDARTS exploits Operator-Based Linearization (OBL) [@Voskov2017; @Khait2017], where the terms in the PDEs are separated into space-dependent terms and thermodynamic state-dependent operators. The later can be parameterized with respect to the nonlinear unknowns using multidimensional tables at different resolutions. The values and derivatives required for the assembly of the linear system can be approximated through multi-linear interpolation in the parameter space using calculated values at the nodes.

Using adaptive parametrization [@Khait2018], derivative computation is performed at nodes of the structured grid in the primary variables space around the required point. Re-using computed values at nodal points can significantly reduce the Jacobian construction stage, especially in the case of ensemble-based simulations.

## Inverse modeling

Inverse modelling methods necessitate a substantial number of simulations to accurately calibrate model parameters against observed data. Such algorithms are highly computationally intensive, particularly when employing gradient-based methods. The implementation of the adjoint method in openDARTS remarkably enhances its efficiency in computing the required gradients for inverse modelling or history matching processes [@Tian2023]. Moreover, the flexibility of openDARTS's Python interface significantly simplifies the coupling process with various data assimilation algorithms. The inverse modelling module of openDARTS accommodates various types of observation data such as: well rates, well temperatures, BHP, time-lapse temperature distributions, and any custom outputs definable in the form of operators within openDARTS.

## Software implementation

The most computationally expensive part of openDARTS is written in C++ with openMP parallelization.
openDARTS can be installed as a Python module and it has a Python-based interface, which makes it suitable for teaching and for users unfamiliar with C++ language. There are several benefits of that compared to a fully C++ code.

- Easy installation via pip and PyPI.
- No need to install compilers.
- Flexible implementation of simulation framework, physical modelling and grids.
- Easy data visualization, including internal arrays and vtk.
- Use popular Python modules within openDARTS and the user's model for data processing and input/output.
- Coupling with other Python-based numerical modelling software.

# Acknowledgements

The authors would like to acknowledge the Netherlands eScience Center for the funding provided under grant number NLESC.OEC.2021.026.

# References
