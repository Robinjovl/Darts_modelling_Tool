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
  - name: Ilshat Saifullin
    orcid: 0009-0001-0089-8629 
    affiliation: 1
  - name: Aleks Novikov
    affiliation: 1
  - name: Michiel Wapperom
    affiliation: 1
    orcid: 0000-0003-3432-4233
  - name: Luisa Orozco
    orcid: 0000-0002-9153-650X
    affiliation: 3
    corresponding: true # (This is how to denote the corresponding author)
  - name: Gabriel Serrão Seabra
    orcid: 0009-0002-0558-8117
    affiliation: "1, 4"
  - name: Yuan Chen
  - affiliation: 1
  - name: Mark Khait
    affiliation: 5
  - name: Xiaocong Lyu
    affiliation: 6
  - name: Xiaoming Tian
    affiliation: 7
    orcid: 0000-0003-0642-6064
  - name: Stephan de Hoop
    affiliation: 8
  - name: Artur Palha
    orcid: 0000-0002-3217-0747
    affiliation: 9
affiliations:
 - name: Department of Geoscience and Engineering, TU Delft, Delft, Netherlands
   index: 1
 - name: Energy Science and Engineering Department, Stanford, CA, USA
   index: 2
 - name: Netherlands eScience Center, Amsterdam, The Netherlands
   index: 3
 - name: Petrobras, Petr\'oleo Brasileiro S.A., Rio de Janeiro, Brazil
   index: 4
 - name: Stone Ridge Technology S.R.L., Milan, Italy
   index: 5
 - name: State Key Laboratory of Petroleum Resources and Prospecting, China University of Petroleum, Beijing, China
   index: 6   
 - name: Guangzhou Institute of Energy Conversion, Chinese Academy of Sciences, Guangzhou, China
   index: 7
 - name: ALTEN, The Netherlands
   index: 8   
 - name: Delft Institute of Applied Mathematics, Delft University of Technology
   index: 9

date: 1 April 2024
bibliography: paper.bib
---
# Summary

Open Delft Advanced Research Terra Simulator [@openDARTS_2023] is a simulation framework for forward and inverse modelling and uncertainty quantification of multi-physics processes in geo-engineering applications such as geothermal, CO2 sequestration, water pumping, and hydrogen storage. To efficiently achieve high levels of accuracy on complex geometries, it utilizes advanced numerical methods such as fully implicit thermo-hydro-mechanical-chemical (THMC) formulation, a highly flexible finite-volume spatial approximation and operator-based linearization for nonlinear terms. openDARTS goals are computational efficiency, extensibility, and simplicity of use. For this reason, openDARTS is based on a hybrid design with an efficient core C++ implementation wrapped around a highly customizable and easy-to-use Python code.

# Statement of need

openDARTS is designed to use Python as its user interface (while other simulators such as GEOS [@GEOS] and MRST [@MRST2019] have C++ and Matlab interfaces, respectively), which makes it easy to use in education and research: 6 graduate courses at TU Delft and 4 external courses, 8 research projects in collaboration with industrial partners and 7 academic projects with various research institutes are considered in 2024. It is a reservoir simulator with advanced capabilities that are not reliant on proprietary nor licensed software, thus significantly reducing the entry barrier for researchers and students interested in energy transition applications for the subsurface. The modules `discretizer` and [`darts-flash`](https://gitlab.com/open-darts/darts-flash) allow efficient processing of Corner Point Geometry meshes and advanced multiphase equilibrium evaluation for complex fluids respectively.

The openDARTS framework is fully validated and benchmarked for geothermal applications showing similar accuracy as state-of-the-art simulators TOUGH2 [@TOUGH2001] and AD-GPRS [@ADGPRS2018] while providing a noticeable reduction in CPU time mainly due to the OBL approach [@Wang2020]. While openDARTS uses the OBL approach to cache evaluation points and calculate derivatives through interpolation, TOUGH2 uses numerical derivatives and AD-GPRS automatic differentiation. In the modeling of CO2 geological storage, openDARTS was one of the frameworks tested on the FluidFlower validation benchmark study [@fluidflower2023], where it was compared with experiments and other simulators [@Wapperom2023; Hoop2024; Ahusborde2024]. openDARTS has been used for studying hydrocarbon production when it was validated against commercial simulator [@Lyu2021]. Recently, the modeling of fault reactivation has been supported in openDARTS [@NovikovThesis] that has been validated against semi-analytical benchmarks and PorePy simulation tool [@PorePy].

OpenDARTS' primary advantage over other simulators is its ability to simultaneously provide an OBL implementation, inverse capabilities, and a flexible, modular framework without compromising on performance [@Khait2021]. Its versatility is evidenced by its capability to cater to a wide range of applications. Furthermore, advanced inverse capabilities based on adjoint gradients allow openDARTS to effectively address data assimilation [@Tian2024] and uncertainty quantification [@Wang2023] for energy transition applications.

# Key features

## Unified thermal-compositional PDE formulation

openDARTS has a generic PDE formulation for thermal compositional flow in porous media [@Khait2018]. This makes it possible to adjust terms in the PDEs to account for various multi-physical phenomena such as darcy flow, gravity, multi-component & multiphase flows, thermal flows, chemical and kinetic reactions, etc.

## Geomechanics

openDARTS is capable of modeling coupled THMC processes in linear thermo-poroelastic media under the assumption of small deformations. Unlike most of simulators, the system of conservation laws is handled by the Finite Volume Method alone that enables support of a wide range of cell topologies. Moreover, the method benefits from the unified formulation of mass, energy and momentum fluxes discretized with multi-point approximations. This formulation allows for calculating displacements and stresses in a single collocated grid for all physics phenomena on complex meshes [@Novikov2022]. The framework is suitable for solving multi-scale hydro-mechanical, discrete fracture networks [@Hoop2022], and friction contact mechanics (slip-fault) problems [@Novikov2024].

## Discretization

openDARTS employs the finite volume method for spatial discretization and the fully implicit backward Euler method for time discretization. This approach supports arbitrary star-shaped polyhedral cells, offering high flexibility. Additionally, openDARTS implements both two-point and multi-point flux approximations.

Different grid types supported by openDARTS are *a) structured grids* for teaching and basic modelling, *b) radial grids* for near-well and core scale laboratory experiments, *c) corner-point geometries* for industry-related applications, *d) unstructured grids* for modelling of flow with complex geometries and discrete fracture networks.

## Operator-Based Linearization

One of the most computationally complex and expensive parts is the calculation of partial derivatives to construct the Jacobian. openDARTS exploits Operator-Based Linearization (OBL) [@Voskov2017; @Khait2017], where the terms in the PDEs are separated into space-dependent terms and thermodynamic state-dependent operators. The latter can be parameterized with respect to the nonlinear unknowns using multidimensional tables at different resolutions. The values and derivatives required for the assembly of the linear system can be approximated through multi-linear interpolation in the parameter space using calculated values at the nodes.

Using adaptive parametrization [@Khait2018], derivative computation is performed at nodes of the structured grid in the primary variables space around the required point. Re-using computed values at nodal points can significantly reduce the Jacobian construction stage, especially in the case of ensemble-based simulations.

## Inverse modelling

Inverse modelling methods necessitate a substantial number of simulations to accurately calibrate model parameters against observed data. Such algorithms are highly computationally intensive, particularly when employing gradient-based methods. The implementation of the adjoint method in openDARTS remarkably enhances its efficiency in computing the required gradients for inverse modelling or history-matching processes [@Tian2023]. Moreover, the flexibility of openDARTS's Python interface significantly simplifies the coupling process with various data assimilation algorithms. The inverse modelling module of openDARTS accommodates various types of observation data such as: well rates, well temperatures, BHP, time-lapse temperature distributions, and any custom outputs definable in the form of operators within openDARTS.

## Software implementation

The most computationally expensive part of openDARTS is written in C++ with OpenMP parallelization.
openDARTS can be installed as a Python module and it has a Python-based interface, which makes it suitable for teaching and users unfamiliar with C++ language. There are several benefits of this approach compared to a code fully written in C++.

- Easy installation via pip and PyPI.
- No need to install compilers.
- Flexible implementation of simulation framework, physical modelling and grids.
- Easy data visualization, including internal arrays and `vtk`.
- Use popular Python modules within openDARTS and the user's model for data processing and input/output.
- Coupling with other Python-based numerical modelling software.

# Acknowledgements

The authors would like to acknowledge the Netherlands eScience Center for the funding provided under grant number NLESC.OEC.2021.026.

# References
