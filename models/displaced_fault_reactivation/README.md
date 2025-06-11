
## Description
This geomechanical poroelastic model with contact mechanics is single-phase and single-component.
The model contains a single displaced fault, where the slip condition is evaluated using a constant friction coefficient or dynamic friction coefficient evaluated by slip-weakening law or RSF.
The numerical slip solution is compared against the analytical solution. The fault slip is triggered by stress changes induced by pore pressure change,
whereas the pore pressure can be changes by setting uniform depletion or adding a production well.
Geomechanics in this model is based on the quasi-static formulation before the slip, and can be changed to dynamic formulation during the slip.

## Notes
During the first run, the script creates a file 'cached_preprocessing.pkl' which contains processed mesh data. For next run, the script check that file existance and if it exists, 
the mesh processing will be skipped and the data from the file will be used instead to reduce the initialization time. 
If the reservoir geometry was changed, the file 'cached_preprocessing.pkl' should be manually deleted.

## Output
This test outputs two groups of VTK files: one for the domain and one for the fault.

The domain output (3D) contains:
- p - pressure, bars
- u_x, u_y, u_z - displacements, m
- stress - effective stresses tensor, bars (negative values are compressive)
- tot_stress - total stresses tensor, bars; tot_stress = stress - p
- porosity - porosity, dimensionless (0..1)

The fault output (2D) contains:
- g_local - slip value, where 'X' stands for the normal to the fault direction, 'Y' and 'Z' tangential direction
- f_local - traction, with the same magnitudes direction names


