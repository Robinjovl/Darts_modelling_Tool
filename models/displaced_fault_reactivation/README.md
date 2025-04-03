
## Description
A single fault slip model


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


