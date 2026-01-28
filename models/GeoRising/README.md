## Features
- Fluid Physics: Geothermal
- Type of Grid: Cartesian
- Discretization: TPFA
- Initialization: Constants
- Well type: Standard
- Wells control: Volumetric Rate

## Description
This model used one of the realizations (66) of [EGG model](https://data.4tu.nl/articles/dataset/The_Egg_Model_-_data_files/12707642) 
permeability for the first 3 layers (60x60x3) and without applied ACTNUM (ideal brick shape). The model 
contains a geothermal doublet with an injector located at [30, 14] and a producer at [30, 46]. Initial 
conditions are constant pressure (p = 200) and temperature (T = 350K) and wells are controlled by equal 
volumetric rate (q = 8000 m3/day) with low re-injection temperature (T = 300K). The model uses 
conventional geothermal physics based on the IAPWS-95 equation of state.

## Notes


## Output


