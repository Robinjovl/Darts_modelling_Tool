# F.A.Q.

1. How to disable the gravity?  Call the `init_grav_coef` with an argument equal to 0 after the Model.init() call: `self.reservoir.mesh.init_grav_coef(0)`
2. What is the variable order in a **state** vector?

   The order in a **state** vector is:
   * Geothermal: Pressure, Enthalpy
   * Compositional: Pressure, Compositions, Temperature, Displacements (x,y,z)
3. Why there are two variable `n_blocks `and `n_res_blocks `in the mesh object?

   There are 2 additional cells per well are internally added by open-DARTS for the non-multi-segment well case, so `n_blocks = n_res_blocks = 2 * n_wells`

   For both multi_segment =True and False, DARTS adds one additional cell per well, and then one more for **all** well connections if multi_segment = False - in that case  the in/out flow from all well connections immediately come to the well rate. If multi_segment is True, DARTS adds one more extra block for **each** well connection to account for the flow friction in a well tube. 