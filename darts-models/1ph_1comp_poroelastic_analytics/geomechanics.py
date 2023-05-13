import numpy as np
import compaction as cpt
#from numba import njit

# arg: t - 1D array of 6 values in Voight notation
# returns (3x3) tensor
def get_tensor_from_voight(t):
    res = np.zeros((3,3))
    res[0,0] = t[0]
    res[1,1] = t[1]
    res[2,2] = t[2]
    res[1,2] = res[2,1] = t[3] #symmetric
    res[0,2] = res[2,0] = t[4]
    res[0,1] = res[1,0] = t[5]
    return res

# compute derivative in point x using finite difference - central schema
# u_minus = u(x - step)
# u_plus  = u(x + step)
def deriv(u_minus, u_plus, step):
    return 0.5 * (u_plus - u_minus) / step

class geomech():
    def __init__(self):
        # elastic constants
        self.poisson = 0.25
        self.young = 1000.  # MPa

        # thermal expansion coefficient
        self.thermal_exp_coeff = 1.3e-5  # 1/°C

        # Mohr-Coulomb
        self.cohesion = 0  # assume no cohesion due to healing
        self.friction = 0.15

        # set initial stress (values from Daniliidis paper)
        self.sigma_v = 45  # MPa
        self.sigma_H = 34  # MPa
        self.sigma_h = 30  # MPa

        self.stress_initial = self.sigma_v
        # TODO sigma_H and sigma_h are not parallel to x and y axis -> rotate grid?


    def calc_thermoelastic_stress(self, delta_temperature):
        # thermoelastic stress (assume there is no interaction between cells)
        # stress_thermoelastic is array for the whole 3D reservoir defined at cell centers

        # delta_temperature is T_init - T_current, so if temperature decreased,
        #delta_temperature = np.zeros(nx*ny*nz) # °C

        self.stress_thermoelastic_1d = self.thermal_exp_coeff * self.young     \
                                       / (1 - self.poisson) * delta_temperature

        # to make the same shape as other arrays in stress formula
        stress_thermoelastic = np.zeros((6, self.stress_thermoelastic_1d.shape[0]))

        # 0,1,2 - diagonal, off-diagonals are zero
        stress_thermoelastic[0,:] = self.stress_thermoelastic_1d[:]
        stress_thermoelastic[1,:] = self.stress_thermoelastic_1d[:]
        stress_thermoelastic[2,:] = self.stress_thermoelastic_1d[:]

        #TODO thermal expansion is not symmetric, use correction

        return stress_thermoelastic

    def calc_displacements(self, points, prisms, delta_pressure):
        '''
        arg: points: points where to compute
        arg: prisms, cells geometry
        arg: delta_pressure, in MPa
        return: displacements at points, in meters
        '''
        ux = cpt.displacement_x_component(points, prisms, delta_pressure, self.poisson, self.young)
        uy = cpt.displacement_y_component(points, prisms, delta_pressure, self.poisson, self.young)
        uz = cpt.displacement_z_component(points, prisms, delta_pressure, self.poisson, self.young)
        return ux, uy, uz


    # calculate strain and stress tensors from displacements on fault_surface
    #@njit
    def calc_strain_stress(self, fault_surface, prisms, delta_pressure):
        # compute displacement derivatives
        step_x = step_y = step_z = 1  # step for derivatives, m.

        fault_surface_y_plus = fault_surface.copy()
        fault_surface_y_plus[0, :] += step_y
        fault_surface_y_minus = fault_surface.copy()
        fault_surface_y_minus[0, :] -= step_y
        fault_surface_x_plus = fault_surface.copy()
        fault_surface_x_plus[1, :] += step_x
        fault_surface_x_minus = fault_surface.copy()
        fault_surface_x_minus[1, :] -= step_x
        fault_surface_z_plus = fault_surface.copy()
        fault_surface_z_plus[2, :] += step_z
        fault_surface_z_minus = fault_surface.copy()
        fault_surface_z_minus[2, :] -= step_z

        # ux, uy, uz = get_displacements_on_surface(fault_surface)

        ux_y_plus,  uy_y_plus,  uz_y_plus  = self.calc_displacements(fault_surface_y_plus, prisms, delta_pressure)
        ux_y_minus, uy_y_minus, uz_y_minus = self.calc_displacements(fault_surface_y_minus, prisms, delta_pressure)

        ux_x_plus,  uy_x_plus,  uz_x_plus  = self.calc_displacements(fault_surface_x_plus, prisms, delta_pressure)
        ux_x_minus, uy_x_minus, uz_x_minus = self.calc_displacements(fault_surface_x_minus, prisms, delta_pressure)

        ux_z_plus,  uy_z_plus,  uz_z_plus  = self.calc_displacements(fault_surface_z_plus, prisms, delta_pressure)
        ux_z_minus, uy_z_minus, uz_z_minus = self.calc_displacements(fault_surface_z_minus, prisms, delta_pressure)

        dux_dx = deriv(ux_x_minus, ux_x_plus, step_x)
        dux_dy = deriv(ux_y_minus, ux_y_plus, step_y)
        dux_dz = deriv(ux_z_minus, ux_z_plus, step_z)

        duy_dx = deriv(uy_x_minus, uy_x_plus, step_x)
        duy_dy = deriv(uy_y_minus, uy_y_plus, step_y)
        duy_dz = deriv(uy_z_minus, uy_z_plus, step_z)

        duz_dx = deriv(uz_x_minus, uz_x_plus, step_x)
        duz_dy = deriv(uz_y_minus, uz_y_plus, step_y)
        duz_dz = deriv(uz_z_minus, uz_z_plus, step_z)

        # Voight notation: 6 values for each fault point: 3 diagonal (xx, yy, zz) + 3 off-diagonal values (yz, xz, xy)
        strain = np.vstack(
            [dux_dx, duy_dy, duz_dz,
             0.5 * (duy_dz + duz_dy),
             0.5 * (dux_dz + duz_dx),
             0.5 * (dux_dy + duy_dx)])

        # volumetric_strain=div(displ)
        volumetric_strain = dux_dx + duy_dy + duz_dz

        # Voight notation: 3 diagonal and 3 off-diagonal values
        n_points = fault_surface.shape[1]
        kronecker = np.array(n_points * [1, 1, 1, 0, 0, 0]).reshape(n_points, 6).transpose()

        #print('strain', strain.shape)
        #print('kronecker', kronecker.shape)

        stress = self.young * (strain + self.poisson / (1 - 2 * self.poisson) *
                               volumetric_strain * kronecker) / (1 + self.poisson)
        return stress, strain


class fault():
    def __init__(self, centers, nx, ny, nz):
        # define fault geometry (vertical plane x=middle_x)
        eps = 1e-2 # center of cell has instability of analytic solution, so add the epsilon

        self.dims = (nz, ny)
        self.fault_axis_labels = ['Y', 'Z']

        y_min = centers[0,:].min()
        y_max = centers[0,:].max()
        x_min = centers[1,:].min()
        x_max = centers[1,:].max()
        z_min = centers[2,:].min()
        z_max = centers[2,:].max()

        fault_z = np.linspace(z_min, z_max, self.dims[0]) #TODO avoid center of cell
        fault_y = np.linspace(y_min, y_max, self.dims[1]) #TODO avoid center of cell
        fault_x = 0.5*(x_max + x_min) + eps  # middle point

        #fault_z[:] = centers[0,2] # for testing distance

        fault_y2, fault_z2 = np.meshgrid(fault_y, fault_z)

        fault_z1 = fault_z2.ravel()
        fault_y1 = fault_y2.ravel()
        fault_x1 = np.zeros(fault_y1.shape) + fault_x
        self.surface = np.vstack([fault_y1, fault_x1, fault_z1])

        self.n_points = self.surface.shape[1]

        # normal vector to vertical fault
        self.normal = np.array([1, 0, 0]) #TODO
        self.normal_T = np.transpose(self.normal)

        # fill  fault_and_array_correspondence - closest cell indices in 1d global array (like pressure), corresponding to fault points
        self.cell_correspondence = np.zeros(self.n_points, dtype=np.int32)
        for i in range(self.n_points):
            distance = np.sqrt((centers[:, 0] - self.surface[0, i]) ** 2 +
                               (centers[:, 1] - self.surface[1, i]) ** 2 +
                               (centers[:, 2] - self.surface[2, i]) ** 2)
            # print(i, distance.argmin(), distance.min())
            self.cell_correspondence[i] = distance.argmin()


    def calc_stress_on_fault(self, fault, pore_pressure_init, delta_pressure):
        #pore_pressure_init = 20 # MPa

        # get thermal stress values on fault
        stress_thermoelastic_fault = np.zeros((6, fault.n_points))
        # 0,1,2 - diagonal
        stress_thermoelastic_fault[0,:] = self.stress_thermoelastic_1d[fault.cell_correspondence[:]]
        stress_thermoelastic_fault[1,:] = self.stress_thermoelastic_1d[fault.cell_correspondence[:]]
        stress_thermoelastic_fault[2,:] = self.stress_thermoelastic_1d[fault.cell_correspondence[:]]

        pore_pressure = pore_pressure_init + delta_pressure # p_current = p_init + delta_p # MPa
        pore_pressure_fault = pore_pressure[fault.fault_and_array_correspondence[:]]

        # do not add pore_pressure assuming hydrostatic part of stress is already defined in sigma_v

        # stress_poroelastic - this is delta stress due to delta pressure
        # calc strain and stress on fault from displacements
        stress_poroelastic_fault, strain_fault = self.calc_strain_stress(fault.surface)

        # stress_thermoelastic - this is delta stress due to delta teperature
        stress_total_fault = self.stress_initial + stress_poroelastic_fault + stress_thermoelastic_fault
        return stress_total_fault

    def mohr_coulomb(self, fault, stress_total_fault, pore_pressure_fault):
        stress_t = np.zeros(fault.n_points)  # tangential vector for each fault point
        stress_n = np.zeros(fault.n_points)  # tangential vector for each fault point
        for i in range(fault.n_points):
            stress_total_fault_tensor = get_tensor_from_voight(stress_total_fault[:, i])
            stress_t[i] = np.linalg.norm(
                (np.eye(3, 3) - np.tensordot(fault.normal, fault.normal_T, axes=0))
                @ stress_total_fault_tensor @ fault.normal)
            stress_n[i] = np.dot(np.transpose(fault.normal), stress_total_fault_tensor @ fault.normal)

        mcc_n_part = self.cohesion + self.friction * (stress_n - pore_pressure_fault)
        mcc = np.abs(stress_t) - mcc_n_part
        mcc_2d = mcc.reshape(fault.dims)

        print ('Values on fault:')
        print ('mcc min =', mcc.min(), 'max =', mcc.max())
        print ('shear stress max =', np.abs(stress_t).max())
        print ('stress_n max =', stress_n.max())
        print ('pore_pressure max =', pore_pressure_fault.max())
        print ('C+m(sigma_n-p) max =', mcc_n_part.max())