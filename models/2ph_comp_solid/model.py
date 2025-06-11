from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params, value_vector
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.kinetics import KineticBasic


# Model class creation here!
class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        self.set_sim_params(first_ts=0.001, mult_ts=2, max_ts=1, runtime=1000, tol_newton=1e-5, tol_linear=1e-6,
                            it_newton=10, it_linear=50, newton_type=sim_params.newton_local_chop)
        self.data_ts.newton_tol_stationary = 1e-5

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        trans_exp = 3
        perm = 100  # / (1 - solid_init) ** trans_exp
        """Reservoir"""
        nx = 1000
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=1, dy=1, dz=1,
                                         permx=perm, permy=perm, permz=perm / 10, poro=1, depth=1000)
        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", cell_index=(1, 1, 1))
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", cell_index=(self.reservoir.nx, 1, 1))

    def set_physics(self):
        self.zero = 1e-12

        components = ['CO2', 'Ions', 'H2O', 'CaCO3']
        phases = ['gas', 'wat', 'sol']
        nc = len(components)
        nc_fl = nc-1
        thermal = 0
        ne = nc + thermal
        Mw = [44.01, (40.078 + 60.008) / 2, 18.015, 100.086, ]

        stoich = [0, -1, 0, 1]
        init_ions = 0.5
        solid_init = 0.7
        equi_prod = (init_ions / 2) ** 2
        solid_inject = self.zero

        zc_fl_init = [self.zero / (1 - solid_init), init_ions]
        zc_fl_init = zc_fl_init + [1 - sum(zc_fl_init)]
        self.ini_stream = [x * (1 - solid_init) for x in zc_fl_init]

        zc_fl_inj_composition_gas = [1 - 2 * self.zero / (1 - solid_inject), self.zero / (1 - solid_inject)]
        zc_fl_inj_composition_gas = zc_fl_inj_composition_gas + [1 - sum(zc_fl_inj_composition_gas)]
        self.inj_composition = [x * (1 - solid_inject) for x in zc_fl_inj_composition_gas]

        """Physical properties"""
        # Create property containers:
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=Mw, nc_sol=1, np_sol=1,
                                               temperature=1., rock_comp=1e-7, min_z=self.zero / 10)

        """ properties correlations """
        property_container.flash_ev = ConstantK(nc - 1, [10, 1e-12, 1e-1], self.zero)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-4, dens0=100)),
                                              ('wat', DensityBasic(compr=1e-6, dens0=1000)),
                                              ('sol', ConstFunc(2000.))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.1)),
                                                ('wat', ConstFunc(1))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('wat', PhaseRelPerm("wat"))])
        property_container.diffusion_ev = dict([('gas', ConstFunc(np.ones(nc) * 1e-9)),
                                                ('wat', ConstFunc(np.ones(nc) * 1e-9))])

        property_container.kinetic_rate_ev[0] = KineticBasic(equi_prod, 1e-0, ne)

        """ Activate physics """
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=101, min_p=1, max_p=1000, min_z=self.zero / 10, max_z=1 - self.zero / 10)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 95,
                              self.physics.vars[1]: self.ini_stream[0],
                              self.physics.vars[2]: self.ini_stream[1],
                              self.physics.vars[3]: self.ini_stream[2]
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MOLAR_RATE,
                                               is_inj=True, target=0.2, phase_name='gas',
                                               inj_composition=self.inj_composition)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=50.)

    def print_and_plot(self, filename):
        import matplotlib.pyplot as plt

        nc = self.physics.nc
        nb = self.mesh.n_res_blocks
        Sg = np.zeros(nb)
        Ss = np.zeros(nb)
        X = np.zeros((nb, nc - 1, 2))

        rel_perm = np.zeros((nb, 2))
        visc = np.zeros((nb, 2))
        density = np.zeros((nb, 3))
        density_m = np.zeros((nb, 3))

        Xn = np.array(self.engine.X, copy=True)


        P = Xn[0:nb * nc:nc]
        z_caco3 = 1 - (Xn[1:nb * nc:nc] + Xn[2:nb * nc:nc] + Xn[3:nb * nc:nc])

        z_co2 = Xn[1:nb * nc:nc] / (1 - z_caco3)
        z_inert = Xn[2:nb * nc:nc] / (1 - z_caco3)
        z_h2o = Xn[3:nb * nc:nc] / (1 - z_caco3)

        for ii in range(nb):
            x_list = Xn[ii*nc:(ii+1)*nc]
            state = value_vector(x_list)
            (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property_container.evaluate(state)

            rel_perm[ii, :] = kr
            visc[ii, :] = mu
            density[ii, :2] = rho
            density_m[ii, :2] = rho_m

            density[2] = self.property_container.solid_dens[-1]

            X[ii, :, 0] = x[1][:-1]
            X[ii, :, 1] = x[0][:-1]
            Sg[ii] = sat[0]
            Ss[ii] = z_caco3[ii]

        # Write all output to a file:
        with open(filename, 'w+') as f:
            # Print headers:
            print('//Gridblock\t gas_sat\t pressure\t C_m\t poro\t co2_liq\t co2_vap\t h2o_liq\t h2o_vap\t ca_plus_co3_liq\t liq_dens\t vap_dens\t solid_dens\t liq_mole_dens\t vap_mole_dens\t solid_mole_dens\t rel_perm_liq\t rel_perm_gas\t visc_liq\t visc_gas', file=f)
            print('//[-]\t [-]\t [bar]\t [kmole/m3]\t [-]\t [-]\t [-]\t [-]\t [-]\t [-]\t [kg/m3]\t [kg/m3]\t [kg/m3]\t [kmole/m3]\t [kmole/m3]\t [kmole/m3]\t [-]\t [-]\t [cP]\t [cP]', file=f)
            for ii in range (nb):
                print('{:d}\t {:6.5f}\t {:7.5f}\t {:7.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:8.5f}\t {:8.5f}\t {:8.5f}\t {:7.5f}\t {:7.5f}\t {:7.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}'.format(
                    ii, Sg[ii], P[ii], Ss[ii] * density_m[ii, 2], 1 - Ss[ii], X[ii, 0, 0], X[ii, 0, 1], X[ii, 2, 0], X[ii, 2, 1], X[ii, 1, 0],
                    density[ii, 0], density[ii, 1], density[ii, 2], density_m[ii, 0], density_m[ii, 1], density_m[ii, 2],
                    rel_perm[ii, 0], rel_perm[ii, 1], visc[ii, 0], visc[ii, 1]), file=f)

        """ start plots """

        font_dict_title = {'family': 'sans-serif',
                           'color': 'black',
                           'weight': 'normal',
                           'size': 14,
                           }

        font_dict_axes = {'family': 'monospace',
                          'color': 'black',
                          'weight': 'normal',
                          'size': 14,
                          }

        fig, axs = plt.subplots(3, 3, figsize=(12, 10), dpi=200, facecolor='w', edgecolor='k')
        """ sg and x """
        axs[0][0].plot(z_co2, 'b')
        axs[0][0].set_xlabel('x [m]', font_dict_axes)
        axs[0][0].set_ylabel('$z_{CO_2}$ [-]', font_dict_axes)
        axs[0][0].set_title('Fluid composition', fontdict=font_dict_title)

        axs[0][1].plot(z_h2o, 'b')
        axs[0][1].set_xlabel('x [m]', font_dict_axes)
        axs[0][1].set_ylabel('$z_{H_2O}$ [-]', font_dict_axes)
        axs[0][1].set_title('Fluid composition', fontdict=font_dict_title)

        axs[0][2].plot(z_inert, 'b')
        axs[0][2].set_xlabel('x [m]', font_dict_axes)
        axs[0][2].set_ylabel('$z_{w, Ca+2} + z_{w, CO_3-2}$ [-]', font_dict_axes)
        axs[0][2].set_title('Fluid composition', fontdict=font_dict_title)

        axs[1][0].plot(X[:, 0, 0], 'b')
        axs[1][0].set_xlabel('x [m]', font_dict_axes)
        axs[1][0].set_ylabel('$x_{w, CO_2}$ [-]', font_dict_axes)
        axs[1][0].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[1][1].plot(X[:, 2, 0], 'b')
        axs[1][1].set_xlabel('x [m]', font_dict_axes)
        axs[1][1].set_ylabel('$x_{w, H_2O}$ [-]', font_dict_axes)
        axs[1][1].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[1][2].plot(X[:, 1, 0], 'b')
        axs[1][2].set_xlabel('x [m]', font_dict_axes)
        axs[1][2].set_ylabel('$x_{w, Ca+2} + x_{w, CO_3-2}$ [-]', font_dict_axes)
        axs[1][2].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[2][0].plot(P, 'b')
        axs[2][0].set_xlabel('x [m]', font_dict_axes)
        axs[2][0].set_ylabel('$p$ [bar]', font_dict_axes)
        axs[2][0].set_title('Pressure', fontdict=font_dict_title)

        axs[2][1].plot(Sg, 'b')
        axs[2][1].set_xlabel('x [m]', font_dict_axes)
        axs[2][1].set_ylabel('$s_g$ [-]', font_dict_axes)
        axs[2][1].set_title('Gas saturation', fontdict=font_dict_title)

        axs[2][2].plot(1 - Ss, 'b')
        axs[2][2].set_xlabel('x [m]', font_dict_axes)
        axs[2][2].set_ylabel('$\phi$ [-]', font_dict_axes)
        axs[2][2].set_title('Porosity', fontdict=font_dict_title)

        left = 0.05  # the left side of the subplots of the figure
        right = 0.95  # the right side of the subplots of the figure
        bottom = 0.05  # the bottom of the subplots of the figure
        top = 0.95  # the top of the subplots of the figure
        wspace = 0.25  # the amount of width reserved for blank space between subplots
        hspace = 0.25  # the amount of height reserved for white space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        for ii in range(3):
            for jj in range(3):
                for tick in axs[ii][jj].xaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

                for tick in axs[ii][jj].yaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

        plt.tight_layout()
        plt.savefig("results_kinetic_brief.pdf")
        plt.show()
