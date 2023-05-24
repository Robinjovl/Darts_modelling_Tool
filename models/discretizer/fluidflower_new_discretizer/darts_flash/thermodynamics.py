import numpy as np
from flash import Flash


class Thermodynamics:
    def __init__(self, fl_comp: list, fl_phases: list, flash_ev: Flash):
        # Fluid components and ions
        self.fl_comp = fl_comp[:]
        self.nc_fl = len(fl_comp)

        self.f = np.zeros(self.nc_fl)

        self.comp_in_z = fl_comp[:]  # Components for which conservation equation exists

        # Fluid phases
        self.phases = fl_phases[:]
        self.np_fl = len(fl_phases)

        # Flash
        self.flash_ev = flash_ev

        # Kinetic components and phases, absent in simple systems
        self.kin_comp = None
        self.nc_kin = 0

        self.kin_phases = None
        self.np_kin = 0
        self.kin_in_vx = True

    def evaluate_flash(self, pressure, temperature, zc):
        # 1. Evaluate flash
        v, x = self.flash_ev.evaluate(pressure, temperature, zc)

        # 2. Determine phases present
        ph = []
        for j, nu in enumerate(v):
            if nu > 0.:
                ph.append(j)

        return ph, v, x

    def evaluate_kinetics(self, pressure, temperature, x, sat):
        return np.zeros(self.nc_fl), 0.

    def calculate_solid_sat(self, sat):
        return 0


class ReactiveThermodynamics(Thermodynamics):
    ion_charge = {"Na+": 1, "Cl-": -1, "Ca+2": 2, "CO3-2": -2, "I-": -1}

    def __init__(self, fl_comp: list, fl_phases: list, flash_ev: Flash,
                 ions: list = None, combined_ions: list = None,
                 kin_comp: list = None, kin_phases: list = None,
                 kin_in_vx=True, reactions: list = None):
        super().__init__(fl_comp, fl_phases, flash_ev)

        # (Combined) ions
        if ions is not None:
            self.ions = ions
            self.ni = len(ions)

            if combined_ions is not None:
                self.combined_ions = []
                for i, idxs in enumerate(combined_ions):
                    ion_stoich = []
                    for idx in idxs:
                        ion_stoich.append(np.abs(self.ion_charge[ions[idx]]))
                    ion_stoich[:] /= np.sum(ion_stoich)
                    self.combined_ions.append(ion_stoich)
                self.comp_in_z += ['ions{}'.format(i) for i, idxs in enumerate(self.combined_ions)]
            else:
                self.combined_ions = None
                self.comp_in_z += ions[:]

        else:
            self.ions = None
            self.ni = 0
            self.combined_ions = None

        # Kinetic components and phases
        self.kin_comp = kin_comp if kin_comp is not None else None
        self.nc_kin = len(kin_comp) if kin_comp is not None else 0

        if kin_comp is not None:
            self.comp_in_z += kin_comp

        self.kin_phases = kin_phases
        self.np_kin = len(kin_phases) if kin_phases is not None else 0
        if kin_phases is not None:
            self.phases += kin_phases

        self.kin_in_vx = kin_in_vx

        # Total number of components
        self.nc_tot = len(self.comp_in_z)

        # Kinetic reactions
        self.reactions = reactions

    def evaluate_flash(self, pressure, temperature, zc):
        """
        1. Remove non-flash components, normalize and separate combined ions
        2. Run flash
        3. Lump combined ions and re-normalize for non-flash components and phases
        4. Determine phases present
        """
        # 1a. Remove non-flash components and normalize
        if self.nc_kin:
            zc, zs = self.normalize(zc)
        # 1b. Separate combined ions
        if self.combined_ions is not None:
            zc = self.separate_ions(zc)

        # 2. Run flash
        v, x = self.flash_ev.evaluate(pressure, temperature, zc)

        # 3a. Lump combined ions
        if self.combined_ions is not None:
            x = self.lump_ions(x)
        # 3b. Re-normalize non-flash components and phases
        if self.kin_in_vx:
            v, x = self.renormalize(v, x, zs)

        # 4. Determine phases present
        ph = []
        for j, nu in enumerate(v):
            if nu > 0.:
                ph.append(j)

        return ph, v, x

    def evaluate_kinetics(self, pressure, temperature, X, sat):
        """
        Calculate kinetic rates
        """
        rates = np.zeros(self.nc_tot)
        reaction_enth = 0.
        for k, reaction in enumerate(self.reactions):
            rate = reaction.evaluate(pressure, temperature, X, sat)
            reaction_enth += reaction.evaluate_enthalpy(pressure, temperature, X, sat)
            rates += rate

        return rates, reaction_enth

    def separate_ions(self, x):
        """
        If combined ions, un-lump them
        """
        X = np.zeros(self.nc_fl + self.ni)

        # Separate the combined ions according to stoichiometry
        X[:self.nc_fl] = x[:self.nc_fl]
        ii = 0
        for i, ion_stoich in enumerate(self.combined_ions):
            for stoich in ion_stoich:
                X[self.nc_fl + ii] = x[self.nc_fl + i] * stoich
                ii += 1
        return X

    def lump_ions(self, x):
        """
        Lump combined ions
        """
        # 1. If ions were combined in z, lump them together again
        X = np.zeros((self.np_fl, self.nc_fl + len(self.combined_ions)))
        X[:, :self.nc_fl] = x[:, :self.nc_fl]
        ii = 0
        for i, ion_stoich in enumerate(self.combined_ions):
            for stoich in ion_stoich:
                for j in range(self.np_fl):
                    X[j, self.nc_fl + i] += x[j, self.nc_fl + ii]
                ii += 1
        return X

    def normalize(self, zc):
        """
        Normalize flash input: remove non-flash components from zc and normalize
        """
        # Remove non-flash components from zc
        zs = zc[-self.nc_kin:]
        zc = zc[:-self.nc_kin] / (1 - np.sum(zs))

        return zc, zs

    def renormalize(self, v, x, zs):
        """
        Re-normalize flash output: Include solid components to v and x
        """
        # Include kinetic components again
        np_tot = self.np_fl + self.np_kin

        V = np.zeros(np_tot)
        X = np.zeros((np_tot, self.nc_tot))

        for j in range(self.np_fl):
            V[j] = v[j] * (1 - np.sum(zs))
            for i in range(self.nc_tot-self.nc_kin):
                X[j, i] = x[j, i]

        for j in range(self.np_kin):
            V[self.np_fl + j] = zs[j]
            Xs = np.zeros(self.nc_tot)
            Xs[self.nc_tot - self.nc_kin + j] = 1.
            X[self.np_fl + j, :] = Xs

        return V, X

    def calculate_solid_sat(self, sat):
        # Calculate solid saturation
        if self.kin_in_vx:
            solid_sat = np.sum(sat[self.np_fl:])
        else:
            solid_sat = np.sum(zc[-self.nm:])
            for j in range(self.np_kin):
                solid_dens = self.density_ev[self.phases_name[self.np_fl + j]].evaluate(pressure, temperature, self.x)
                self.solid_dens[j] = solid_dens / self.Mw[self.nc_fl + j]
        return solid_sat
