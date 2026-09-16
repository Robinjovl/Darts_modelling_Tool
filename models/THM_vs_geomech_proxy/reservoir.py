import numpy as np
import os
import time
import glob
import hashlib
import json
import meshio
from darts.discretizer import elem_type, elem_loc
from darts.discretizer import matrix33 as disc_matrix33
from darts.discretizer import Stiffness as disc_stiffness
from darts.discretizer import vector_matrix33, stf_vector
from darts.reservoirs.unstruct_reservoir_mech import set_domain_tags, get_lambda_mu, get_biot_modulus, get_rock_compressibility
from darts.reservoirs.unstruct_reservoir_mech import UnstructReservoirMech
from darts.input.input_data import InputData
from darts.engines import timer_node, ms_well, ms_well_vector
import copy
from scipy.interpolate import griddata as gd
from functools import reduce

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The expensive discretization steps (displacement gradients, interface and
# cell-centered stress approximations) are cached in meshes/<case>/.cache/ and
# reused while nothing they depend on changes, see
# UnstructReservoirCustom.discretization_cache_key. DARTS_DISCR_CACHE=0 disables it.
DISCR_CACHE_FORMAT = 1
DISCR_CACHE_KEEP = 2  # cache files kept per mesh folder (most recently used); they are large


def file_sha256(filename):
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_update(h, obj):
    """Feed obj (None, scalars, strings, arrays, lists, dicts) into the hash h, values bit-exact."""
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            h.update(repr(str(k)).encode())
            hash_update(h, obj[k])
        return
    if obj is None or isinstance(obj, str):
        h.update(repr(obj).encode())
        return
    try:
        a = np.asarray(obj)
    except ValueError:  # ragged nested lists
        a = None
    if a is None or a.dtype == object:
        if isinstance(obj, (list, tuple)) or (isinstance(obj, np.ndarray) and obj.ndim > 0):
            h.update(b'[%d]' % len(obj))
            for x in obj:
                hash_update(h, x)
        else:
            h.update(repr(obj).encode())
        return
    h.update(('%s%s' % (a.dtype.str, a.shape)).encode())
    h.update(np.ascontiguousarray(a).tobytes())


class UnstructReservoirCustom(UnstructReservoirMech):
    def __init__(self, timer, idata: InputData, model_folder, fluid_vars=['p'], uniform_props=False, generate_mesh=False,
                 cache_discretization=None):
        self.idata = idata
        if cache_discretization is None:
            cache_discretization = os.getenv('DARTS_DISCR_CACHE', '1') != '0'
        self.cache_discretization = cache_discretization

        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        thermoporoelasticity = True if 'temperature' in fluid_vars else False
        super().__init__(timer, discretizer='mech_discretizer',
                         thermoporoelasticity=thermoporoelasticity, fluid_vars=fluid_vars)

        self.bnd_tags = idata.mesh.bnd_tags
        self.domain_tags = set_domain_tags(matrix_tags=idata.mesh.matrix_tags, bnd_tags=list(self.bnd_tags.values()))

        self.field_reservoir(model_folder=model_folder, idata=idata, uniform_props=uniform_props, generate_mesh=generate_mesh)
        self.init_reservoir_main(idata=idata)
        t1 = None
        if thermoporoelasticity:
            t1 = np.mean(self.t_init)
        self.set_pzt_bounds(p=np.mean(self.p_init), z=self.z_init, t=t1) #TODO how this mean() affects when gradient is applied
        self.wells = []


    def get_reservoir_initial_pressure(self, depths):
        return self.idata.initial.pressure_at_ref_depth + self.idata.initial.pressure_gradient * depths

    def get_reservoir_initial_temperature(self, depths):
        return self.idata.initial.temperature_at_ref_depth + self.idata.initial.temperature_gradient * depths

    def field_reservoir(self, idata: InputData, model_folder, uniform_props=False, generate_mesh=False):

        self.mesh_filename = os.path.join(BASE_DIR, 'meshes', model_folder, 'mesh.msh')
        # Structured cases generate their box mesh here. Unstructured cases
        # such as case_5 generate their mesh externally in main.py and do not
        # define nx/ny/nz or Xc/Yc/Zc.
        generate_structured_mesh = generate_mesh and hasattr(idata.other, 'nx')
        if generate_structured_mesh:
            nx, ny, nz = idata.other.nx, idata.other.ny, idata.other.nz
            self.Xc = idata.other.Xc
            self.Yc = idata.other.Yc
            self.Zc = idata.other.Zc

        # define permeable reservoir geometric boundaries
        self.rsv_top = idata.other.rsv_top
        self.rsv_bottom = idata.other.rsv_bottom
        self.rsv_xy = idata.other.rsv_xy
        self.rsv_x1 = idata.other.rsv_x1
        self.rsv_x2 = idata.other.rsv_x2
        self.rsv_y1 = idata.other.rsv_y1
        self.rsv_y2 = idata.other.rsv_y2

        if generate_structured_mesh:
            print('Mesh generation started')
            self.timer.node["initialization"].node["mesh_generation"] = timer_node()
            self.timer.node["initialization"].node["mesh_generation"].start()
            # refine by Z also around rsv
            #self.Zc = np.hstack([np.arange(0, self.rsv_top-100, 100), np.arange(self.rsv_top-100, self.rsv_bottom+100, 20),np.arange(self.rsv_bottom+100, 6000, 100)])

            # extend by Z more
            #self.Zc = np.hstack([np.arange(0, self.rsv_top, 100),np.arange(self.rsv_top, self.rsv_bottom, 20), np.arange(self.rsv_bottom, 6000, 100), np.array([6500, 10000, 15000])])

            # check case name ane generated arrays are consistent
            assert nx == self.Xc.size-1, "nx = {0}, Xc.size = {1}".format(nx, self.Xc.size)
            assert ny == self.Yc.size-1, "ny = {0}, Yc.size = {1}".format(nx, self.Yc.size)
            assert nz == self.Zc.size-1, "nz = {0}, Zc.size = {1}".format(nz, self.Zc.size)

            # check layers boundaries defined without layers deterioration
            assert np.unique(self.Xc).size == self.Xc.size, "Xc has duplicates {0}".format(self.Xc)
            assert np.unique(self.Yc).size == self.Yc.size, "Yc has duplicates {0}".format(self.Yc)
            assert np.unique(self.Zc).size == self.Zc.size, "Zc has duplicates {0}".format(self.Zc)

            print('nx = ', self.Xc.size-1, 'ny = ', self.Yc.size-1, 'nz = ', self.Zc.size-1)
            print('self.rsv_top', self.rsv_top)
            print('self.rsv_bottom', self.rsv_bottom)
            print('self.rsv_xy', self.rsv_xy)
            #print('Zc', self.Zc)

            from gen_msh import generate_box_3d
            generate_box_3d(X=2000, Y=2000, Z=4000, NX=21, NY=21, NZ=21, tags=idata.mesh.tags,  # XYZ are ignored since Xc, Yc, Zc are passed
                            is_transfinite=True, is_recombine=True, Xc=self.Xc, Yc=self.Yc, Zc=self.Zc,
                            filename=self.mesh_filename)# msh_ver=4.1)
            self.timer.node["initialization"].node["mesh_generation"].stop()
            print('Mesh generation finished')

        print('Mesh reading...')
        self.timer.node["initialization"].node["mesh_reading"] = timer_node()
        self.timer.node["initialization"].node["mesh_reading"].start()
        self.mesh_data = meshio.read(self.mesh_filename)
        self.timer.node["initialization"].node["mesh_reading"].stop()
        print('Mesh reading finished')

        print('Init reservoir (incl. mesh processing)...', flush=True)
        #self.set_uniform_initial_conditions(idata=idata)
        self.u_init = [0., 0., 0.]  # [m]
        self.p_init = None
        self.z_init = None
        self.set_boundary_conditions(idata=idata)

        # hash the mesh around its read, so the cache key describes the content gmsh actually read
        self.mesh_sha256 = file_sha256(self.mesh_filename) if self.cache_discretization else None
        self.timer.node["initialization"].node["init_mech_discretizer"] = timer_node()
        self.timer.node["initialization"].node["init_mech_discretizer"].start()
        self.init_mech_discretizer(idata=idata)
        self.timer.node["initialization"].node["init_mech_discretizer"].stop()
        if self.mesh_sha256 is not None and file_sha256(self.mesh_filename) != self.mesh_sha256:
            print('[WARN] %s changed while it was read, the discretization is not cached' % self.mesh_filename)
            self.cache_discretization = False

        self.grav = 9.80665e-5
        self.init_gravity(gravity_on=True, gravity_coeff=self.grav)
        #self.init_gravity(gravity_on=False)

        self.depths = np.array([c.values[2] for c in self.centroids])
        # specify initial temperature and pressure #TODO get it from model.set_initial_conditions()
        self.p_init = self.get_reservoir_initial_pressure(self.depths[:self.n_matrix])
        if self.thermoporoelasticity: # specify initial temperature
            self.t_init = self.get_reservoir_initial_temperature(self.depths[:self.n_matrix])

        if uniform_props:
            self.init_uniform_properties(idata=idata)
        elif idata.other.set_props_by_tags:
            # per-tag rock properties: the parent's set_props_tags builds self.props from
            # idata.rock arrays (one value per matrix tag), and init_heterogeneous_properties_by_tags
            # applies them per cell using self.tags (the parent's init_heterogeneous_properties
            # without its Python loop over cells). This model specifies permeability as per-tag
            # permx/permy/permz (leaving the 'perm' tensor unset), which the base handles.
            super().set_props_tags(idata=idata, matrix_tags=idata.mesh.matrix_tags)
            self.init_heterogeneous_properties_by_tags()
        else:  # don't use mesh tags, set by interpolation
            self.set_heterogeneous_props_by_interpolation(
                idata=idata, generate_mesh=generate_structured_mesh
            )
            self.init_heterogeneous_properties(idata=idata)

        # per-cell biot array used by write_to_vtk (eff_stress = tot_stress - biot * pressure).
        # idata.rock.biot may be a scalar (homogeneous) or a per-tag array (e.g. case_4);
        # expand the per-tag case to one value per matrix cell so it broadcasts against pressure.
        if np.isscalar(idata.rock.biot):
            self.biot_cell = idata.rock.biot
        else:
            self.biot_cell = np.array([self.props[tag]['biot'] for tag in self.tags[:self.n_matrix]])

        self.init_arrays_boundary_condition()
        self.update_boundary_conditions()
        print('Init reservoir finished')

        # Discretization
        print('Discretization (trans calc, etc) ...')
        self.timer.node["initialization"].node["discretization"] = timer_node()
        self.timer.node["initialization"].node["discretization"].start()
        # The pressure(/temperature) gradients are cheap and the engine reads them
        # (eval_stresses_and_velocities), so they are always reconstructed; the
        # expensive steps below come from the cache when it is up to date.
        if self.thermoporoelasticity:
            self.discr.reconstruct_pressure_temperature_gradients_per_cell(self.cpp_flow, self.cpp_heat)
        else:
            self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_flow)
        cache_file = self.discretization_cache_file(idata, uniform_props) if self.cache_discretization else None
        if cache_file is None or not self.load_discretization(cache_file):
            self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
            self.discr.calc_interface_approximations()
            self.discr.calc_cell_centered_stress_velocity_approximations()
            if cache_file is not None:
                self.save_discretization(cache_file)
        elif hasattr(self.discr, 'check_displacement_diagonal'):  # older darts builds lack it
            # calc_interface_approximations runs this check; a cache hit skips it
            self.discr.check_displacement_diagonal()
        self.timer.node["initialization"].node["discretization"].stop()
        print('Discretization finished')

    # ------------------------------------------------------------------
    # Discretization cache
    # ------------------------------------------------------------------
    def discretization_cache_arrays(self):
        """
        The discretizer arrays used after the discretization: by
        conn_mesh.init_p(m)e_mech_discretizer in init_reservoir_main and by the
        engine in eval_stresses_and_velocities (with discr.p_grads, discr.biots).
        """
        names = ['cell_m', 'cell_p', 'flux_stencil', 'flux_offset', 'hooke', 'hooke_rhs',
                 'biot_traction', 'biot_traction_rhs', 'darcy', 'darcy_rhs',
                 'biot_vol_strain', 'biot_vol_strain_rhs', 'stress_approx', 'velocity_approx']
        if self.thermoporoelasticity:
            names += ['thermal_traction', 'fourier']
        return names

    def discretization_cache_key(self, idata, uniform_props):
        """
        What the discretization depends on: mesh file content (as read), domain tags,
        boundary-condition coefficients, the per-cell discretizer inputs (perms, biots,
        stiffness, heat conductions, thermal expansions) as set by the Python property
        code, rock input and its mapping to cells, gravity, discretizer type and flags,
        and the discretizer build. The rock input is hashed as a whole (all of
        idata.rock), so changing any rock property invalidates the cache, including
        those the discretizer ignores.
        """
        import darts.discretizer
        h_bc = hashlib.sha256()
        bcs = [self.cpp_bc.flow, self.cpp_bc.mech_normal, self.cpp_bc.mech_tangen]
        if self.thermoporoelasticity:
            bcs.append(self.cpp_bc.thermal)
        for bc in bcs:
            hash_update(h_bc, np.asarray(bc.a))
            hash_update(h_bc, np.asarray(bc.b))
        h_rock = hashlib.sha256()
        hash_update(h_rock, vars(idata.rock))
        hash_update(h_rock, list(idata.mesh.matrix_tags))  # per-tag rock arrays follow this order
        hash_update(h_rock, [bool(uniform_props), bool(getattr(idata.other, 'set_props_by_tags', False))])
        hash_update(h_rock, self.tags)
        # the per-cell discretizer inputs themselves: Python code (init_heterogeneous_properties_by_tags,
        # get_lambda_mu, set_props_tags, ...) derives them from idata.rock, and the rest of the key does not cover it
        for name in ['perms', 'biots', 'stfs'] + (['heat_conductions', 'thermal_expansions']
                                                  if self.thermoporoelasticity else []):
            hash_update(h_rock, np.array([v.values for v in getattr(self.discr, name)]))
        build_info_file = os.path.join(os.path.dirname(darts.discretizer.__file__), 'build_info.txt')
        build_info = ''
        if os.path.exists(build_info_file):
            with open(build_info_file) as f:
                build_info = f.read().strip()
        return {'format': DISCR_CACHE_FORMAT,
                'mesh_sha256': self.mesh_sha256,
                'domain_tags': {str(k): sorted(int(t) for t in v) for k, v in self.domain_tags.items()},
                'discretizer': type(self.discr).__name__,
                'neumann_boundaries_grad_reconstruction': bool(self.discr.neumann_boundaries_grad_reconstruction),
                'gradients_extended_stencil': bool(self.discr.gradients_extended_stencil),
                'gravity': [float(g) for g in self.discr.grav_vec.values],
                'boundary_conditions_sha256': h_bc.hexdigest(),
                'rock_sha256': h_rock.hexdigest(),
                'darts_discretizer_sha256': file_sha256(darts.discretizer.__file__),
                'darts_build_info': build_info}

    def discretization_cache_file(self, idata, uniform_props):
        self.discr_cache_key = self.discretization_cache_key(idata, uniform_props)
        digest = hashlib.sha256(json.dumps(self.discr_cache_key, sort_keys=True).encode()).hexdigest()
        # .cache/ is git-ignored
        return os.path.join(os.path.dirname(self.mesh_filename), '.cache', 'discretization_%s.npz' % digest[:16])

    def load_discretization(self, cache_file):
        """
        Restore the discretizer arrays from cache_file.
        :return: True on success; False (nothing restored) when the file is missing or unusable
        """
        from darts.discretizer import index_vector as disc_index_vector, value_vector as disc_value_vector
        if not os.path.exists(cache_file):
            print('No discretization cache yet, it will be written to', cache_file)
            return False
        t0 = time.time()
        names = self.discretization_cache_arrays()
        int_names = ('cell_m', 'cell_p', 'flux_stencil', 'flux_offset')
        try:
            with np.load(cache_file) as data:
                meta = json.loads(str(data['meta']))
                if meta['key'] != self.discr_cache_key or meta['n_conns'] != len(self.adj_matrix):
                    raise ValueError('it belongs to another input')
                for name in names:
                    vec_type = disc_index_vector if name in int_names else disc_value_vector
                    setattr(self.discr, name, vec_type(data[name]))
            if len(self.discr.cell_m) != len(self.adj_matrix) or \
                    self.discr.flux_offset[len(self.discr.flux_offset) - 1] != len(self.discr.flux_stencil):
                raise ValueError('inconsistent array sizes')
        except Exception as e:  # truncated, corrupt or foreign file
            print('[WARN] discretization cache %s is unusable (%s), discretizing' % (cache_file, e))
            for name in names:  # the discretizer appends to some of these arrays
                setattr(self.discr, name, disc_index_vector() if name in int_names else disc_value_vector())
            return False
        try:
            os.utime(cache_file)  # mark as recently used, see save_discretization
        except OSError:
            pass
        print('Discretization loaded from cache %s (%.1f s)' % (cache_file, time.time() - t0))
        return True

    def save_discretization(self, cache_file):
        t0 = time.time()
        folder = os.path.dirname(cache_file)
        tmp_file = '%s.tmp%d' % (cache_file, os.getpid())
        try:
            os.makedirs(folder, exist_ok=True)
            # zero-copy views of the C++ arrays
            arrays = {name: np.asarray(getattr(self.discr, name)) for name in self.discretization_cache_arrays()}
            meta = json.dumps({'key': self.discr_cache_key, 'n_conns': len(self.adj_matrix),
                               'mesh_file': self.mesh_filename})
            with open(tmp_file, 'wb') as f:
                np.savez(f, meta=np.array(meta), **arrays)
            os.replace(tmp_file, cache_file)  # a reader never sees a partial file
        except Exception as e:  # read-only folder, full disk, ...
            print('[WARN] discretization cache is not saved (%s)' % e)
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            return
        print('Discretization cached in %s (%.1f s, %.0f MB)'
              % (cache_file, time.time() - t0, os.path.getsize(cache_file) / 1e6))

        # keep the most recently used caches only, and drop leftovers of killed runs
        def mtime(fn):
            try:
                return os.path.getmtime(fn)
            except OSError:
                return 0.0
        caches = sorted(glob.glob(os.path.join(folder, 'discretization_*.npz')), key=mtime, reverse=True)
        leftovers = [fn for fn in glob.glob(os.path.join(folder, 'discretization_*.npz.tmp*'))
                     if time.time() - mtime(fn) > 3600]
        for fn in caches[DISCR_CACHE_KEEP:] + leftovers:
            if fn != cache_file:
                try:
                    os.remove(fn)
                    print('Removed old discretization cache', fn)
                except OSError:
                    pass

    def set_boundary_conditions(self, idata: InputData):
        self.boundary_conditions = {}
        self.boundary_conditions[idata.mesh.bnd_tags['BND_X-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[idata.mesh.bnd_tags['BND_X+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[idata.mesh.bnd_tags['BND_Y-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[idata.mesh.bnd_tags['BND_Y+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        if True:  # free Z-
            self.boundary_conditions[idata.mesh.bnd_tags['BND_Z-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.FREE }
            self.boundary_conditions[idata.mesh.bnd_tags['BND_Z+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.STUCK(0.,0.)}
        else:     # free Z+
            self.boundary_conditions[idata.mesh.bnd_tags['BND_Z-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
            self.boundary_conditions[idata.mesh.bnd_tags['BND_Z+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.FREE}

        if self.thermoporoelasticity:
            for key, bc in self.boundary_conditions.items():
                bc['temp'] = self.bc_type.AQUIFER(0.0)

    def update_boundary_conditions(self):
        return
        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]
            # flow
            self.bc_rhs[self.n_bc_vars * ids + self.p_bc_var] = bc['flow']['r']
            # energy
            if self.thermoporoelasticity:
                # keep initial temperature at the boundary
                self.bc_rhs[self.n_bc_vars * ids + self.t_bc_var] = \
                    self.get_reservoir_initial_temperature(self.depths[ids + self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]])
            # mechanics
            for id in ids:
                assert (self.adj_matrix_cols[self.id_sorted[id]] == id +
                        self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0])
                conn = self.conns[self.id_boundary_conns[id]]
                n = np.array(conn.n.values)
                conn_c = np.array(conn.c.values)
                c1 = np.array(self.centroids[conn.elem_id1].values)
                if n.dot(conn_c - c1) < 0: n *= -1.0
                self.bc_rhs[self.n_bc_vars * id + self.u_bc_var:self.n_bc_vars * id + self.u_bc_var + self.n_dim] = \
                    bc['mech']['rn'] * n + bc['mech']['rt']

    def init_heterogeneous_properties_by_tags(self):
        """
        UnstructReservoirMech.init_heterogeneous_properties (mech discretizer) with the
        same result, but the properties are evaluated once per tag and the per-cell
        discretizer arrays are filled from those, instead of a Python loop over cells.
        """
        m0, m1 = self.discr_mesh.region_ranges[elem_loc.MATRIX]
        tags, tag_ids = np.unique(self.tags[m0:m1], return_inverse=True)
        perms, biots, stfs, rconds, th_expns = [], [], [], [], []
        poro, cs, hcap = np.zeros(len(tags)), np.zeros(len(tags)), np.zeros(len(tags))
        for i, tag in enumerate(tags):
            p = self.props[tag]
            kx, ky, kz = (p['perm'],) * 3 if 'perm' in p else (p['permx'], p['permy'], p['permz'])
            lam, mu = get_lambda_mu(p['E'], p['nu'])
            perms.append(disc_matrix33(kx, ky, kz))
            biots.append(disc_matrix33(p['biot']))
            stfs.append(disc_stiffness(lam, mu))
            if self.thermoporoelasticity:
                rconds.append(disc_matrix33(p['thermal_conductivity']))
                th_expns.append(disc_matrix33(p['th_expn']))
                hcap[i] = p['heat_capacity']
            poro[i] = p['porosity']
            cs[i] = get_rock_compressibility(kd=p['kd'], biot=p['biot'], poro0=p['porosity'])
        self.discr.perms = vector_matrix33([perms[i] for i in tag_ids])
        self.discr.biots = vector_matrix33([biots[i] for i in tag_ids])
        self.discr.stfs = stf_vector([stfs[i] for i in tag_ids])
        if self.thermoporoelasticity:
            self.discr.heat_conductions = vector_matrix33([rconds[i] for i in tag_ids])
            self.discr.thermal_expansions = vector_matrix33([th_expns[i] for i in tag_ids])
        self.porosity = np.zeros(self.n_matrix + self.n_fracs)
        self.cs = np.zeros(self.n_matrix + self.n_fracs)
        self.hcap = np.zeros(self.n_matrix + self.n_fracs)
        self.porosity[m0:m1] = poro[tag_ids]
        self.cs[m0:m1] = cs[tag_ids]
        self.hcap[m0:m1] = hcap[tag_ids]

    def init_heterogeneous_properties(self, idata: InputData):
        '''
        set matrix properties using InputData
        :return:
        '''
        self.porosity = idata.rock.porosity
        lam, mu = get_lambda_mu(E=idata.rock.E, nu=idata.rock.nu)
        self.cs = idata.rock.compressibility

        self.hcap = np.zeros(self.n_matrix + self.n_fracs)
        for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                          self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
            #permx = idata.rock.permx[3 * cell_id]
            #permy = idata.rock.permy[3 * cell_id + 1]
            #permz = idata.rock.permz[3 * cell_id + 2]
            permx = idata.rock.permx[cell_id]
            permy = idata.rock.permy[cell_id]
            permz = idata.rock.permz[cell_id]
            self.discr.perms.append(disc_matrix33(permx, permy, permz))
            self.discr.biots.append(disc_matrix33(idata.rock.biot))
            self.discr.stfs.append(disc_stiffness(lam[cell_id], mu[cell_id]))
            if self.thermoporoelasticity:
                self.discr.heat_conductions.append(disc_matrix33(idata.rock.thermal_conductivity))
                self.discr.thermal_expansions.append(disc_matrix33(idata.rock.th_expn))#[cell_id]))

    def write_to_vtk(self, output_directory, ith_step, engine, viscosity=None):
        """
        Class method which writes output of unstructured grid to VTK format
        :param output_directory: directory of output files
        :param property_array: np.array containing all cell properties (N_cells x N_prop)
        :param cell_property: list with property names (visible in ParaView (format strings)
        :param ith_step: integer containing the output step
        :param viscosity: optional per-(reservoir-block) viscosity array [cP] computed from the
                          current state via darts output property interpolator. If None, the
                          constant idata.fluid.viscosity is written instead.
        :return:
        """
        # First check if output directory already exists:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Allocate empty new cell_data dictionary:
        property_array = np.array(engine.X, copy=False)
        props_num = self.n_vars
        available_matrix_geometries_cpp = [elem_type.HEX, elem_type.PRISM, elem_type.TETRA, elem_type.PYRAMID]
        available_fracture_geometries_cpp = [elem_type.QUAD, elem_type.TRI]
        available_matrix_geometries = {'hexahedron': elem_type.HEX,
                                       'wedge': elem_type.PRISM,
                                       'tetra': elem_type.TETRA,
                                       'pyramid': elem_type.PYRAMID}
        available_fracture_geometries = ['quad', 'triangle']

        # Stresses and velocities
        engine.eval_stresses_and_velocities()
        total_stresses = -np.array(engine.total_stresses, copy=False)# make positive for compressive stresses

        if not hasattr(self, 'displs_initial'):
            self.displs_initial = dict()
        if not hasattr(self, 'tot_stress_initial'):
            self.tot_stress_initial = total_stresses.copy()

        # Matrix
        cells = []
        cell_data = {}
        for cell_block in self.mesh_data.cells:
            if cell_block.type in available_matrix_geometries:
                cells.append(cell_block)
                cell_ids = np.array(self.discr_mesh.elem_type_map[available_matrix_geometries[cell_block.type]], dtype=np.int64)
                for i in range(props_num):
                    if self.cell_property[i] in ['ux', 'uy', 'uz']:
                        if self.cell_property[i] not in self.displs_initial:
                            self.displs_initial[self.cell_property[i]] = property_array[props_num * cell_ids + i]
                    if self.cell_property[i] == 'pressure':
                        pressure = property_array[props_num * cell_ids + i]
                        if not hasattr(self, 'pressure_initial') :
                            self.pressure_initial = property_array[props_num * cell_ids + i].copy()
                    if self.cell_property[i] == 'temperature':
                        temperature = property_array[props_num * cell_ids + i]
                        if not hasattr(self, 'temperature_initial'):
                            self.temperature_initial = property_array[props_num * cell_ids + i].copy()

                    if self.cell_property[i] not in cell_data: cell_data[self.cell_property[i]] = []
                    if self.cell_property[i] in ['ux', 'uy', 'uz']:  # eliminate displacements got after the initialization stage (equilibration)
                        cell_data[self.cell_property[i]].append(property_array[props_num * cell_ids + i] - self.displs_initial[self.cell_property[i]])
                    else:
                        cell_data[self.cell_property[i]].append(property_array[props_num * cell_ids + i])

                if 'tot_stress' not in cell_data: cell_data['tot_stress'] = []
                cell_data['tot_stress'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                for j in range(6):
                    cell_data['tot_stress'][-1][:, j] = total_stresses[j::6]

                # Terzaghi/Biot: sigma'_ij = sigma_ij - biot * p * delta_ij, so the pore
                # pressure comes off the NORMAL components only - shear stress is
                # unaffected by pore pressure. Components are ordered
                # xx, yy, zz, xy, xz, yz, so the first three are the normal ones.
                # This used to subtract biot * p from all six and wrap every component
                # in np.fabs(), which discarded the sign of the whole tensor and left
                # the shear slots holding -biot * p instead of a shear stress.
                if 'eff_stress' not in cell_data: cell_data['eff_stress'] = []
                cell_data['eff_stress'].append(cell_data['tot_stress'][-1].copy())
                biot_pressure = self.biot_cell * pressure
                for j in range(3):
                    cell_data['eff_stress'][-1][:, j] -= biot_pressure

                if 'delta_tot_stress' not in cell_data: cell_data['delta_tot_stress'] = []
                cell_data['delta_tot_stress'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                for j in range(6):
                    cell_data['delta_tot_stress'][-1][:, j] = total_stresses[j::6] - self.tot_stress_initial[j::6]

                delta_pressure = pressure - self.pressure_initial
                if 'delta_pressure' not in cell_data: cell_data['delta_pressure'] = []
                cell_data['delta_pressure'].append(np.zeros(self.n_matrix, dtype=np.float64))
                cell_data['delta_pressure'][-1][:] = delta_pressure

                # same rule as eff_stress above: the normal components only
                if 'delta_eff_stress' not in cell_data: cell_data['delta_eff_stress'] = []
                cell_data['delta_eff_stress'].append(cell_data['delta_tot_stress'][-1].copy())
                biot_delta_pressure = self.biot_cell * delta_pressure
                for j in range(3):
                    cell_data['delta_eff_stress'][-1][:, j] -= biot_delta_pressure

                if hasattr(self, 'temperature_initial'): # if thermal simulation
                    if 'delta_temperature' not in cell_data: cell_data['delta_temperature'] = []
                    cell_data['delta_temperature'].append(np.zeros(self.n_matrix, dtype=np.float64))
                    cell_data['delta_temperature'][-1][:] = temperature - self.temperature_initial

                if True:#ith_step == 0:
                    if 'perm' not in cell_data: cell_data['perm'] = []
                    if 'E' not in cell_data: cell_data['E'] = []
                    if 'poisson' not in cell_data: cell_data['poisson'] = []
                    if 'poro' not in cell_data: cell_data['poro'] = []
                    cell_data['perm'].append(np.zeros((len(cell_ids), 9), dtype=np.float64))
                    cell_data['E'].append(np.zeros(len(cell_ids), dtype=np.float64))
                    cell_data['poisson'].append(np.zeros(len(cell_ids), dtype=np.float64))
                    cell_data['poro'].append(np.array(self.mesh.poro, copy=False)[:self.n_matrix])
                    if 'viscosity' not in cell_data: cell_data['viscosity'] = []
                    if viscosity is not None:
                        # state-dependent viscosity computed via darts output property interpolator;
                        # 'viscosity' is per reservoir block, so index it by cell_ids to align with
                        # this geometry group's cells (same indexing as pressure/temperature above).
                        cell_data['viscosity'].append(np.asarray(viscosity)[cell_ids])
                    else:
                        # fallback for the initial frame (written by THMCModel.reinit before the
                        # output property interpolator exists): constant fluid viscosity.
                        cell_data['viscosity'].append(np.full(len(cell_ids), self.idata.fluid.viscosity))
                    for i, cell_id in enumerate(cell_ids):
                        cell_data['perm'][-1][i] = np.array(self.discr.perms[cell_id].values)
                        stf = np.array(self.discr.stfs[cell_id].values)
                        la = stf[1]
                        mu = (stf[0] - la) / 2
                        E = mu * (3 * la + 2 * mu) / (la + mu)
                        poisson = la / (2 * (la + mu))
                        cell_data['E'][-1][i] = E
                        cell_data['poisson'][-1][i] = poisson

                # compute strain from stress and geomech props
                # https://en.wikipedia.org/wiki/Hooke%27s_law, In matrix form, Hooke's law for isotropic materials can be written as
                if 'strain' not in cell_data: cell_data['strain'] = []
                cell_data['strain'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                stress = cell_data['delta_eff_stress'][-1]
                E = cell_data['E'][-1]
                poisson = cell_data['poisson'][-1]
                cell_data['strain'][-1][:, 0] = -(stress[:, 0] - poisson * (stress[:, 1] + stress[:, 2])) / E
                cell_data['strain'][-1][:, 1] = -(stress[:, 1] - poisson * (stress[:, 0] + stress[:, 2])) / E
                cell_data['strain'][-1][:, 2] = -(stress[:, 2] - poisson * (stress[:, 0] + stress[:, 1])) / E
                for k in range(3,6):  # shear part
                    cell_data['strain'][-1][:, k] = -(2 + 2 * poisson) * stress[:, k]/E

        # Store solution for each time-step:
        mesh = meshio.Mesh(
            self.mesh_data.points,
            cells,
            cell_data=cell_data)
        meshio.write("{:s}/solution{:d}.vtu".format(output_directory, ith_step), mesh)

        time = engine.t if ith_step > 0 else 0.0
        self.write_pvd_file(ith_step, time, output_directory)

        # fault surface with tractions for the FSP post-processing (see fault.py)
        self.save_fault_traction(output_directory, ith_step, engine, time)

        return 0

    # ------------------------------------------------------------------
    # Fault surface with tractions for the FSP post-processing (fault.py)
    # ------------------------------------------------------------------
    def _setup_fault_mapping(self, fault_mesh_filename):
        """
        One-time setup: read the fault surface from the *_fault.msh companion mesh
        (same geometry as mesh.msh plus the FAULT physical group) and map every
        fault face to the engine connection lying on it. Fault faces must be faces
        of the simulation mesh; they are matched to the discretizer connections by
        their centroids.
        """
        self._fault_ok = False
        if not os.path.exists(fault_mesh_filename):
            print('[INFO] no fault mesh %s: fault tractions are not saved' % fault_mesh_filename)
            return

        from scipy.spatial import cKDTree
        from fault import read_fault_mesh

        fault = read_fault_mesh(fault_mesh_filename)

        # matrix-matrix interfaces of the discretizer mesh
        conns = [c for c in self.discr_mesh.conns if c.elem_id1 < self.n_matrix and c.elem_id2 < self.n_matrix]
        dist, k = cKDTree(np.array([c.c.values for c in conns])).query(fault['centers'])
        conns = [conns[i] for i in k]
        conn_n = np.array([c.n.values for c in conns])
        matched = (dist < 0.1 * np.sqrt(fault['areas'])) & (np.abs(np.einsum('ij,ij->i', conn_n, fault['normals'])) > 0.99)
        if not matched.all():
            print('[WARN] %d of %d fault faces are not faces of the simulation mesh: fault tractions are not saved'
                  % ((~matched).sum(), matched.size))
            return

        cells = np.array([[c.elem_id1, c.elem_id2] for c in conns], dtype=np.int64)

        # engine connection directed elem_id1 -> elem_id2 (forces are stored per directed connection)
        block_m = np.array(self.mesh.block_m, dtype=np.int64)
        block_p = np.array(self.mesh.block_p, dtype=np.int64)
        n_ids = max(block_m.max(), block_p.max()) + 1
        keys = block_m * n_ids + block_p
        order = np.argsort(keys)
        face_keys = cells[:, 0] * n_ids + cells[:, 1]
        pos = np.minimum(np.searchsorted(keys[order], face_keys), len(keys) - 1)
        assert np.all(keys[order[pos]] == face_keys), 'fault faces are not found among the engine connections'

        # The engine force of connection (i, j) is -(sigma . n_out) * area, n_out being the
        # outward normal of cell i. Orient it along the fault-face normal and make it
        # compression positive: traction = sign(n_out . normal) * force / area.
        c1 = np.array([self.discr_mesh.centroids[i].values for i in cells[:, 0]])
        n_out = conn_n * np.sign(np.einsum('ij,ij->i', np.array([c.c.values for c in conns]) - c1, conn_n))[:, None]
        sign = np.sign(np.einsum('ij,ij->i', n_out, fault['normals']))

        self._fault = fault
        self._fault_conn = order[pos]
        self._fault_cells = cells
        self._fault_scale = (sign / np.array([c.area for c in conns]))[:, None]
        self._fault_pvd = {}
        self._fault_ok = True
        print('[INFO] fault tractions are saved for %d fault faces' % matched.size)

    def save_fault_traction(self, output_directory, ith_step, engine, time):
        """
        Write <output_directory>/fault<ith_step>.vtu: the fault surface with cell data
          traction (3) total-stress traction [bar], compression positive
          normal   (3) unit fault normal
          pressure     pore pressure, mean of both sides of the fault [bar]
          temperature  temperature, mean of both sides of the fault [K] (thermal runs only)
        and update <output_directory>/fault.pvd.
        """
        if not hasattr(self, '_fault_ok'):
            base, ext = os.path.splitext(self.mesh_filename)
            self._setup_fault_mapping(base + '_fault' + ext)
        if not self._fault_ok:
            return

        force = np.zeros((self._fault_conn.size, 3))
        for name in ['hooke_forces', 'biot_forces', 'thermal_forces']:
            f = np.array(getattr(engine, name), copy=False)
            if f.size:  # thermal_forces is empty for isothermal engines
                force += f.reshape(-1, 3)[self._fault_conn]
        traction = self._fault_scale * force

        X = np.array(engine.X, copy=False)
        p_id = self.cell_property.index('pressure')
        pressure = X[self.n_vars * self._fault_cells + p_id].mean(axis=1)

        from fault import split_by_blocks
        cells = self._fault['cells']
        cell_data = {'traction': split_by_blocks(traction, cells),
                     'normal': split_by_blocks(self._fault['normals'], cells),
                     'pressure': split_by_blocks(pressure, cells)}
        if 'temperature' in self.cell_property:
            t_id = self.cell_property.index('temperature')
            cell_data['temperature'] = split_by_blocks(X[self.n_vars * self._fault_cells + t_id].mean(axis=1), cells)
        meshio.write(os.path.join(output_directory, 'fault%d.vtu' % ith_step),
                     meshio.Mesh(self._fault['points'], cells, cell_data=cell_data))

        self._fault_pvd[ith_step] = time
        with open(os.path.join(output_directory, 'fault.pvd'), 'w') as f:
            f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n  <Collection>\n')
            for step, t in sorted(self._fault_pvd.items()):
                f.write('    <DataSet timestep="%s" file="fault%d.vtu"/>\n' % (t, step))
            f.write('  </Collection>\n</VTKFile>\n')

    def set_heterogeneous_props_by_interpolation(self, idata, generate_mesh):
        # set different values in the reservoir and lateral surrounding+over/under-burden
        # first, create a struct grid to easily set heterogeneous rock properties
        # second, interpolate them to unstructured mesh used for computation
        if generate_mesh:
            self.nx, self.ny, self.nz  = idata.other.nx, idata.other.ny, idata.other.nz

            # fill the whole array with non-rsv values, the rsv part will be replaced later on
            porosity_struct = np.zeros(self.nz * self.ny * self.nx) + idata.rock.poro_non_rsv
            permeability_struct = np.zeros(self.nz * self.ny * self.nx) + idata.rock.perm_non_rsv # mD
            E_struct = np.zeros(self.nz * self.ny * self.nx) + idata.rock.E_non_rsv # [bars]

            centers = np.array([np.array(c.values) for c in self.centroids[:self.n_matrix]])
            x = centers[:, 0]
            y = centers[:, 1]
            z = centers[:, 2]

            # rsv cell centers
            xs = (self.Xc[1:] + self.Xc[:-1]) * 0.5
            ys = (self.Yc[1:] + self.Yc[:-1]) * 0.5
            zs = (self.Zc[1:] + self.Zc[:-1]) * 0.5

            centers_struct_x, centers_struct_y, centers_struct_z = np.meshgrid(xs, ys, zs)
            centers_struct_x, centers_struct_y, centers_struct_z = centers_struct_x.flatten(), centers_struct_y.flatten(), centers_struct_z.flatten()

            rsv = reduce(np.logical_and, [self.rsv_top <= centers_struct_z, centers_struct_z <= self.rsv_bottom,
                                        self.rsv_y1 <= centers_struct_y,  centers_struct_y <= self.rsv_y2,
                                        self.rsv_x1 <= centers_struct_x,  centers_struct_x <= self.rsv_x2])

            # set juxtaposed rsv
            if False:
                rsv_thickness = np.fabs(self.rsv_bottom - self.rsv_top)
                self.rsv_z_middle_1 = self.rsv_top + rsv_thickness * 0.25
                self.rsv_z_middle_2 = self.rsv_top + rsv_thickness * 0.75
                self.rsv_x_middle = (self.rsv_x1 + self.rsv_x2) * 0.5
                rsv_left = reduce(np.logical_and, [self.rsv_z_middle_1 <= centers_struct_z, centers_struct_z <= self.rsv_bottom,
                                            self.rsv_y1 <= centers_struct_y,  centers_struct_y <= self.rsv_y2,
                                            self.rsv_x1 <= centers_struct_x,  centers_struct_x <= self.rsv_x_middle])
                rsv_right = reduce(np.logical_and, [self.rsv_top <= centers_struct_z, centers_struct_z <= self.rsv_z_middle_2,
                                            self.rsv_y1 <= centers_struct_y,  centers_struct_y <= self.rsv_y2,
                                            self.rsv_x_middle <= centers_struct_x,  centers_struct_x <= self.rsv_x2])
                rsv = reduce(np.logical_or, [rsv_left, rsv_right])

            porosity_struct[rsv] = idata.rock.porosity
            permeability_struct[rsv] = idata.rock.permx # [mD]
            E_struct[rsv] = idata.rock.E #[bars]

            porosity = np.zeros(self.nz * self.ny * self.nx)
            permeability = np.zeros(self.nz * self.ny * self.nx)
            E = np.zeros(self.nz * self.ny * self.nx)

            arrays = [porosity, permeability, E]
            arrays_struct = [porosity_struct, permeability_struct, E_struct]

            for arr, arr_struct in zip(arrays, arrays_struct):
                arr[:] = gd((centers_struct_x, centers_struct_y, centers_struct_z), arr_struct, (x, y, z), method='nearest')


        else:
            centers = np.array([np.array(c.values) for c in self.centroids[:self.n_matrix]])
            z1, z2 = min(self.rsv_top, self.rsv_bottom), max(self.rsv_top, self.rsv_bottom)
            rsv = reduce(np.logical_and, [z1 <= centers[:, 2], centers[:, 2] <= z2,
                                        self.rsv_y1 <= centers[:, 1], centers[:, 1] <= self.rsv_y2,
                                        self.rsv_x1 <= centers[:, 0], centers[:, 0] <= self.rsv_x2])

            porosity = np.full(self.n_matrix, idata.rock.poro_non_rsv)
            permeability = np.full(self.n_matrix, idata.rock.perm_non_rsv) # mD
            E = np.full(self.n_matrix, idata.rock.E_non_rsv) # [bars]

            porosity[rsv] = idata.rock.porosity
            permeability[rsv] = idata.rock.permx # [mD]
            E[rsv] = idata.rock.E #[bars]

        idata.rock.porosity = porosity
        idata.rock.permx = idata.rock.permy = idata.rock.permz = permeability
        idata.rock.E = E  # bars

    def decouple_geomech(self):
        '''
        turns off mechanics->porosity (so pressure and flow) influence
        :return:
        '''
        vol_strain_tran = np.array(self.mesh.vol_strain_tran, copy=False)
        vol_strain_rhs = np.array(self.mesh.vol_strain_rhs, copy=False)
        vol_strain_tran[:] = 0.0
        vol_strain_rhs[:] = 0.0

    def create_vtk_wells(self, output_directory: str, prolongation=-3000, tube_radius=20, dz=10):
        '''
        creates a file wells.vtk with a tube per well based on its first perforation
        :param output_directory:
        :return:
        '''

        try:
            import vtk
        except ModuleNotFoundError:
            import subprocess
            import sys

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "vtk"]
            )
            import vtk
        well_vtk_filename = os.path.join(output_directory, 'wells.vtk')
        # Append multiple cylinders into one polydata
        appendFilter = vtk.vtkAppendPolyData()

        def create_tube(center, prolongation, tube_radius):
            # Create points for the polyline
            points = vtk.vtkPoints()
            points.InsertNextPoint(center[0], center[1], -center[2] + prolongation)  # Point 1
            points.InsertNextPoint(center[0], center[1], -center[2])  # Point 2

            # Create a polyline that connects the points
            lines = vtk.vtkCellArray()
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(2)  # Number of points
            line.GetPointIds().SetId(0, 0)
            line.GetPointIds().SetId(1, 1)
            lines.InsertNextCell(line)

            # Create a polydata to hold the points and the polyline
            polyData = vtk.vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetLines(lines)

            # Apply vtkTubeFilter to create a tube around the polyline
            tubeFilter = vtk.vtkTubeFilter()
            tubeFilter.SetInputData(polyData)
            tubeFilter.SetRadius(tube_radius)  # Tube radius
            tubeFilter.SetNumberOfSides(50)  # Smoothness of the tube
            tubeFilter.Update()

            return tubeFilter.GetOutput()

        for w in self.wells:
            is_first = True
            for p in w.perforations:
                well_block, res_block_local, well_index, well_indexD = p
                c = np.array(self.centroids[res_block_local].values, copy=True)
                c[2] = -c[2]
                if is_first:
                    cyl = create_tube(c, prolongation=prolongation, tube_radius=tube_radius)
                    appendFilter.AddInputData(cyl)
                    is_first = False
                c[2] -= dz * 0.5
                cyl = create_tube(c, prolongation=dz, tube_radius=tube_radius * 2)
                appendFilter.AddInputData(cyl)
                #break  # use only the first perf

        # Update the append filter to combine the polydata
        appendFilter.Update()

        # Write the cylinders to a VTK file
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(well_vtk_filename)
        writer.SetInputConnection(appendFilter.GetOutputPort())
        writer.Write()
