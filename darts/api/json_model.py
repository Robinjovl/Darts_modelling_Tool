import re
from typing import Any

from darts.models.darts_model import DartsModel


class JsonModel(DartsModel):
    def set_reservoir(self):
        # Reservoir is attached by ModelBuilder before init()
        assert hasattr(self, 'reservoir') and self.reservoir is not None

    def set_physics(self):
        # Physics is attached by ModelBuilder before init()
        assert hasattr(self, 'physics') and self.physics is not None

    def set_wells(self, verbose: bool = False):
        wells_spec = getattr(self, '_wells_spec', None)
        if not wells_spec or not getattr(wells_spec, 'wells', None):
            return
        for well in wells_spec.wells:
            self.reservoir.add_well(well.name)
            for perf in well.perforations:
                i, j, k = perf.ijk
                well_radius = (
                    perf.well_radius if perf.well_radius is not None else 0.0762
                )
                self.reservoir.add_perforation(
                    well.name,
                    res_cell_idx=(i, j, k),
                    well_diameter=2.0 * well_radius,
                    skin=perf.skin if perf.skin is not None else 0.0,
                )

    def set_initial_conditions(self):
        ic_spec = getattr(self, '_initial_conditions_spec', None)
        if not ic_spec or not getattr(ic_spec, 'by_array', None):
            return
        # Normalize keys to engine-expected variable names (pressure and component variables)
        by_array = dict(ic_spec.by_array)
        norm: dict[str, Any] = {}
        # Build expected variable names from physics (e.g., ['pressure','CO2','C1'])
        expected_vars = list(self.physics.vars)
        expected_lower = {v.lower(): v for v in expected_vars}
        comp_vars = expected_vars[1:]
        # Map provided keys
        for k, v in by_array.items():
            kl = str(k).strip().lower()
            # Pressure synonyms
            if kl in ("p", "pressure"):
                norm_key = expected_lower.get('pressure', 'pressure')
                norm[norm_key] = v
                continue
            # z-indexed composition (z0,z_1, etc.) -> map to corresponding component var name
            m = re.fullmatch(r"z\s*_?(\d+)", kl)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < len(comp_vars):
                    norm_key = comp_vars[idx]
                    norm[norm_key] = v
                continue
            # Component name provided (case-insensitive)
            if kl in expected_lower:
                norm_key = expected_lower[kl]
                norm[norm_key] = v
                continue
            # Fallback: keep as-is
            norm[k] = v
        # Ensure all expected component variables exist; fill missing with 0.0
        for var in comp_vars:
            if var not in norm:
                norm[var] = 0.0
        # Ensure pressure exists; if not provided, leave it to physics defaults
        self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh, input_distribution=norm
        )

    def set_well_controls(self):
        wc_spec = getattr(self, '_well_controls_spec', None)
        # Top-level fallbacks (may be None if not provided in JSON)
        inj_bhp = getattr(wc_spec, 'inj_bhp', None) if wc_spec else None
        prod_bhp = getattr(wc_spec, 'prod_bhp', None) if wc_spec else None
        inj_comp = getattr(wc_spec, 'inj_composition', None) if wc_spec else None
        inj_temp = getattr(wc_spec, 'inj_temp', None) if wc_spec else None
        inj_rate = getattr(wc_spec, 'inj_rate', None) if wc_spec else None
        rate_type_str = getattr(wc_spec, 'rate_type', None) if wc_spec else None
        inj_phase = getattr(wc_spec, 'phase_name', None) if wc_spec else None

        from darts.engines import well_control_iface

        def _map_rate_type(name: str):
            if not name:
                return well_control_iface.MOLAR_RATE
            key = name.strip().upper()
            mapping = {
                'MOLAR_RATE': well_control_iface.MOLAR_RATE,
                'MASS_RATE': well_control_iface.MASS_RATE,
                'VOLUMETRIC_RATE': well_control_iface.VOLUMETRIC_RATE,
                'ADVECTIVE_HEAT_RATE': well_control_iface.ADVECTIVE_HEAT_RATE,
            }
            return mapping.get(key, well_control_iface.MOLAR_RATE)

        for _i, w in enumerate(self.reservoir.wells):
            # Per-well overrides from spec if provided
            well_spec = None
            wells_spec = getattr(self, '_wells_spec', None)
            if wells_spec and getattr(wells_spec, 'wells', None):
                for ws in wells_spec.wells:
                    if ws.name == w.name:
                        well_spec = ws
                        break
            per_well = getattr(well_spec, 'controls', None)
            inj_bhp_w = getattr(per_well, 'inj_bhp', None) if per_well else None
            prod_bhp_w = getattr(per_well, 'prod_bhp', None) if per_well else None
            inj_comp_w = (
                getattr(per_well, 'inj_composition', None) if per_well else None
            )
            inj_temp_w = getattr(per_well, 'inj_temp', None) if per_well else None
            inj_rate_w = getattr(per_well, 'inj_rate', None) if per_well else None
            rate_type_w = getattr(per_well, 'rate_type', None) if per_well else None
            inj_phase_w = getattr(per_well, 'phase_name', None) if per_well else None

            # Prefer explicit per-well injection controls
            if inj_rate_w is not None or inj_bhp_w is not None:
                if inj_rate_w is not None:
                    ctrl_type = _map_rate_type(rate_type_w)
                    self.physics.set_well_controls(
                        wctrl=w.control,
                        control_type=ctrl_type,
                        is_inj=True,
                        target=inj_rate_w,
                        phase_name=inj_phase_w,
                        inj_composition=inj_comp_w,
                        inj_temp=inj_temp_w,
                    )
                else:
                    self.physics.set_well_controls(
                        wctrl=w.control,
                        control_type=well_control_iface.BHP,
                        is_inj=True,
                        target=inj_bhp_w,
                        inj_composition=inj_comp_w,
                        inj_temp=inj_temp_w,
                    )
                continue

            # Next, allow top-level injection controls if per-well not provided
            if inj_rate is not None or inj_bhp is not None:
                if inj_rate is not None:
                    ctrl_type = _map_rate_type(rate_type_str)
                    self.physics.set_well_controls(
                        wctrl=w.control,
                        control_type=ctrl_type,
                        is_inj=True,
                        target=inj_rate,
                        phase_name=inj_phase,
                        inj_composition=inj_comp,
                        inj_temp=inj_temp,
                    )
                else:
                    self.physics.set_well_controls(
                        wctrl=w.control,
                        control_type=well_control_iface.BHP,
                        is_inj=True,
                        target=inj_bhp,
                        inj_composition=inj_comp,
                        inj_temp=inj_temp,
                    )
                continue

            # Producer controls per-well
            if prod_bhp_w is not None:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=prod_bhp_w,
                )
                continue

            # Producer controls from top-level if per-well not provided
            if prod_bhp is not None:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=prod_bhp,
                )
