"""
DartsModel subclass with lifecycle overrides for JSON-driven execution.

JsonModel extends the standard DartsModel so that reservoir, physics, and
well setup steps are no-ops — ModelBuilder has already configured them
before ``init()`` is called.  It also provides helpers for parsing simulation
runtime from the spec and collecting output file paths after a run completes.
"""

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
                i, j, k0 = perf.ijk
                k1 = perf.k_end if getattr(perf, 'k_end', None) is not None else k0
                if k1 < k0:
                    raise ValueError(
                        f"Invalid perforation interval for well {well.name}: k_end({k1}) < k({k0})"
                    )
                well_radius = (
                    perf.well_radius if perf.well_radius is not None else 0.0762
                )
                for k in range(k0, k1 + 1):
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

        def _well_role(name: str, per_well: Any) -> bool | None:
            if per_well:
                if (
                    getattr(per_well, 'inj_rate', None) is not None
                    or getattr(per_well, 'inj_bhp', None) is not None
                    or getattr(per_well, 'inj_composition', None) is not None
                ):
                    return True
                if getattr(per_well, 'prod_bhp', None) is not None:
                    return False
            upper = name.upper()
            if 'INJ' in upper:
                return True
            if 'PRD' in upper or 'PROD' in upper:
                return False
            return None

        def _ctrl(per_well: Any, attr: str) -> Any:
            """Resolve a well-control attribute: per-well value takes precedence over global."""
            val = getattr(per_well, attr, None) if per_well else None
            return val if val is not None else getattr(wc_spec, attr, None)

        wells_spec = getattr(self, '_wells_spec', None)
        well_cfg = {
            ws.name: getattr(ws, 'controls', None)
            for ws in getattr(wells_spec, 'wells', []) or []
        }

        for w in self.reservoir.wells:
            per_well = well_cfg.get(w.name)
            role = _well_role(w.name, per_well)

            inj_rate = _ctrl(per_well, 'inj_rate')
            inj_bhp = _ctrl(per_well, 'inj_bhp')
            inj_comp = _ctrl(per_well, 'inj_composition')
            inj_temp = _ctrl(per_well, 'inj_temp')
            inj_phase = _ctrl(per_well, 'phase_name')
            rate_type = _ctrl(per_well, 'rate_type')
            prod_bhp = _ctrl(per_well, 'prod_bhp')

            if role is True and (inj_rate is not None or inj_bhp is not None):
                if inj_rate is not None:
                    self.physics.set_well_controls(
                        wctrl=w.control,
                        control_type=_map_rate_type(rate_type),
                        is_inj=True,
                        target=inj_rate,
                        phase_name=inj_phase,
                        inj_composition=inj_comp,
                        inj_temp=inj_temp,
                    )
                    if inj_bhp is not None:
                        self.physics.set_well_controls(
                            wctrl=w.constraint,
                            control_type=well_control_iface.BHP,
                            is_inj=True,
                            target=inj_bhp,
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

            if role is False and prod_bhp is not None:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=prod_bhp,
                )
