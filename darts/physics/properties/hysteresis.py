from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class KilloughLandModel:
    swc: float
    sgrmax: float
    epsilon: float = 1e-12

    @property
    def land_constant(self) -> float:
        return 1.0 / self.sgrmax - 1.0 / (1.0 - self.swc)

    def residual_gas_saturation(self, sg_max: float) -> float:
        sg_max = float(np.clip(sg_max, 0.0, 1.0 - self.swc))
        return sg_max / (1.0 + self.land_constant * sg_max)

    def update_sg_max(self, sg: float, sg_max: float) -> float:
        sg = float(np.clip(sg, 0.0, 1.0))
        sg_max = float(np.clip(sg_max, 0.0, 1.0))
        if sg >= sg_max:
            return sg

        # sgr = self.residual_gas_saturation(sg_max)
        sgr = sg_max / 2
        if sg < sgr:
            return float(np.clip(sgr * 2, 0.0, 1.0))
        return sg_max


class _LookupTableMixin:
    @staticmethod
    def normalize_saturation_axis(axis: str | None) -> str | None:
        if axis is None:
            return None

        normalized = axis.strip().lower().replace("-", "_").replace(" ", "_")
        alias_map = {
            "sw": "wetting",
            "sg": "nonwetting",
            "wetting": "wetting",
            "nonwetting": "nonwetting",
            "wetting_saturation": "wetting",
            "nonwetting_saturation": "nonwetting",
        }
        if normalized not in alias_map:
            raise ValueError(f"Unsupported saturation axis label '{axis}'")
        return alias_map[normalized]

    @classmethod
    def parse_section_axis_metadata(cls, line: str) -> str | None:
        stripped = line.lstrip("#").strip()
        if ":" in stripped:
            key, value = stripped.split(":", 1)
        elif "=" in stripped:
            key, value = stripped.split("=", 1)
        else:
            return None

        key = key.strip().lower().replace("-", "_").replace(" ", "_")
        if key not in {"x_axis", "axis", "saturation_axis"}:
            return None
        return cls.normalize_saturation_axis(value)

    @staticmethod
    def infer_section_axis(section: str) -> str:
        # Backward-compatible fallback for legacy tables without explicit metadata.
        del section
        return "wetting"

    @classmethod
    def load_lookup_table(
        cls,
        filename: str,
        section: str,
        target_axis: str | None = None,
    ):
        sat_list = []
        value_list = []
        in_section = False
        source_axis = None
        target_axis = cls.normalize_saturation_axis(target_axis)
        with open(filename, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("--"):
                    in_section = line[2:].strip() == section
                    source_axis = None
                    continue
                if not in_section:
                    continue
                if line.startswith("#"):
                    parsed_axis = cls.parse_section_axis_metadata(line)
                    if parsed_axis is not None:
                        source_axis = parsed_axis
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue
                sat = float(parts[0])
                value = float(parts[1])

                source_axis_eff = source_axis or cls.infer_section_axis(section)
                target_axis_eff = target_axis or source_axis_eff
                if source_axis_eff != target_axis_eff:
                    sat = 1.0 - sat

                sat_list.append(sat)
                value_list.append(value)

        return np.asarray(sat_list, dtype=float), np.asarray(value_list, dtype=float)


class _KilloughRelPermBase:
    supports_history = True

    def __init__(self, corey, phase: str):
        self.phase = phase.lower()
        self.corey = corey
        self.history_model = KilloughLandModel(
            swc=corey.swc,
            sgrmax=corey.sgrmax,
        )

    def evaluate(self, sat, Sg_max=0, sg_max=None):
        if sg_max is not None:
            Sg_max = sg_max

        sat = float(sat)
        if self.phase in {"aq", "water", "w"}:
            return float(self.evaluate_drainage(sat))

        if sat >= Sg_max:
            return float(self.evaluate_drainage(sat))
        return float(self.evaluate_scanning(sat, Sg_max))

    def evaluate_drainage(self, sat: float) -> float:
        raise NotImplementedError

    def evaluate_scanning(self, sat: float, sg_max: float) -> float:
        raise NotImplementedError


class KilloughRelPermCorey(_KilloughRelPermBase):
    def __init__(self, corey, phase: str):
        super().__init__(corey, phase)
        self.land_constant = self.history_model.land_constant

    def evaluate_drainage(self, sat: float) -> float:
        if self.phase in {"aq", "water", "w"}:
            se = (sat - self.corey.swc) / (1.0 - self.corey.swc - self.corey.sgc)
            se = np.clip(se, 0.0, 1.0)
            return self.corey.krwe * se**self.corey.nw

        se = (sat - self.corey.sgc) / (1.0 - self.corey.swc - self.corey.sgc)
        se = np.clip(se, 0.0, 1.0)
        return self.corey.krge * se**self.corey.ng

    def evaluate_scanning(self, sat: float, sg_max: float) -> float:
        sg_max = min(1.0 - self.corey.swc, float(sg_max))
        sgr = self.history_model.residual_gas_saturation(sg_max)
        sat = min(1.0 - self.corey.swc, float(sat))
        mobile_sg = max(sat - sgr, 0.0)
        denom = max(sg_max - sgr, 1e-12)

        kr_inflection = self.evaluate_drainage(sg_max)
        factor = (mobile_sg / denom) ** self.corey.a
        return kr_inflection * factor


class KilloughRelPermTable(_LookupTableMixin, _KilloughRelPermBase):
    def __init__(self, corey, phase: str, lookup_file: str = "LookupTable.txt"):
        super().__init__(corey, phase)
        self.kind = "linear"
        table_section = (
            corey.wetting_type
            if self.phase in {"aq", "water", "w"}
            else corey.nowetting_d
        )
        target_axis = "wetting" if self.phase in {"aq", "water", "w"} else "nonwetting"
        self.sat_data, self.kr_data = self.load_lookup_table(
            lookup_file,
            table_section,
            target_axis=target_axis,
        )
        idx = np.argsort(self.sat_data)
        self.sat_data = self.sat_data[idx]
        self.kr_data = self.kr_data[idx]
        self.kr_interpolator = interp1d(
            self.sat_data,
            self.kr_data,
            kind=self.kind,
            bounds_error=False,
            fill_value=(self.kr_data[0], self.kr_data[-1]),
        )

        self.sat_dr, self.kr_dr = self.load_lookup_table(
            lookup_file,
            corey.nowetting_d,
            target_axis="nonwetting",
        )
        idx = np.argsort(self.sat_dr)
        self.sat_dr = self.sat_dr[idx]
        self.kr_dr = self.kr_dr[idx]
        self.sg_max_limit = self.sat_dr[-2]

        self.sat_im, self.kr_im = self.load_lookup_table(
            lookup_file,
            corey.nowetting_i,
            target_axis="nonwetting",
        )
        idx = np.argsort(self.sat_im)
        self.sat_im = self.sat_im[idx]
        self.kr_im = self.kr_im[idx]
        self.sgci_max = self.sat_im[1]
        self._scan_cache = {}

    def evaluate_drainage(self, sat: float) -> float:
        return float(self.kr_interpolator(sat))

    def _make_scanning_interp(self, sg_max: float):
        sg_max = min(float(sg_max), self.sg_max_limit)
        # sgr = self.history_model.residual_gas_saturation(sg_max)
        sgr = sg_max / 2
        if abs(sg_max - sgr) < 1e-12:
            self._scan_cache[sg_max] = self.kr_interpolator
            return self.kr_interpolator

        s1 = np.linspace(sgr, sg_max, 1000)
        s_star = self.sgci_max + (s1 - sgr) * (self.sg_max_limit - self.sgci_max) / (
            sg_max - sgr
        )
        kr_inflection = np.interp(sg_max, self.sat_dr, self.kr_dr)
        kr_ref = np.interp(self.sg_max_limit, self.sat_dr, self.kr_dr)
        kr1 = np.interp(s_star, self.sat_im, self.kr_im) * kr_inflection / kr_ref

        mask = self.sat_dr > sg_max
        s2 = self.sat_dr[mask]
        kr2 = self.kr_dr[mask]

        sats = np.concatenate([s1, s2])
        krs = np.concatenate([kr1, kr2])
        interp = interp1d(
            sats,
            krs,
            kind=self.kind,
            bounds_error=False,
            fill_value=(krs[0], krs[-1]),
        )
        self._scan_cache[sg_max] = interp
        return interp

    def evaluate_scanning(self, sat: float, sg_max: float) -> float:
        interp = self._scan_cache.get(sg_max)
        if interp is None:
            interp = self._make_scanning_interp(sg_max)
        return float(interp(sat))


class _KilloughCapillaryPressureBase:
    supports_history = True

    def __init__(self, corey, phase: str, epsilon: float = 0.1):
        self.phase = phase.lower()
        self.corey = corey
        self.epsilon = epsilon
        self.history_model = KilloughLandModel(
            swc=corey.swc,
            sgrmax=corey.sgrmax,
            epsilon=epsilon,
        )

    def evaluate(self, sat, Sg_max=0, sg_max=None):
        if sg_max is not None:
            Sg_max = sg_max

        if self.phase in {"gas", "v"}:
            return 0.0

        sg = float(np.clip(1.0 - sat, 0.0, self.sg_max_limit))
        pc_dr = self.evaluate_drainage(sg)
        if Sg_max is None:
            return pc_dr

        Sg_max = float(np.clip(Sg_max, 0.0, self.sg_max_limit))
        if sg >= Sg_max:
            return pc_dr

        # sgr = self.history_model.residual_gas_saturation(Sg_max)
        sgr = Sg_max / 2
        numerator = (
            1.0 / (1.0 - sg - (1.0 - Sg_max) + self.epsilon) - 1.0 / self.epsilon
        )
        denominator = (
            1.0 / ((1.0 - sgr) - (1.0 - Sg_max) + self.epsilon) - 1.0 / self.epsilon
        )
        fraction = numerator / denominator if abs(denominator) > 0 else 0.0
        fraction = float(np.clip(fraction, 0.0, 1.0))
        pc_im = self.evaluate_imbibition(sg)
        return pc_dr + fraction * (pc_im - pc_dr)

    def evaluate_drainage(self, sg: float) -> float:
        raise NotImplementedError

    def evaluate_imbibition(self, sg: float) -> float:
        raise NotImplementedError


class KilloughCapillaryPressureTable(_LookupTableMixin, _KilloughCapillaryPressureBase):
    def __init__(
        self,
        corey,
        phase: str,
        lookup_file: str = "LookupTable.txt",
        epsilon: float = 0.1,
    ):
        super().__init__(corey, phase, epsilon=epsilon)

        sat_dr, pc_dr = self.load_lookup_table(
            lookup_file,
            corey.Pc_drainage_section,
            target_axis="nonwetting",
        )
        idx = np.argsort(sat_dr)
        self.sat_dr = sat_dr[idx]
        self.pc_dr = pc_dr[idx]
        self.pc_drainage = interp1d(
            self.sat_dr,
            self.pc_dr,
            kind="linear",
            bounds_error=False,
            fill_value=(self.pc_dr[0], self.pc_dr[-1]),
        )
        self.sg_max_limit = self.sat_dr[-2]

        sat_im, pc_im = self.load_lookup_table(
            lookup_file,
            corey.Pc_imbibition_section,
            target_axis="nonwetting",
        )
        idx = np.argsort(sat_im)
        self.sat_im = sat_im[idx]
        self.pc_im = pc_im[idx]
        self.pc_imbibition = interp1d(
            self.sat_im,
            self.pc_im,
            kind="linear",
            bounds_error=False,
            fill_value=(self.pc_im[0], self.pc_im[-1]),
        )
        zero_im = np.where(np.isclose(self.pc_im, 0.0))[0]
        self.sgci_max = self.sat_im[zero_im[-1]] if len(zero_im) > 0 else self.sat_im[0]

    def evaluate_drainage(self, sg: float) -> float:
        return float(self.pc_drainage(sg)) * 1e-5

    def evaluate_imbibition(self, sg: float) -> float:
        return float(self.pc_imbibition(sg)) * 1e-5


# Backward-compatible aliases while the rest of the codebase migrates.
KilloughRelPerm = KilloughRelPermTable
KilloughCapillaryPressure = KilloughCapillaryPressureTable

__all__ = [
    "KilloughLandModel",
    "KilloughRelPerm",
    "KilloughRelPermCorey",
    "KilloughRelPermTable",
    "KilloughCapillaryPressure",
    "KilloughCapillaryPressureTable",
]
