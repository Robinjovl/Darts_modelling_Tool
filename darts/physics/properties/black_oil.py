from typing import Literal

from darts.interpolators import property_evaluator_iface
from darts.physics.properties.evaluator_base import (
    EvaluatorBase,
    EvaluatorConfigBase,
    register_evaluator,
)
from darts.tools.interpolation import TableInterpolation
from darts.tools.keyword_file_tools import *


class _EvaluatorIfaceMeta(type(property_evaluator_iface), type(EvaluatorBase)):
    """Combined metaclass so a class can inherit both the C++ pybind11
    ``property_evaluator_iface`` and the Python ``EvaluatorBase`` ABC."""


class _PvtFileConfigBase(EvaluatorConfigBase):
    """Base config for evaluators parameterized by a single PVT file path."""

    pvt: str


class DensityGasConfig(_PvtFileConfigBase):
    """Configuration for DensityGas (black-oil PVT-driven gas density)."""

    kind: Literal["bo_density_gas"] = "bo_density_gas"


class DensityWatConfig(_PvtFileConfigBase):
    """Configuration for DensityWat (black-oil PVT-driven water density)."""

    kind: Literal["bo_density_wat"] = "bo_density_wat"


class ViscGasConfig(_PvtFileConfigBase):
    """Configuration for ViscGas (black-oil PVT-driven gas viscosity)."""

    kind: Literal["bo_visc_gas"] = "bo_visc_gas"


class ViscWatConfig(_PvtFileConfigBase):
    """Configuration for ViscWat (black-oil PVT-driven water viscosity)."""

    kind: Literal["bo_visc_wat"] = "bo_visc_wat"


class FlashBlackOilConfig(_PvtFileConfigBase):
    """Configuration for the black-oil flash evaluator."""

    kind: Literal["bo_flash"] = "bo_flash"


class DensityOilConfig(_PvtFileConfigBase):
    """Configuration for DensityOil (black-oil PVT-driven oil density)."""

    kind: Literal["bo_density_oil"] = "bo_density_oil"


class ViscOilConfig(_PvtFileConfigBase):
    """Configuration for ViscOil (black-oil PVT-driven oil viscosity)."""

    kind: Literal["bo_visc_oil"] = "bo_visc_oil"


class WatRelPermConfig(_PvtFileConfigBase):
    """Configuration for WatRelPerm (black-oil PVT-driven water relperm)."""

    kind: Literal["bo_wat_rel_perm"] = "bo_wat_rel_perm"


class GasRelPermConfig(_PvtFileConfigBase):
    """Configuration for GasRelPerm (black-oil PVT-driven gas relperm)."""

    kind: Literal["bo_gas_rel_perm"] = "bo_gas_rel_perm"


class OilRelPermConfig(_PvtFileConfigBase):
    """Configuration for OilRelPerm (black-oil PVT-driven oil relperm)."""

    kind: Literal["bo_oil_rel_perm"] = "bo_oil_rel_perm"


class BlackOilRockCompactionConfig(_PvtFileConfigBase):
    """Configuration for the black-oil rock compaction evaluator."""

    kind: Literal["bo_rock_compaction"] = "bo_rock_compaction"


class CapillaryPressurePcowConfig(_PvtFileConfigBase):
    """Configuration for CapillaryPressurePcow (Pc oil/water from SWOF)."""

    kind: Literal["bo_pcow"] = "bo_pcow"


class CapillaryPressurePcgoConfig(_PvtFileConfigBase):
    """Configuration for CapillaryPressurePcgo (Pc gas/oil from SGOF)."""

    kind: Literal["bo_pcgo"] = "bo_pcgo"


class DensityGas(EvaluatorBase):
    """Black-oil gas density from PVDG/DENSITY tables: ``rho_g = dens_sc / Bg(p)``."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.dens_sc = get_table_keyword(self.pvt, 'DENSITY')[0][2]
        self.table = get_table_keyword(self.pvt, 'PVDG')

    def evaluate(self, pres, pbub, xgo):
        """
        :param pres: pressure [bar]
        :type pres: float
        :param pbub: bubble point pressure [bar] (unused)
        :type pbub: float
        :param xgo: gas-in-oil mass fraction (unused)
        :type xgo: float
        :return: gas density [kg/m3]
        :rtype: float
        """
        pres_index = 0  # first column in the table - pressure
        Bg_index = 1  # second column in the table - volume factor
        Table = TableInterpolation()

        if pres < self.table[0][0] or pres > self.table[len(self.table) - 1][0]:
            Bg = Table.LinearExtraP(self.table, pres, pres_index, Bg_index)
        else:
            Bg = Table.LinearInterP(self.table, pres, pres_index, Bg_index)

        return self.dens_sc / Bg


register_evaluator("bo_density_gas", DensityGas, DensityGasConfig)


class DensityWat(EvaluatorBase):
    """Black-oil water density from PVTW/DENSITY tables (Taylor-expanded
    formation volume factor)."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.dens_sc = get_table_keyword(self.pvt, 'DENSITY')[0][1]
        self.table = get_table_keyword(self.pvt, 'PVTW')[0]

    def evaluate(self, pres, pbub, xgo):
        """
        :param pres: pressure [bar]
        :type pres: float
        :param pbub: bubble point pressure [bar] (unused)
        :type pbub: float
        :param xgo: gas-in-oil mass fraction (unused)
        :type xgo: float
        :return: water density [kg/m3]
        :rtype: float
        """
        X = self.table[2] * (pres - self.table[0])
        Bw = self.table[1] / (1 + X + X * X / 2)

        return self.dens_sc / Bw


register_evaluator("bo_density_wat", DensityWat, DensityWatConfig)


class ViscGas(EvaluatorBase):
    """Black-oil gas viscosity from the PVDG table."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.table = get_table_keyword(self.pvt, 'PVDG')

    def evaluate(self, pres, pbub):
        """
        :param pres: pressure [bar]
        :type pres: float
        :param pbub: bubble point pressure [bar] (unused)
        :type pbub: float
        :return: gas viscosity [cP]
        :rtype: float
        """
        pres_index = 0  # first column in the table - pressure
        vgas_index = 2  # third column in the table - viscosity
        Table = TableInterpolation()

        if pres < self.table[0][0] or pres > self.table[len(self.table) - 1][0]:
            visco_gas = Table.LinearExtraP(self.table, pres, pres_index, vgas_index)
        else:
            visco_gas = Table.LinearInterP(self.table, pres, pres_index, vgas_index)

        return visco_gas


register_evaluator("bo_visc_gas", ViscGas, ViscGasConfig)


class ViscWat(EvaluatorBase):
    """Black-oil water viscosity from the PVTW table (Taylor-expanded form)."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.table = get_table_keyword(self.pvt, 'PVTW')[0]

    def evaluate(self, pres, pbub):
        """
        :param pres: pressure [bar]
        :type pres: float
        :param pbub: bubble point pressure [bar] (unused)
        :type pbub: float
        :return: water viscosity [cP]
        :rtype: float
        """
        Y = -self.table[4] * (pres - self.table[0])
        visco_wat = self.table[3] / (1 + Y + Y * Y / 2)

        return visco_wat


register_evaluator("bo_visc_wat", ViscWat, ViscWatConfig)


class flash_black_oil(EvaluatorBase):
    """Black-oil two-phase flash returning ``(xgo, V, pbub)`` from the PVTO table."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.dens = get_table_keyword(self.pvt, 'DENSITY')[0]
        self.oil_dens = self.dens[0]
        self.gas_dens = self.dens[2]
        self.table = get_table_keyword(self.pvt, 'PVTO')
        self.len_table = len(self.table)

    def bubble_point_pressure(self, pres, z):
        """
        :param pres: pressure [bar]
        :type pres: float
        :param z: overall composition ``[zg, zo, ...]``
        :type z: list[float] | np.ndarray
        :return: bubble point pressure [bar]
        :rtype: float
        """
        zg = z[0]
        zo = z[1]
        sat_rs = self.oil_dens / (self.gas_dens * (zo + zg) / zg - self.gas_dens)
        rs_index = 0  # first column in the table - rs
        pres_index = 1  # second column in the table - pressure
        Table = TableInterpolation()

        # find the index of saturated Rs
        for i in range(self.len_table - 1):
            if self.table[i][0] == self.table[i + 1][0]:
                num = i
                break

        if sat_rs < self.table[0][0]:
            pbub = Table.LinearExtraP(self.table, sat_rs, rs_index, pres_index)
        elif sat_rs > self.table[self.len_table - 1][0]:
            pbub = Table.SatExtrapolation(self.table, sat_rs, rs_index, pres_index, num)
        else:
            pbub = Table.LinearInterP(self.table, sat_rs, rs_index, pres_index)

        if pres > pbub:
            pbub = pbub
        else:
            pbub = pres

        return pbub

    def evaluate(self, pres, z):
        """
        :param pres: pressure [bar]
        :type pres: float
        :param z: overall composition ``[zg, zo, ...]``
        :type z: list[float] | np.ndarray
        :return: tuple ``(xgo, V, pbub)`` -- gas-in-oil mass fraction, vapour
            mole fraction, and bubble point pressure
        :rtype: tuple[float, float, float]
        """
        zg = z[0]
        zo = z[1]
        zw = 1 - zg - zo

        pbub = self.bubble_point_pressure(pres, z)

        rs_index = 0  # first column in the table - rs
        pres_index = 1  # second column in the table - pressure
        Table = TableInterpolation()

        # find the index of saturated Rs
        for i in range(self.len_table - 1):
            if self.table[i][0] == self.table[i + 1][0]:
                num = i
                break

        # undersaturated condition
        if pres > pbub:
            rs = self.oil_dens / (self.gas_dens * (zo + zg) / zg - self.gas_dens)
        # saturated condition
        else:
            if pres < self.table[0][1]:
                rs = Table.LinearExtraP(self.table, pres, pres_index, rs_index)
            elif pres > self.table[num][1]:
                rs = Table.SatExtrapolation(self.table, pres, pres_index, rs_index, num)
            else:
                rs = Table.LinearInterP(self.table, pres, pres_index, rs_index)

        xgo = self.gas_dens * rs / (self.oil_dens + self.gas_dens * rs)

        V = 1 - zw - zo / (1 - xgo)

        return xgo, V, pbub


register_evaluator("bo_flash", flash_black_oil, FlashBlackOilConfig)


class DensityOil(EvaluatorBase):
    """Black-oil oil density from the PVTO table with saturated/undersaturated branches."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.dens_sc = get_table_keyword(self.pvt, 'DENSITY')[0][0]
        self.table = get_table_keyword(self.pvt, 'PVTO')
        self.len_table = len(self.table)

    def evaluate(self, pres, pbub, xgo):
        """
        :param pres: pressure [bar]
        :type pres: float
        :param pbub: bubble point pressure [bar]
        :type pbub: float
        :param xgo: gas-in-oil mass fraction
        :type xgo: float
        :return: oil density [kg/m3]
        :rtype: float
        """
        pres_index = 1
        Bo_index = 2

        Table = TableInterpolation()

        # find the index of max Bo
        for i in range(self.len_table - 1):
            if self.table[i][2] >= self.table[i + 1][2]:
                num = i
                break

        # calculate the saturated Bo
        if pbub < self.table[0][1]:
            Bo_bub = Table.LinearExtraP(self.table, pbub, pres_index, Bo_index)
        elif pbub > self.table[num][1]:
            Bo_bub = Table.SatExtrapolation(self.table, pbub, pres_index, Bo_index, num)
        else:
            Bo_bub = Table.LinearInterP(self.table, pbub, pres_index, Bo_index)

        # calculate Bo in current pressure
        # (1) saturated condition
        if pres < pbub:
            if pres < self.table[0][1]:
                Bo = Table.LinearExtraP(self.table, pres, pres_index, Bo_index)
            elif pres > self.table[num][1]:
                Bo = Table.SatExtrapolation(self.table, pres, pres_index, Bo_index, num)
            else:
                Bo = Table.LinearInterP(self.table, pres, pres_index, Bo_index)
        # (2) undersaturated condition
        else:
            pres_undersat = pres + self.table[num][1] - pbub
            Bo_undersat = Table.SatExtrapolation(
                self.table, pres_undersat, pres_index, Bo_index, num + 1
            )
            if pbub < self.table[num][1]:
                Bo = Bo_undersat * Bo_bub / self.table[num][2]
            else:
                Bo = Bo_undersat - (self.table[num][2] - Bo_bub)

        return self.dens_sc / Bo / (1 - xgo)


register_evaluator("bo_density_oil", DensityOil, DensityOilConfig)


class ViscOil(EvaluatorBase):
    """Black-oil oil viscosity from the PVTO table with saturated/undersaturated branches."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.table = get_table_keyword(self.pvt, 'PVTO')
        self.len_table = len(self.table)

    def evaluate(self, pres, pbub):
        """
        :param pres: pressure [bar]
        :type pres: float
        :param pbub: bubble point pressure [bar]
        :type pbub: float
        :return: oil viscosity [cP]
        :rtype: float
        """
        pres_index = 1
        visco_index = 3

        Table = TableInterpolation()

        # find the index of min visco
        for i in range(self.len_table - 1):
            if self.table[i][3] <= self.table[i + 1][3]:
                num = i
                break

        # calculate the saturated viscosity
        if pbub < self.table[0][1]:
            visco_bub = Table.LinearExtraP(self.table, pbub, pres_index, visco_index)
        elif pbub > self.table[num][1]:
            visco_bub = Table.SatExtrapolation(
                self.table, pbub, pres_index, visco_index, num
            )
        else:
            visco_bub = Table.LinearInterP(self.table, pbub, pres_index, visco_index)

        # calculate viscosity at current pressure
        # (1) saturated condition
        if pres < pbub:
            if pres < self.table[0][1]:
                visco = Table.LinearExtraP(self.table, pres, pres_index, visco_index)
            elif pres > self.table[num][1]:
                visco = Table.SatExtrapolation(
                    self.table, pres, pres_index, visco_index, num
                )
            else:
                visco = Table.LinearInterP(self.table, pres, pres_index, visco_index)
        # (2) undersaturated condition
        else:
            pres_undersat = pres + self.table[num][1] - pbub
            visco_undersat = Table.SatExtrapolation(
                self.table, pres_undersat, pres_index, visco_index, num + 1
            )
            if pbub < self.table[num][1]:
                visco = visco_undersat * visco_bub / self.table[num][3]
            else:
                visco = visco_undersat - (self.table[num][3] - visco_bub)

        return visco


register_evaluator("bo_visc_oil", ViscOil, ViscOilConfig)


class WatRelPerm(EvaluatorBase):
    """Black-oil water relative permeability from the SWOF table."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.SWOF = get_table_keyword(self.pvt, 'SWOF')

    def evaluate(self, gas_sat, wat_sat):
        """
        :param gas_sat: gas saturation (unused)
        :type gas_sat: float
        :param wat_sat: water saturation
        :type wat_sat: float
        :return: water relative permeability
        :rtype: float
        """
        wat_index = 0
        krw_index = 1

        Table = TableInterpolation()
        if wat_sat < self.SWOF[0][0] or wat_sat > self.SWOF[len(self.SWOF) - 1][0]:
            krw = Table.SCALExtraP(self.SWOF, wat_sat, wat_index, krw_index)
        else:
            krw = Table.LinearInterP(self.SWOF, wat_sat, wat_index, krw_index)

        return krw


register_evaluator("bo_wat_rel_perm", WatRelPerm, WatRelPermConfig)


class GasRelPerm(EvaluatorBase):
    """Black-oil gas relative permeability from the SGOF table."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.SGOF = get_table_keyword(self.pvt, 'SGOF')

    def evaluate(self, gas_sat, wat_sat):
        """
        :param gas_sat: gas saturation
        :type gas_sat: float
        :param wat_sat: water saturation (unused)
        :type wat_sat: float
        :return: gas relative permeability
        :rtype: float
        """
        gas_index = 0
        krg_index = 1

        Table = TableInterpolation()
        if gas_sat < self.SGOF[0][0] or gas_sat > self.SGOF[len(self.SGOF) - 1][0]:
            krg = Table.SCALExtraP(self.SGOF, gas_sat, gas_index, krg_index)
        else:
            krg = Table.LinearInterP(self.SGOF, gas_sat, gas_index, krg_index)

        return krg


register_evaluator("bo_gas_rel_perm", GasRelPerm, GasRelPermConfig)


# here we use Stone I model to calculate oil relperm
class OilRelPerm(EvaluatorBase):
    """Three-phase oil relative permeability via Stone I, combining SWOF (Krow)
    and SGOF (Krog) tables."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.SGOF = get_table_keyword(self.pvt, 'SGOF')
        self.SWOF = get_table_keyword(self.pvt, 'SWOF')

    def Krog(self, gas_sat):
        """
        :param gas_sat: gas saturation
        :type gas_sat: float
        :return: oil relative permeability in oil-gas system
        :rtype: float
        """
        gas_index = 0
        krog_index = 2

        Table = TableInterpolation()
        if gas_sat < self.SGOF[0][0] or gas_sat > self.SGOF[len(self.SGOF) - 1][0]:
            krog = Table.SCALExtraP(self.SGOF, gas_sat, gas_index, krog_index)
        else:
            krog = Table.LinearInterP(self.SGOF, gas_sat, gas_index, krog_index)

        return krog

    def Krow(self, wat_sat):
        """
        :param wat_sat: water saturation
        :type wat_sat: float
        :return: oil relative permeability in oil-water system
        :rtype: float
        """
        wat_index = 0
        krow_index = 2

        Table = TableInterpolation()
        if wat_sat < self.SWOF[0][0] or wat_sat > self.SWOF[len(self.SWOF) - 1][0]:
            krow = Table.SCALExtraP(self.SWOF, wat_sat, wat_index, krow_index)
        else:
            krow = Table.LinearInterP(self.SWOF, wat_sat, wat_index, krow_index)

        return krow

    def evaluate(self, gas_sat, wat_sat):
        """
        :param gas_sat: gas saturation
        :type gas_sat: float
        :param wat_sat: water saturation
        :type wat_sat: float
        :return: oil relative permeability (Stone I)
        :rtype: float
        """
        len_SWOF = len(self.SWOF)
        len_SGOF = len(self.SGOF)
        Sorw = 1 - self.SWOF[len_SWOF - 1][0]
        Sorg = 1 - self.SGOF[len_SGOF - 1][0]
        Swc = self.SWOF[0][0]
        MINIMAL_FOR_COMPARE = 1e-12
        Krocw = self.SWOF[0][2]
        SatLimit = 1 - 2 * MINIMAL_FOR_COMPARE
        oil_sat = 1 - wat_sat - gas_sat

        krow = self.Krow(wat_sat)
        krog = self.Krog(gas_sat)

        if gas_sat < MINIMAL_FOR_COMPARE:
            kro = krow  # water-oil two phases
        elif wat_sat < Swc:
            kro = krog  # water phase not mobile
        else:
            # Stone I model -> alpha
            alpha = 1 - gas_sat / (1 - min(Swc + Sorg, SatLimit))
            # -> Som
            Som = alpha * Sorw + (1.0 - alpha) * Sorg
            # -> denom
            denom = 1.0 / (1.0 - min(Swc + Som, SatLimit))
            # Normalized saturations
            if (oil_sat - Som) > 0:
                Ma = oil_sat - Som
            else:
                Ma = 0
            SoStar = Ma * denom
            SwStar = min(wat_sat - Swc, SatLimit) * denom
            SgStar = min(gas_sat, SatLimit) * denom

            kro = (SoStar * krow * krog) / (Krocw * (1.0 - SwStar) * (1.0 - SgStar))

        return kro


register_evaluator("bo_oil_rel_perm", OilRelPerm, OilRelPermConfig)


class RockCompactionEvaluator(EvaluatorBase):
    """Black-oil rock compaction multiplier from the ROCK keyword:
    ``mult = 1 + cr * (p - p_ref)``."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.rock_table = get_table_keyword(self.pvt, 'ROCK')

    def evaluate(self, state):
        """
        :param state: state vector with pressure as the first element
        :type state: list[float] | np.ndarray
        :return: rock compaction multiplier
        :rtype: float
        """
        pressure = state[0]
        pressure_ref = self.rock_table[0][0]
        compressibility = self.rock_table[0][1]

        return 1.0 + compressibility * (pressure - pressure_ref)


register_evaluator(
    "bo_rock_compaction", RockCompactionEvaluator, BlackOilRockCompactionConfig
)


# capillary pressure based on table
class CapillaryPressurePcow(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """Oil-water capillary pressure interpolated from the SWOF table."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.SWOF = get_table_keyword(self.pvt, 'SWOF')

    def evaluate(self, wat_sat):
        """
        :param wat_sat: water saturation
        :type wat_sat: float
        :return: oil-water capillary pressure
        :rtype: float
        """
        wat_index = 0
        pc_index = 3

        Table = TableInterpolation()
        if wat_sat < self.SWOF[0][0] or wat_sat > self.SWOF[len(self.SWOF) - 1][0]:
            pcow = Table.SCALExtraP(self.SWOF, wat_sat, wat_index, pc_index)
        else:
            pcow = Table.LinearInterP(self.SWOF, wat_sat, wat_index, pc_index)

        return pcow


register_evaluator("bo_pcow", CapillaryPressurePcow, CapillaryPressurePcowConfig)


class CapillaryPressurePcgo(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """Gas-oil capillary pressure interpolated from the SGOF table."""

    def __init__(self, pvt):
        """
        :param pvt: path to the PVT keyword file
        :type pvt: str | os.PathLike
        """
        super().__init__()
        self.pvt = str(pvt)
        self.SGOF = get_table_keyword(self.pvt, 'SGOF')

    def evaluate(self, gas_sat):
        """
        :param gas_sat: gas saturation
        :type gas_sat: float
        :return: gas-oil capillary pressure
        :rtype: float
        """
        gas_index = 0
        pc_index = 3

        Table = TableInterpolation()
        if gas_sat < self.SGOF[0][0] or gas_sat > self.SGOF[len(self.SGOF) - 1][0]:
            pcgo = Table.SCALExtraP(self.SGOF, gas_sat, gas_index, pc_index)
        else:
            pcgo = Table.LinearInterP(self.SGOF, gas_sat, gas_index, pc_index)

        return pcgo


register_evaluator("bo_pcgo", CapillaryPressurePcgo, CapillaryPressurePcgoConfig)
