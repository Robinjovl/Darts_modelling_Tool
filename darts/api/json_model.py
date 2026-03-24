"""
DartsModel subclass with lifecycle overrides for JSON-driven execution.

JsonModel extends the standard DartsModel so that reservoir, physics, and
well setup steps are no-ops — ModelBuilder has already configured them
before ``init()`` is called.  It also provides helpers for parsing simulation
runtime from the spec and collecting output file paths after a run completes.
"""

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
        self.set_wells_from_dict(wells_spec.model_dump())

    def set_initial_conditions(self):
        ic_spec = getattr(self, '_initial_conditions_spec', None)
        if not ic_spec or not getattr(ic_spec, 'by_array', None):
            return
        self.set_initial_conditions_from_dict(ic_spec.model_dump())

    def set_well_controls(self):
        wc_spec = getattr(self, '_well_controls_spec', None)
        wells_spec = getattr(self, '_wells_spec', None)
        wells_dict = wells_spec.model_dump() if wells_spec else None
        wc_dict = wc_spec.model_dump() if wc_spec else None
        self.set_well_controls_from_dict(
            wells_dict=wells_dict, well_controls_dict=wc_dict
        )
