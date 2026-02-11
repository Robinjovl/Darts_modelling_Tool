from reaktoro import *
import os


class Model:
    def __init__(self):
        db = SupcrtDatabase("supcrtbl")  # let's use the SUPCRTBL database for this simulation
        params = Params.embedded("PalandriKharaka.yaml")
        self.system = ChemicalSystem(db,
            AqueousPhase().set(ActivityModelPitzer()),
            GaseousPhase("CO2(g)").set(ActivityModelPengRobinsonPhreeqcOriginal()),
            MineralPhases("Calcite Magnesite Dolomite"),
            MineralReaction("Calcite").setRateModel(ReactionRateModelPalandriKharaka(params)),
            MineralReaction("Magnesite").setRateModel(ReactionRateModelPalandriKharaka(params)),
            MineralReaction("Dolomite").setRateModel(ReactionRateModelPalandriKharaka(params)),
            MineralSurface("Calcite", 6.0, "cm2/cm3"),
            MineralSurface("Magnesite", 6.0, "cm2/cm3"),
            MineralSurface("Dolomite", 6.0, "cm2/cm3"),
        )

        self.state = ChemicalState(self.system)
        self.state.set("H2O(aq)"  , 1.0, "kg")
        self.state.set("CO2(g)"   , 5.0, "mol")
        self.state.set("Calcite"  , 1.0, "mol")
        self.state.set("Magnesite", 1.0, "mol")
        # Important: when O2(aq) or H2(aq) are in the system, add an insignificant tiny amount of one of them to avoid numerical problems due to floating-point rounding errors!
        self.state.set("O2(aq)"   , 1.0, "umol")

        self.output_folder = 'output'
        self.output_filename = 'data.txt'


    def run(self):
        solver = KineticsSolver(self.system)  # the chemical kinetics solver to simulate the dissolution/precipitation of minerals

        day = units.seconds(1.0, "day")  # 1 day in seconds

        duration = 1500 * day  # simulate kinetics for 1500 days
        steps = 500  # consider 500 time steps

        dt = duration/steps  # compute time step

        # Initiate the chemical kinetics simulation
        self.table = Table()  # used to create a table of collected data during the simulation

        for i in range(steps + 1):

            result = solver.solve(self.state, dt)  # compute the chemical state of the system after it reacted for given time length

            assert result.succeeded(), f"Calculation failed at time step #{i}!"

            props = self.state.props()  # get the current thermodynamic and chemical properties of the system
            aprops = AqueousProps.compute(props)  # compute the current aqueous properties of the system

            # At every time step, collect chemical state data to be used later for plotting!
            self.table.column("Time")           << i*dt / day  # convert time from seconds to days
            self.table.column("Calcite")        << props.speciesAmount("Calcite")   # in mol
            self.table.column("Magnesite")      << props.speciesAmount("Magnesite") # in mol
            self.table.column("Dolomite")       << props.speciesAmount("Dolomite")  # in mol
            self.table.column("Ca+2")           << props.speciesAmount("Ca+2")      # in mol
            self.table.column("Mg+2")           << props.speciesAmount("Mg+2")      # in mol
            self.table.column("RateCalcite")    << props.reactionRate("Calcite")    # in mol/s
            self.table.column("RateMagnesite")  << props.reactionRate("Magnesite")  # in mol/s
            self.table.column("RateDolomite")   << props.reactionRate("Dolomite")   # in mol/s
            self.table.column("pH")             << aprops.pH()
            self.table.column("OmegaCalcite")   << aprops.saturationRatio("Calcite")
            self.table.column("OmegaMagnesite") << aprops.saturationRatio("Magnesite")
            self.table.column("OmegaDolomite")  << aprops.saturationRatio("Dolomite")

        # Once time steps have finished, you may want to save collected data to a file (e.g., to use in a spreadsheet software)
        if not os.path.exists(self.output_folder): os.makedirs(self.output_folder)
        output_file_path = os.path.join(self.output_folder, self.output_filename)
        self.table.save(output_file_path)
        return output_file_path
