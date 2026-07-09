import shutil
import sys
from pathlib import Path

from parflow import Run
from parflow.tools.compare import pf_test_file
from parflow.tools.fs import mkdir, rm
from parflow.tools.settings import set_working_directory

from colm_ci_settings import (
    CORRECT_DIR,
    DZ,
    DX,
    DY,
    DZ_SCALE_VALUES,
    FORCING_DIR,
    GEOLOGY_GEOMS,
    INDICATOR_GEOMS,
    INPUT_DIR,
    NX,
    NY,
    NZ,
    PERM_VALUES,
    POROSITY_VALUES,
    RUN_DIR,
    RUN_NAME,
    SOIL_GEOMS,
    SRES_VALUES,
    STOP_TIME,
    VG_VALUES,
)
def copy_inputs() -> None:
    if RUN_DIR.exists():
        rm(str(RUN_DIR))
    mkdir(str(RUN_DIR))
    for source in INPUT_DIR.iterdir():
        if source.is_file():
            shutil.copy(source, RUN_DIR)


def set_constant_value(target, value: float) -> None:
    target.Type = "Constant"
    target.Value = value


def configure_domain(model: Run) -> None:
    model.FileVersion = 4
    model.Process.Topology.P = 1
    model.Process.Topology.Q = 1
    model.Process.Topology.R = 1
    model.ComputationalGrid.Lower.X = 0.0
    model.ComputationalGrid.Lower.Y = 0.0
    model.ComputationalGrid.Lower.Z = 0.0
    model.ComputationalGrid.DX = DX
    model.ComputationalGrid.DY = DY
    model.ComputationalGrid.DZ = DZ
    model.ComputationalGrid.NX = NX
    model.ComputationalGrid.NY = NY
    model.ComputationalGrid.NZ = NZ
    model.GeomInput.Names = "domaininput indi_input"
    model.GeomInput.domaininput.InputType = "Box"
    model.GeomInput.domaininput.GeomName = "domain"
    model.Geom.domain.Lower.X = 0.0
    model.Geom.domain.Lower.Y = 0.0
    model.Geom.domain.Lower.Z = 0.0
    model.Geom.domain.Upper.X = NX * DX
    model.Geom.domain.Upper.Y = NY * DY
    model.Geom.domain.Upper.Z = NZ * DZ
    model.Domain.GeomName = "domain"
    model.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"


def configure_indicator(model: Run) -> None:
    model.GeomInput.indi_input.InputType = "IndicatorField"
    model.GeomInput.indi_input.GeomNames = " ".join(INDICATOR_GEOMS)
    model.Geom.indi_input.FileName = "subsurface_11layer.pfb"
    for index, geom in enumerate(SOIL_GEOMS, start=1):
        model.GeomInput[geom].Value = index
    for index, geom in enumerate(GEOLOGY_GEOMS, start=19):
        model.GeomInput[geom].Value = index


def configure_variable_dz(model: Run) -> None:
    model.Solver.Nonlinear.VariableDz = True
    model.dzScale.GeomNames = "domain"
    model.dzScale.Type = "nzList"
    model.dzScale.nzListNumber = NZ
    for index, value in enumerate(DZ_SCALE_VALUES):
        model.Cell[f"_{index}"].dzScale.Value = value


def configure_subsurface(model: Run) -> None:
    model.TopoSlopesX.Type = "PFBFile"
    model.TopoSlopesX.GeomNames = "domain"
    model.TopoSlopesX.FileName = "slope_x.pfb"
    model.TopoSlopesY.Type = "PFBFile"
    model.TopoSlopesY.GeomNames = "domain"
    model.TopoSlopesY.FileName = "slope_y.pfb"
    model.Geom.Perm.Names = " ".join(["domain"] + INDICATOR_GEOMS)
    for geom, value in PERM_VALUES.items():
        set_constant_value(model.Geom[geom].Perm, value)
    model.Perm.TensorType = "TensorByGeom"
    model.Geom.Perm.TensorByGeom.Names = "domain"
    model.Geom.domain.Perm.TensorValX = 1.0
    model.Geom.domain.Perm.TensorValY = 1.0
    model.Geom.domain.Perm.TensorValZ = 1.0
    model.SpecificStorage.Type = "Constant"
    model.SpecificStorage.GeomNames = "domain"
    model.Geom.domain.SpecificStorage.Value = 0.0001
    model.Geom.Porosity.GeomNames = " ".join(POROSITY_VALUES)
    for geom, value in POROSITY_VALUES.items():
        set_constant_value(model.Geom[geom].Porosity, value)


def configure_van_genuchten(model: Run) -> None:
    vg_geoms = " ".join(VG_VALUES)
    model.Phase.RelPerm.Type = "VanGenuchten"
    model.Phase.RelPerm.GeomNames = vg_geoms
    model.Phase.Saturation.Type = "VanGenuchten"
    model.Phase.Saturation.GeomNames = vg_geoms
    for geom, (alpha, n_value) in VG_VALUES.items():
        model.Geom[geom].RelPerm.Alpha = alpha
        model.Geom[geom].RelPerm.N = n_value
        model.Geom[geom].Saturation.Alpha = alpha
        model.Geom[geom].Saturation.N = n_value
        model.Geom[geom].Saturation.SRes = SRES_VALUES.get(geom, 0.0001)
        model.Geom[geom].Saturation.SSat = 1.0


def configure_solver(model: Run) -> None:
    model.Mannings.Type = "Constant"
    model.Mannings.GeomNames = "domain"
    model.Mannings.Geom.domain.Value = 0.0000044
    model.Phase.Names = "water"
    set_constant_value(model.Phase.water.Density, 1.0)
    set_constant_value(model.Phase.water.Viscosity, 1.0)
    set_constant_value(model.Phase.water.Mobility, 1.0)
    model.Contaminants.Names = ""
    model.Gravity = 1.0
    model.Wells.Names = ""
    model.PhaseSources.water.Type = "Constant"
    model.PhaseSources.water.GeomNames = "domain"
    model.PhaseSources.water.Geom.domain.Value = 0.0
    model.TimingInfo.BaseUnit = 1.0
    model.TimingInfo.StartCount = 0
    model.TimingInfo.StartTime = 0.0
    model.TimingInfo.StopTime = STOP_TIME
    model.TimingInfo.DumpInterval = 1
    model.TimeStep.Type = "Constant"
    model.TimeStep.Value = 1.0
    model.Cycle.Names = "constant"
    model.Cycle.constant.Names = "alltime"
    model.Cycle.constant.alltime.Length = 1
    model.Cycle.constant.Repeat = -1


def configure_boundary_conditions(model: Run) -> None:
    model.BCPressure.PatchNames = model.Geom.domain.Patches
    for patch in ["x_lower", "x_upper", "y_lower", "y_upper", "z_lower"]:
        model.Patch[patch].BCPressure.Type = "FluxConst"
        model.Patch[patch].BCPressure.Cycle = "constant"
        model.Patch[patch].BCPressure.alltime.Value = 0.0
    model.Patch.z_upper.BCPressure.Type = "OverlandKinematic"
    model.Patch.z_upper.BCPressure.Cycle = "constant"
    model.Patch.z_upper.BCPressure.alltime.Value = 0.0
    model.ICPressure.Type = "PFBFile"
    model.ICPressure.GeomNames = "domain"
    model.Geom.domain.ICPressure.RefPatch = "top"
    model.Geom.domain.ICPressure.FileName = "initial_press_11layer.pfb"


def configure_colm(model: Run) -> None:
    model.Solver.LSM = "CoLM"
    model.Solver.CLM.CLMFileDir = "clm_output/"
    model.Solver.CLM.Print1dOut = False
    model.Solver.CLM.DailyRST = False
    model.Solver.CLM.CLMDumpInterval = 1
    model.Solver.CLM.MetFileName = "E5L"
    model.Solver.CLM.MetFilePath = str(FORCING_DIR)
    model.Solver.CLM.MetForcing = "3D"
    model.Solver.CLM.MetFileNT = 24
    model.Solver.CLM.IstepStart = 1
    model.Solver.CLM.EvapBeta = "Linear"
    model.Solver.CLM.VegWaterStress = "Saturation"
    model.Solver.CLM.ResSat = 0.1
    model.Solver.CLM.WiltingPoint = 0.12
    model.Solver.CLM.FieldCapacity = 0.98
    model.Solver.CLM.IrrigationType = "none"
    model.Solver.CLM.RootZoneNZ = 10
    model.Solver.CLM.SoiLayer = 10


def configure_outputs_and_numerics(model: Run) -> None:
    model.Solver.PrintSubsurfData = False
    model.Solver.PrintPressure = True
    model.Solver.PrintSaturation = True
    model.Solver.PrintMask = False
    model.Solver.PrintVelocities = False
    model.Solver.PrintEvapTrans = True
    model.Solver.CLM.SingleFile = True
    model.Solver.PrintSlopes = False
    model.Solver.PrintMannings = False
    model.Solver.WriteCLMBinary = False
    model.Solver.PrintCLM = True
    model.Solver = "Richards"
    model.Solver.TerrainFollowingGrid = True
    model.Solver.TerrainFollowingGrid.SlopeUpwindFormulation = "Upwind"
    model.Solver.Linear.Preconditioner = "PFMG"
    model.KnownSolution = "NoKnownSolution"
    model.Solver.MaxIter = 25000
    model.Solver.Drop = 1e-20
    model.Solver.AbsTol = 1e-8
    model.Solver.MaxConvergenceFailures = 8
    model.Solver.Nonlinear.MaxIter = 1000
    model.Solver.Nonlinear.ResidualTol = 1e-6
    model.Solver.Nonlinear.EtaChoice = "EtaConstant"
    model.Solver.Nonlinear.EtaValue = 0.001
    model.Solver.Nonlinear.UseJacobian = True
    model.Solver.Nonlinear.DerivativeEpsilon = 1e-16
    model.Solver.Nonlinear.StepTol = 1e-15
    model.Solver.Nonlinear.Globalization = "LineSearch"
    model.Solver.Linear.KrylovDimension = 70
    model.Solver.Linear.MaxRestarts = 2


def configure_model() -> Run:
    model = Run(RUN_NAME, __file__)
    configure_domain(model)
    configure_indicator(model)
    configure_variable_dz(model)
    configure_subsurface(model)
    configure_van_genuchten(model)
    configure_solver(model)
    configure_boundary_conditions(model)
    configure_colm(model)
    configure_outputs_and_numerics(model)
    return model


def validate_results() -> bool:
    test_files = [
        "PF_CoLM_CI.out.press.00005.pfb",
        "PF_CoLM_CI.out.satur.00005.pfb",
        "PF_CoLM_CI.out.evaptrans.00005.pfb",
        "PF_CoLM_CI.out.clm_output.00005.C.pfb",
    ]
    passed = True
    for name in test_files:
        if not pf_test_file(str(RUN_DIR / name), str(CORRECT_DIR / name), f"Max difference in {name}"):
            passed = False
    return passed


def main() -> None:
    copy_inputs()
    set_working_directory(str(RUN_DIR))
    model = configure_model()
    for filename in ["slope_x.pfb", "slope_y.pfb", "subsurface_11layer.pfb", "initial_press_11layer.pfb"]:
        model.dist(str(RUN_DIR / filename))
    model.run(working_directory=str(RUN_DIR))
    passed = validate_results()
    rm(str(RUN_DIR))
    print(f"{RUN_NAME} : {'PASSED' if passed else 'FAILED'}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
