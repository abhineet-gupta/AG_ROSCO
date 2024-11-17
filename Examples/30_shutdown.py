"""
30_shutdown
----------------
This example demonstrates turbine shutdown.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from rosco.toolbox.ofTools.case_gen.run_FAST import run_FAST_ROSCO
from rosco.toolbox.ofTools.case_gen import CaseLibrary as cl
from rosco.toolbox.ofTools.fast_io import output_processing

rpm2RadSec = 2.0 * (np.pi) / 60.0
deg2rad = np.pi/180.0

# directories
this_dir = os.path.dirname(os.path.abspath(__file__))
example_out_dir = os.path.join(this_dir, "examples_out")
os.makedirs(example_out_dir, exist_ok=True)


def main():
    # Test Config
    FULL_TEST = False  # FULL_TEST for local testing, otherwise shorter for CI

    # Input yaml and output directory
    parameter_filename = os.path.join(this_dir, "Tune_Cases/IEA15MW.yaml")

    # Set DISCON input dynamically through yaml/dict
    controller_params = {}
    controller_params["DISCON"] = {}
    controller_params["DISCON"]["Echo"] = 1
    controller_params["LoggingLevel"] = 3
    controller_params["SD_Mode"] = 1
    controller_params["DISCON"]["SD_EnablePitch"] = 1
    controller_params["DISCON"]["SD_MaxPit"] = 30*deg2rad
    controller_params["DISCON"]["SD_MaxPitchRate"] = 0.0348
    controller_params["DISCON"]["SD_MaxTorqueRate"] = 4500000
    controller_params["DISCON"]["SD_EnableTime"] = 1
    controller_params["DISCON"]["SD_Time"] = 20

    # simulation set up
    r = run_FAST_ROSCO()
    r.tuning_yaml = parameter_filename
    r.case_inputs = {}

    # Disable floating DOFs for clarity
    r.case_inputs[("ElastoDyn", "PtfmSgDOF")] = {"vals": ["False"], "group": 0}
    r.case_inputs[("ElastoDyn", "PtfmSwDOF")] = {"vals": ["False"], "group": 0}
    r.case_inputs[("ElastoDyn", "PtfmHvDOF")] = {"vals": ["False"], "group": 0}
    r.case_inputs[("ElastoDyn", "PtfmRDOF")] = {"vals": ["False"], "group": 0}
    r.case_inputs[("ElastoDyn", "PtfmPDOF")] = {"vals": ["False"], "group": 0}
    r.case_inputs[("ElastoDyn", "PtfmYDOF")] = {"vals": ["False"], "group": 0}


    t_max = 80

    run_dir = os.path.join(example_out_dir, "30_shutdown_demo/1_pitch")

    # Wind case
    r.wind_case_fcn = cl.ramp
    r.wind_case_opts = {
        "U_start": 25,
        "U_end": 50,
        "t_start": 0,
        "t_end": t_max,
    }
    r.case_inputs[("ElastoDyn", "BlPitch1")] = {"vals": [20.0], "group": 0}
    r.case_inputs[("ElastoDyn", "BlPitch2")] = {"vals": [20.0], "group": 0}
    r.case_inputs[("ElastoDyn", "BlPitch3")] = {"vals": [20.0], "group": 0}
    r.case_inputs[("ElastoDyn", "RotSpeed")] = {"vals": [10.0], "group": 0}

    # Run simulation 
    os.makedirs(run_dir, exist_ok=True)
    r.controller_params = controller_params
    r.save_dir = run_dir
    r.run_FAST()

    # Plot output
    outfile = [os.path.join(run_dir, "IEA15MW", "ramp", "base", "IEA15MW_0.outb")]
    cases = {}
    cases["Baseline"] = ["Wind1VelX", "BldPitch1", "GenTq", "RotSpeed", "GenPwr"]
    fast_out = output_processing.output_processing()
    fastout = fast_out.load_fast_out(outfile)
    fast_out.plot_fast_out(cases=cases, showplot=False)

    plt.savefig(os.path.join(example_out_dir, "30_shutdown.png"))


if __name__ == "__main__":
    main()