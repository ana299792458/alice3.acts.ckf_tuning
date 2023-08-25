#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
July 2023
@author: anaelle chalumeau

Usage: python3 Examples/Scripts/Optimization/scriptsEtaSegments/mergeRootFiles.py -o Optuna -r run_name
"""

import subprocess
from pathlib import Path


def merge(optim, run_name, etaStep=1.0, etaStart=-4.0, etaStop=4.0):
    # Define the list of eta ranges
    eta_ranges =  [(int(round(x, 1)), int(round(x + etaStep, 1))) for x in [i * etaStep for i in range(int(etaStart / etaStep), int(etaStop / etaStep))]]
    # Prepare the list of input file paths
    input_files = []
    for eta_range in eta_ranges:
        eta_min, eta_max = eta_range
        if optim == "ckf":
            file_path = Path(f"/Users/achalume/alice/actsdir/test-ckf/output/test-fatras/fatras-subranges/e{eta_min}_{eta_max}seed_fatras/performance_ckf.root")
        #    if etaStep < 1.0:
        #        file_path = Path(f"/Users/achalume/alice/actsdir/test-ckf/output/n10/subruns/step,{str(etaStep)[2:]}/ckf_default_{eta_min}_{eta_max}_{run_name}/performance_ckf.root")
        #    else:
        #        file_path = Path(f"/Users/achalume/alice/actsdir/test-ckf/output/n10/subruns/step{etaStep}/ckf_default_{eta_min}_{eta_max}_{run_name}/performance_ckf.root")
        else:
            file_path = f"Examples/Scripts/Optimization/{optim}_output_CKF/{optim}_output_CKF_{eta_min}_{eta_max}_{run_name}/performance_ckf.root"
        input_files.append(file_path)

    # Define the output file path
    if optim == "ckf":
        output_file = f"/Users/achalume/alice/actsdir/test-ckf/output/test-fatras/fatras-subranges/merged_performance_ckf_default_{run_name}.root"
        #if etaStep < 1.0:
        #    output_file = f"/Users/achalume/alice/actsdir/test-ckf/output/n10/subruns/step,{str(etaStep)[2:]}/merged_performance_ckf_ONLY_default_{run_name}.root"
        #else:
        #    output_file = f"/Users/achalume/alice/actsdir/test-ckf/output/n10/subruns/step,{etaStep}/merged_performance_ckf_ONLY_default_{run_name}.root"
    else:
        output_file = f"Examples/Scripts/Optimization/{optim}_output_CKF/merged_performance_ckf_{run_name}.root"

    # Run the hadd command to merge the files
    subprocess.run(["hadd", output_file] + input_files)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Command line argument for merging performance_ckf.root')
    parser.add_argument('-o','--optim', type=str, required=True, help='Name of the optimization algo used: Optuna or Orion')
    parser.add_argument('-r','--runName', type=str, required=True, help='Name of the run for the output file') 
    parser.add_argument('-etaStep','--etaStep', type=float, required=False, default=1.0, help='Eta step value, default=1.0') 
    parser.add_argument('-etaStart','--etaStart', type=float, required=False, default=-4.0, help='Eta lowest value, default=-4.0') 
    parser.add_argument('-etaStop','--etaStop', type=float, required=False, default=4.0, help='Eta highest value, default=4.0') 
    args = parser.parse_args()

    optim = args.optim
    run_name = args.runName
    etaStep = args.etaStep
    etaStart = args.etaStart
    etaStop = args.etaStop

    merge(optim, run_name, etaStep, etaStart, etaStop)
 