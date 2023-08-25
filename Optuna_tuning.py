#!/usr/bin/env python3
import sys
from pathlib import Path

import os
import yaml
import pprint
import time
import datetime
import warnings

import optuna
import logging
import uproot

import pathlib
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import random
import subprocess
import multiprocessing
import numpy as np
import json
import array
import sys
import argparse
import re
import pandas as pd

from typing import Optional, Union
from optuna.samplers import TPESampler

n_events = 10
n_trials = 100
run_name = "n"+str(n_events)+"t"+str(n_trials) + "FR" + "e12" 
#run_name2 = "FR_rerunckf_optimparams_fatras-44"

etaRanges = [(1,2)]#[(-4,-3), (-3,-2), (-2,-1), (-1,0), (0,1), (1,2), (2,3), (3,4)]
nEta = len(etaRanges)

outputDir = Path("OptunaResults")
outputDir.mkdir(exist_ok=True)
#log_name = 'log-optuna_'+run_name+'.txt'

from acts import UnitConstants as u

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

srcDir = Path(__file__).resolve().parent


def run_ckf(params, names, outDir, EtaMin=-4.0, EtaMax=4.0):

    if len(params) != len(names):
        raise Exception("Length of Params must equal names")

    ckf_script = srcDir / "ckf.py"
    nevts = "--nEvents=" + str(n_events)
    indir = "--indir=" + str(srcDir)
    outdir = "--output=" + str(outDir)
    etamin = "--etaMin=" + str(EtaMin)
    etamax = "--etaMax=" + str(EtaMax)

    ret = ["python3"]
    ret.append(ckf_script)
    ret.append(nevts)
    ret.append(indir)
    ret.append(outdir)
    ret.append(etamin)
    ret.append(etamax)

    i = 0
    for param in params:
        arg = "--sf_" + names[i] + "=" + str(param)
        ret.append(arg)
        i += 1

    # Run CKF for the given parameters
    subprocess.call(ret)


class Objective:
    def __init__(self, k_dup, k_time, EtaMin, EtaMax):
        self.res = {
            "eff": [],
            "fakerate": [],
            "duplicaterate": [],
            "runtime": [],
        }

        self.k_dup = k_dup
        self.k_time = k_time
        self.EtaMin = EtaMin
        self.EtaMax = EtaMax

    def __call__(self, trial):
        params = []

        maxSeedsPerSpM = trial.suggest_int("maxSeedsPerSpM", 0, 10)
        params.append(maxSeedsPerSpM)
        cotThetaMax = trial.suggest_float("cotThetaMax", -28.0, 28.0)
        params.append(cotThetaMax)
        sigmaScattering = trial.suggest_float("sigmaScattering", 0.2, 10)
        params.append(sigmaScattering)
        radLengthPerSeed = trial.suggest_float("radLengthPerSeed", 0.001, 0.1)
        params.append(radLengthPerSeed)
        impactMax = trial.suggest_float("impactMax", 0.1, 25)
        params.append(impactMax)
        maxPtScattering = trial.suggest_float("maxPtScattering", 1, 100)
        params.append(maxPtScattering)
        deltaRMin = trial.suggest_float("deltaRMin", 0.25, 5)
        params.append(deltaRMin)
        deltaRMax = trial.suggest_float("deltaRMax", 40, 80)
        params.append(deltaRMax)
        keys = [
            "maxSeedsPerSpM",
            "cotThetaMax",
            "sigmaScattering",
            "radLengthPerSeed",
            "impactMax",
            "maxPtScattering",
            "deltaRMin",
            "deltaRMax",
        ]

        outputDir = Path(srcDir / "Optuna_output_CKF/tmp_FR")
        outputfile = srcDir / "Optuna_output_CKF/tmp_FR/performance_ckf.root"
        outputDir.mkdir(exist_ok=True)
        run_ckf(params, keys, outputDir, self.EtaMin, self.EtaMax)
        rootFile = uproot.open(outputfile)
        self.res["eff"].append(float(rootFile["eff_particles"].member("fElements")[0]))
        self.res["fakerate"].append(float(rootFile["fakerate_tracks"].member("fElements")[0]))
        self.res["duplicaterate"].append(
            float(rootFile["duplicaterate_tracks"].member("fElements")[0])
        )

        timingfile = srcDir / "Optuna_output_CKF/tmp_FR/timing.tsv"
        timing = pd.read_csv(timingfile, sep="\t")
        time_ckf = float(
            timing[timing["identifier"].str.match("Algorithm:TrackFindingAlgorithm")][
                "time_perevent_s"
            ]
        )
        time_seeding = float(
            timing[timing["identifier"].str.match("Algorithm:SeedingAlgorithm")][
                "time_perevent_s"
            ]
        )
        self.res["runtime"].append(float(time_ckf + time_seeding))

        efficiency = self.res["eff"][-1]
        penalty = (
            self.res["fakerate"][-1]
            + self.res["duplicaterate"][-1] / self.k_dup
            #+ self.res["runtime"][-1] / self.k_time
        )

        print(f'\n efficiency: {efficiency}')
        print(f'\n  penalty: \
                \n self.res["fakerate"][-1]: {self.res["fakerate"][-1]} \
                \n self.res["duplicaterate"][-1]: {self.res["duplicaterate"][-1]} \
                \n self.res["runtime"][-1]: {self.res["runtime"][-1]}\n')

        return efficiency - penalty


def main():

    k_dup = 7
    k_time = 7

    start_time = time.time()

    # Define eta ranges. The optimization is performed for each eta range indepently 
    # The optimized parameters for each range are saved in filenames ending with '_eta-1_1' for range (-1,1)

    for i in range(nEta):
        EtaMin = etaRanges[i][0]
        EtaMax = etaRanges[i][1]
        EtaMin_name = str(EtaMin)
        EtaMax_name = str(EtaMax)
        EtaRange_filename = "_eta"+EtaMin_name+"_"+EtaMax_name+"_"+run_name
        EtaRange_data = {"eta": [EtaMin, EtaMax]} #[EtaMin,EtaMax] #{"eta": [EtaMin, EtaMax]} #"eta=["+EtaMin_name+','+EtaMax_name+"]"

        print(f'\n\n eta range: [{EtaMin, EtaMax}] \n\n')

        # Initializing the objective (score) function
        objective = Objective(k_dup, k_time, EtaMin, EtaMax)

        start_values = {
            "maxSeedsPerSpM": 1,
            "cotThetaMax": 27.2899,
            "sigmaScattering": 5.,
            "radLengthPerSeed": 0.1,
            "impactMax": 3.,
            "maxPtScattering": 100.0,
            "deltaRMin": 1.0,
            "deltaRMax": 60.0
    }


        # Optuna logger
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = str(outputDir)+"/test_study"+EtaRange_filename
        storage_name = "sqlite:///{}.db".format(study_name)
        sampler = TPESampler(seed=10) #for reproducibility

        # creating a new optuna study
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///{}.db".format(study_name),
            direction="maximize",
            load_if_exists=True,
            sampler=sampler,
        )

        study.enqueue_trial(start_values)
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        # Start Optimization
        study.optimize(objective, n_trials=n_trials)

        # Printout the best trial values
        print("Best Trial until now", flush=True)
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}", flush=True)

        best_values = {
        "best_trial": study.best_trial.number,
        "efficiency": objective.res["eff"][study.best_trial.number],
        "fakerate": objective.res["fakerate"][study.best_trial.number],
        "duplicaterate": objective.res["duplicaterate"][study.best_trial.number]
        }

        print(f"\n best_values: {best_values}")

        outputFile = "results_optuna_"+run_name+".txt" 
        with open(outputDir / outputFile, "a") as fp:
            #fp.write("eta={} \n".format(EtaRange_data))
            json.dump(EtaRange_data, fp)
            fp.write('\n')
            json.dump(study.best_params, fp)
            fp.write('\n')
            json.dump(best_values, fp)
            fp.write('\n\n') 
            #json.dump(study.best_params, fp)

    optim_time = (time.time()-start_time) /60
    print(f'Optim duration: {optim_time:.2f} min')

    # Then run ckf.py in each range with the optimized parameters:

    # # For debug if optimization already done and json file exists
    # etaRanges = [ (-4,-3), (-3,-2), (-2,-1), (-1,0), (0,1), (1,2), (2,3), (3,4) ]
    # nEta = len(etaRanges)
    # outputDir = Path("OptunaResults")
    # outputFile = "results_optuna_"+run_name+".txt" 


    print('Parameter optimization done. \nRun of CKF algo with optimized params in each eta range.')


    # Read the optimized parameters from the outputed JSON file:
    eta_ranges = []
    parameters = []

    with open(outputDir / outputFile, "r") as fp:
        content = fp.read()
        ranges = content.strip().split("\n\n")

        # Process each range separately
        for eta_range in ranges:
            # Split the range into eta line and parameter line
            lines = eta_range.strip().split("\n")
            eta_line = lines[0]
            param_line = lines[1]

            # Extract the eta range from the eta line
            eta_range_value = json.loads(eta_line) #json.loads(eta_line.strip().split("=")[1].strip())

            # Parse the parameter line as JSON
            parameter_values = json.loads(param_line)

            # Append the eta range and parameter values to the respective lists
            eta_ranges.append(list(eta_range_value.values())[0])
            parameters.append(list(parameter_values.values()))
            keys = list(parameter_values.keys())

        # Print the extracted data
        print("\n Eta Ranges:", eta_ranges)
        print("\n Parameters:", parameters)


    for i in range(nEta):
        EtaMin = eta_ranges[i][0]
        EtaMax = eta_ranges[i][1]
        params = parameters[i]
        outputDir_eta = "Optuna_output_CKF/Optuna_output_CKF_"+str(EtaMin)+"_"+str(EtaMax)+"_"+run_name
        outputPath_eta = Path(srcDir / outputDir_eta )
    

        print(f'\netaRange from code: {EtaMin,EtaMax}')
        print(f'etaRange from results: {eta_ranges[i][0], eta_ranges[i][1]}')
        print(f'params: {params}')
        print('\n')

        run_ckf(params, keys, outputPath_eta, EtaMin, EtaMax)

    tot_time = (time.time()-start_time) /60
    print(f'Optim duration: {optim_time:.2f} min')
    print(f'Tot duration: {tot_time:.2f} min')

    # Read log file and count number of finished/improved/failed trials
    finished_trials = []
    improved_trials = []
    failed_trials = []
    outputLogFile = 'log-optuna_' + run_name + '.txt'
    with open(outputDir / outputLogFile, "r") as f:
        content = f.read()
        lines = content.strip().split("\n")

    def parse_log_file(log_file):
        #finished_trials = set()
        #improved_trials = set()
        #failed_trials = set()
        trial_stats = {}

        eta_range_pattern = re.compile(r'eta range: \[(.*?)\]')
        finished_pattern = re.compile(r'Trial (\d+) finished')
        improved_pattern = re.compile(r'Best is trial (\d+)')
        failed_pattern = re.compile(r'Trial (\d+) failed')

        with open(log_file, 'r') as file:
            current_eta_range = None

            for line in file:
                line = line.strip()
                # Check if the line indicates an eta range
                eta_range_match = eta_range_pattern.search(line)
                if eta_range_match:
                    eta_range = eval(eta_range_match.group(1))
                    current_eta_range = eta_range
                    trial_stats[current_eta_range] = {
                        'finished_trials': set(),
                        'improved_trials': set(),
                        'failed_trials': set()
                    }

                # Check if the line indicates a finished trial
                finished_match = finished_pattern.search(line)
                if finished_match and current_eta_range is not None:
                    trial_number = int(finished_match.group(1))
                    trial_stats[current_eta_range]['finished_trials'].add(trial_number)

                # Check if the line indicates an improved trial
                improved_match = improved_pattern.search(line)
                if improved_match and current_eta_range is not None:
                    trial_number = int(improved_match.group(1))
                    trial_stats[current_eta_range]['improved_trials'].add(trial_number)

                # Check if the line indicates a failed trial
                failed_match = failed_pattern.search(line)
                if failed_match and current_eta_range is not None:
                    trial_number = int(failed_match.group(1))
                    trial_stats[current_eta_range]['failed_trials'].add(trial_number)

        return trial_stats

    # Usage
    trial_stats = parse_log_file(outputDir / outputLogFile)
    print(f'\n\n| TRIAL-STATS out of {n_trials} trials:')

    for eta_range, stats in trial_stats.items():
        finished_trials = stats['finished_trials']
        improved_trials = sorted(stats['improved_trials'])
        failed_trials = stats['failed_trials']

        print(f'\n| eta: {eta_range}')
        print(f'|Improved trials: {len(improved_trials)} -> {len(improved_trials)*100/n_trials}% : {improved_trials}')
        print(f'|Finished trials: {len(finished_trials)} -> {len(finished_trials)*100/n_trials}%')
        print(f'|Failed trials: {len(failed_trials)} -> {len(failed_trials)*100/n_trials}%')
        
        if len(finished_trials) > len(failed_trials):
            print(f'Failed trial numbers: {failed_trials}')
        else:
            print(f'Finished trial numbers: {finished_trials}')


    # Merge the root files:
    if len(etaRanges) >= 3:
        print("\n\nMerging outputed performance_ckf.root for each eta range...")
        subprocess.call(["python3", "Examples/Scripts/Optimization/mergeRootFiles.py", "-o", "Optuna", "-r", run_name])
        print("Done")
    

if __name__ == "__main__":
    main()


