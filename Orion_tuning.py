#!/usr/bin/env python3
import sys
from pathlib import Path

import os
import yaml
import pprint
import time
import datetime
import warnings

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

from orion.client import build_experiment

srcDir = Path(__file__).resolve().parent

n_events = 10
n_trials = 100
run_name = "n"+str(n_events)+"t"+str(n_trials) + "FR" + "e34"
#run_name2 = "FR_rerunckf_optimparams_fatras-44"

etaRanges = [(3,4)] #[(-4,-3), (-3,-2), (-2,-1), (-1,0), (0,1), (1,2), (2,3), (3,4)] 
nEta = len(etaRanges)

outputDir = Path("OrionResults")
outputDir.mkdir(exist_ok=True)
#log_name = 'log-orion_'+run_name+'.txt'

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

    def __call__(
        self,
        maxSeedsPerSpM,
        cotThetaMax,
        sigmaScattering,
        radLengthPerSeed,
        impactMax,
        maxPtScattering,
        deltaRMin,
        deltaRMax,
    ):

        params = [
            maxSeedsPerSpM,
            cotThetaMax,
            sigmaScattering,
            radLengthPerSeed,
            impactMax,
            maxPtScattering,
            deltaRMin,
            deltaRMax,
        ]
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

        outputDir = Path(srcDir / "Orion_output_CKF/tmp_FR")
        outputfile = srcDir / "Orion_output_CKF/tmp_FR/performance_ckf.root"
        outputDir.mkdir(exist_ok=True)
        run_ckf(params, keys, outputDir, self.EtaMin, self.EtaMax)
        rootFile = uproot.open(outputfile)
        self.res["eff"].append(rootFile["eff_particles"].member("fElements")[0])
        self.res["fakerate"].append(rootFile["fakerate_tracks"].member("fElements")[0])
        self.res["duplicaterate"].append(
            rootFile["duplicaterate_tracks"].member("fElements")[0]
        )
        timingfile = srcDir / "Orion_output_CKF/tmp_FR/timing.tsv"
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
        self.res["runtime"].append(time_ckf + time_seeding)

        print(f'\n self.res["eff"][-1]: {self.res["eff"][-1]}')
        print(f'self.res["fakerate"][-1]: {self.res["fakerate"][-1]} ')
        print(f' self.res["duplicaterate"][-1]: {self.res["duplicaterate"][-1]}')
        print(f'\n')

        efficiency = self.res["eff"][-1]
        penalty = (
            self.res["fakerate"][-1]
            + self.res["duplicaterate"][-1] / self.k_dup
            #+ self.res["runtime"][-1] / self.k_time
        )


        return [
            {"name": "objective", "type": "objective", "value": -(efficiency - penalty)}
        ]



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
        EtaRange_data = {"eta": [EtaMin, EtaMax]} #[EtaMin,EtaMax] #"eta=["+EtaMin_name+','+EtaMax_name+"]"

        print(f'\n\n eta range: [{EtaMin, EtaMax}] \n\n')

        # Initializing the objective (score) function
        objective = Objective(k_dup, k_time, EtaMin, EtaMax)

        # Defining the parameter space

        space = {
            "maxSeedsPerSpM": "uniform(0,10,discrete=True, default_value=1)",
            "cotThetaMax": "uniform(-28.0,28.0, default_value=27.2899)",
            "sigmaScattering": "uniform(0.2,10.0, default_value=5.)",
            "radLengthPerSeed": "uniform(.001,0.1, default_value=0.1)", 
            "impactMax": "uniform(0.1,25.0, default_value=3.)", 
            "maxPtScattering": "uniform(1.0, 100.0, default_value=100.0)", 
            "deltaRMin": "uniform(0.25, 5.0, default_value=1.0)", 
            "deltaRMax": "uniform(40.0,80.0, default_value=60.0)", 
        }

        # Remove storage file if already exists (conflicts with present run if not deleted)
        if os.path.exists("./OrionResults/db_FR.pkl"):
            os.remove("./OrionResults/db_FR.pkl")

        # location to store metadata
        storage = {
            "type": "legacy",
            "database": {
                "type": "pickleddb",
                "host": "./OrionResults/db_FR.pkl",
            },
        }

        # Build new orion experiment
        experiment = build_experiment(
            "orion_new",
            space=space,
            storage=storage,
        )

        # Start Optimization
        experiment.workon(objective, max_trials=n_trials, max_broken=1000)
        
        #failed_trials = [index for index, status in enumerate(is_failed_trial) if status == 1]
        #print(f'failed_trials: {len(failed_trials)} -> {len(failed_trials)/n_trials} : {failed_trials}')

        trials_list = experiment.fetch_trials()
        #trialsByStatus_list = experiment.fetch_trials_by_status()
        trialsNC_list = experiment.fetch_noncompleted_trials()

        print(f'trials_list: {trials_list}')
        print('\n')
        #print(f'trialsByStatus_list: {trialsByStatus_list}')
        print(f'trialsNC_list: {trialsNC_list}\n')


        # fetching trials in a dataframe
        df = experiment.to_pandas()
        outputFile_trials = "trial_results"+EtaRange_filename+".txt"
        df.to_csv(outputDir / outputFile_trials)

        # Getting the best parameters
        df_imp = df[
            [
                "objective",
                "maxSeedsPerSpM",
                "cotThetaMax",
                "sigmaScattering",
                "radLengthPerSeed",
                "impactMax",
                "maxPtScattering",
                "deltaRMin",
                "deltaRMax",
            ]
        ]
        df_obj = df["objective"]
        min_obj = df_obj.min()
        df_final = df_imp[df_imp["objective"] == min_obj]
        # Retrieve the best trial
        best_trial = df_final.iloc[0]
        best_trial_data = {'best_trial': int(best_trial.name)}

        print(f'\n best_trial: {int(best_trial.name)}')


        #best_values = {'best_trial': int(best_trial.name), 
        #            'eff': eff_list[best_trial.name],
        #            'fakerate': fakerate_list[best_trial.name],
        #            'duplicaterate': duplicaterate_list[best_trial.name]
        #            }

        #print(f'\n best_values: {best_values}')

        print(f'\n df_final: {df_final}')

        outputFile = "results_orion_"+run_name+".txt"
        outputPath = outputDir / outputFile
        if df_final.empty:
            df_dict = {}
        else:
            df_final_ = df_final.drop("objective", axis=1)
            df_dict = df_final_.iloc[0].to_dict()
            df_dict['maxSeedsPerSpM'] = int(df_dict['maxSeedsPerSpM'])
        
        
        print(f'\n df_dict: {df_dict}')

        #df_final.to_csv(outputPath, mode='a', header=not os.path.exists(outputPath))
        with open(outputDir / outputFile, "a") as fp:
            json.dump(EtaRange_data, fp) #fp.write("eta={} \n".format(EtaRange_data))
            fp.write('\n')
            json.dump(df_dict, fp)
            fp.write('\n')
            json.dump(best_trial_data, fp)
            fp.write('\n\n')

    optim_time = (time.time()-start_time) /60
    print(f'Optim duration: {optim_time:.2f} min')

    # Then run ckf.py in each range with the optimized parameters:
    
    # # For debug if optimization already done and json file exists:
    # etaRanges = [ (-4,-3), (-3,-2), (-2,-1), (-1,0), (0,1), (1,2), (2,3), (3,4) ]
    # nEta = len(etaRanges)
    # outputDir = Path("OrionResults")
    outputFile = "results_orion_"+run_name+".txt"

    print('Parameter optimization done. \nRun of CKF algo with optimized params in each eta range.')


    # Read the optimized parameters from the outputed JSON file:
    eta_ranges = []
    parameters = []
    keys = ['maxSeedsPerSpM', 'cotThetaMax', 'sigmaScattering', 'radLengthPerSeed', 'impactMax', 'maxPtScattering', 'deltaRMin', 'deltaRMax']

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
            eta_range_value = json.loads(eta_line)  #json.loads(eta_line.strip().split("=")[1].strip())

            # Parse the parameter line as JSON
            parameter_values = json.loads(param_line)

            # Append the eta range and parameter values to the respective lists
            eta_ranges.append(list(eta_range_value.values())[0])
            parameters.append(list(parameter_values.values()))
            #keys = list(parameter_values.keys())

        # Print the extracted data
        print("\n Eta Ranges:", eta_ranges)
        print("\n Parameters:", parameters)

    with open(outputDir / outputFile2, "a") as fp:
        fp.write(f'-----------\n')
        for i in range(nEta):
            EtaMin = eta_ranges[i][0]
            EtaMax = eta_ranges[i][1]
            params = parameters[i]
            outputDir_eta = "Orion_output_CKF/Orion_output_CKF_"+str(EtaMin)+"_"+str(EtaMax)+"_"+run_name
            outputPath_eta = Path(srcDir / outputDir_eta )
            res = {"eff": [], "fakerate": [], "duplicaterate":[], "score": []}

            print(f'\netaRange from code: {EtaMin,EtaMax}')
            print(f'etaRange from results: {eta_ranges[i][0], eta_ranges[i][1]}')
            print(f'params: {params}')
            print('\n')
            if params != []:
                run_ckf(params, keys, outputPath_eta, EtaMin, EtaMax)
                outrootfile = outputPath_eta / "performance_ckf.root"
                print(f"outrootfile: {outrootfile}")
                rootFile = uproot.open(outrootfile)
                res["eff"].append(rootFile["eff_particles"].member("fElements")[0])
                res["fakerate"].append(rootFile["fakerate_tracks"].member("fElements")[0])
                res["duplicaterate"].append(
                    rootFile["duplicaterate_tracks"].member("fElements")[0]
                )


                res["score"].append(-(res["eff"][-1] - (res["fakerate"][-1]+res["duplicaterate"][-1])))

                print(f'During evaluation of best params:')
                print(f'\n res["eff"][-1]: {res["eff"][-1]}')
                print(f'res["fakerate"][-1]: {res["fakerate"][-1]} ')
                print(f' res["duplicaterate"][-1]: {res["duplicaterate"][-1]}')
                print(f'\n')
                
            # Write the results to the same text file
            fp.write(f'eta: {eta_ranges[i]}\n')
            fp.write(f'res: {res}\n')
            fp.write('\n')

            
    tot_time = (time.time()-start_time) /60
    print(f'Optim duration: {optim_time:.2f} min')
    print(f'Tot duration: {tot_time:.2f} min')


    if len(etaRanges) >= 3:
        # Merge the root files:
        print("Merging outputed performance_ckf.root for each eta range...")
        subprocess.call(["python3", "Examples/Scripts/Optimization/mergeRootFiles.py", "-o", "Orion", "-r", run_name])
        print("Done")


if __name__ == "__main__":
    main()
