#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
July 2023
@author: anaelle chalumeau
"""

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

n_events = 10
run_name = "n"+str(n_events)+"e-44"+"_fatras-subranges"

#etaStep = 0.1
etaStart = -4.0
etaStop = 4.0
#etaRanges = [(round(x, 1), round(x + etaStep, 1)) for x in [i * etaStep for i in range(int(etaStart / etaStep), int(etaStop / etaStep))]] 
etaRanges = [(-4,-3), (-3,-2), (-2,-1), (-1,0), (0,1), (1,2), (2,3), (3,4)]
nEta = len(etaRanges)

srcDir = Path(__file__).resolve().parent

#if etaStep < 1:
#    outputDir = Path(f"/Users/achalume/alice/actsdir/test-ckf/output/n10/subruns/step,{str(etaStep)[2:]}/")
#else:
#    outputDir = Path(f"/Users/achalume/alice/actsdir/test-ckf/output/n10/subruns/step{etaStep}/")

outputDir = Path(f"/Users/achalume/alice/actsdir/test-ckf/output/test-fatras/fatras-subranges/")
outputDir.mkdir(exist_ok=True)


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


def main():

    default_param_dict = {"maxSeedsPerSpM": 1, 
                            "cotThetaMax": 27.2899,
                            "sigmaScattering": 5.0,
                            "radLengthPerSeed": 0.1,
                            "impactMax": 3.0,
                            "maxPtScattering": 100.0,
                            "deltaRMin": 1.0,
                            "deltaRMax": 60.0}
    keys = list(default_param_dict.keys())

    # Create .txt file with default parameters in each eat range, to be read and run by ckf.py after
    for i in range(nEta):
        EtaMin = etaRanges[i][0]
        EtaMax = etaRanges[i][1]
        EtaMin_name = str(EtaMin)
        EtaMax_name = str(EtaMax)
        EtaRange_filename = "_eta"+EtaMin_name+"_"+EtaMax_name+"_"+run_name
        EtaRange_data = {"eta": [EtaMin, EtaMax]} #[EtaMin,EtaMax] #"eta=["+EtaMin_name+','+EtaMax_name+"]"

        outputFile = "results_ckf_default"+run_name+".txt"
        outputPath = outputDir / outputFile
        
        with open(outputDir / outputFile, "a") as fp:
            json.dump(EtaRange_data, fp) 
            fp.write('\n')
            json.dump(default_param_dict, fp)
            fp.write('\n\n')


    # Then run ckf.py in each range with the written parameters:

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


    with open(outputDir / outputFile, "a") as fp:
        fp.write(f'-----------\n')
        for i in range(nEta):
            EtaMin = eta_ranges[i][0]
            EtaMax = eta_ranges[i][1]
            params = parameters[i]
            outputDir_eta = "ckf_default_"+str(EtaMin)+"_"+str(EtaMax)+"_"+run_name
            outputPath_eta = Path(outputDir / outputDir_eta )
            outputPath_eta.mkdir(exist_ok=True)
            #outputPath_eta = Path(outputDir / subDir )
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

    # Merge the root files:
    print("Merging outputed performance_ckf.root for each eta range...")
    #subprocess.call(["python3", "Examples/Scripts/Optimization/mergeRootFiles.py", "-o", "ckf", "-r", run_name,
    #                "-etaStep", str(etaStep) , "-etaStart", str(etaStart), "-etaStop", str(etaStop)])

    subprocess.call(["python3", "Examples/Scripts/Optimization/mergeRootFiles.py", "-o", "ckf", "-r", run_name,
                     "-etaStart", str(etaStart), "-etaStop", str(etaStop)])

    print("Done")
    


if __name__ == "__main__":
    main()
