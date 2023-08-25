#!/usr/bin/env python3
"""
August 2023
@author: anaelle chalumeau
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import uproot

path = "/Users/achalume/alice/actsdir/acts/bin/output/python/again/"
dir = "ckfC_13_npart10_nEvents10_noseedconf_ambi_infl100_nhits7_2Confs/"

tree_name = "estimatedparams"
file = uproot.open(path+dir+tree_name+".root")
tree = file[tree_name]
true_param_names = ["t_loc0", "t_loc1", "t_phi", "t_theta", "t_qop", "t_time", "t_eta", "t_cotT"]
# t_loc0 = []
# t_loc1 = []
# t_phi = []
# t_theta = []
# t_qop = []
# t_time = []
# true_stats = []
true_param_lists = []#[t_loc0, t_loc1, t_phi, t_theta, t_qop, t_time]

# Initialize lists to store parameter values
p1 = []
p2 = []
p3 = []
p4 = []
p5 = []
p6 = []

# Regular expression pattern to match the PropagatorError line
pattern = r"Propagation reached the configured maximum number of steps with the initial parameters:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"

# Read the log file
# log_file = path+dir+"log.txt"
log_file = path+dir+"log.txt"

with open(log_file, "r") as f:
    lines = f.read()

# Find all matches of the pattern
matches = re.findall(pattern, lines)

# Loop through matches and extract parameter values
for match in matches:
    p1.append(float(match[0]))
    p2.append(float(match[1]))
    p3.append(float(match[2]))
    p4.append(float(match[3]))
    p5.append(float(match[4]))
    p6.append(float(match[5]))

p4_eta = -np.log(np.tan(np.array(p4)/2))
p4_cotT = 1/(np.tan(np.array(p4)))

# Calculate statistics for each parameter
failed_param_lists = [p1, p2, p3, p4, p5, p6, p4_eta, p4_cotT]
failed_stats = [(min(p), max(p), np.mean(p), np.var(p)) for p in failed_param_lists]

# Parameter names
failed_param_names = ["p1", "p2", "p3", "p4", "p5", "p6"]

# Create subplots for parameter distributions
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
fig.suptitle(f"Parameter distributions \n #failed seeds: {len(p1)}", fontsize=10)

for i,name in enumerate(true_param_names):

    if name != 't_eta' and name != 't_cotT':
        true_param_values = list(tree[true_param_names[i]].array())
        true_param_lists.append(list(true_param_values))

    if name == 't_theta':
        t_eta = -np.log(np.tan(np.array(true_param_values)/2))
        t_cotT = 1/(np.tan(np.array(true_param_values)))

true_param_lists.append(list(t_eta))
true_param_lists.append(list(t_cotT))

true_stats = [(min(p), max(p), np.mean(p), np.var(p)) for p in true_param_lists]

for i, ax in enumerate(axs.flatten()):

    failed_stats_text = (
        f"Failed seeds\n"
        f"max={failed_stats[i][1]:.2f}\n"
        f"min={failed_stats[i][0]:.2f}\n"
        fr"$\mu$={failed_stats[i][2]:.2f}"f"\n"
        fr"$\sigma$={np.sqrt(failed_stats[i][3]):.2f}"
    )

    true_stats_text = (
        f"Non failed\n"
        f"max={true_stats[i][1]:.2f}\n"
        f"min={true_stats[i][0]:.2f}\n"
        fr"$\mu$={true_stats[i][2]:.2f}"f"\n"
        fr"$\sigma$={np.sqrt(true_stats[i][3]):.2f}"
    )

    ax.hist(failed_param_lists[i], bins=50, histtype="step", label=failed_stats_text, density=True)
    ax.hist(true_param_lists[i], bins=50, alpha=0.5, histtype='step', label=true_stats_text, density=True)

    ax.set_title(true_param_names[i], fontsize=10)
    

    # ax.text(0.89, 0.50, failed_stats_text, transform=ax.transAxes, va="top", ha="right", fontsize=8, color='#1f77b4') 
    # ax.text(0.22, 0.50, true_stats_text, transform=ax.transAxes, va="top", ha="right", fontsize=8, color='#ff7f0e')
    ax.legend(fontsize=6)


plt.tight_layout()
plt.savefig(path+dir+"params-distrib.png")
plt.show()

t_loc0 = true_param_lists[0]
t_loc1 = true_param_lists[1]
t_phi = true_param_lists[2]
t_theta = true_param_lists[3]
t_qop = true_param_lists[4]
t_time = true_param_lists[5]
t_eta = true_param_lists[6]
t_cotT = true_param_lists[7]

true_data = {
    "t_loc0": t_loc0,
    "t_loc1": t_loc1,
    "t_phi": t_phi,
    "t_theta": t_theta,
    "t_qop": t_qop, 
    "t_time": t_time,
    "t_eta": t_eta,
    "t_cotT": t_cotT
}
true_df = pd.DataFrame(true_data)

sns.pairplot(true_df, plot_kws={'s': 1}, diag_kws={'bins': 50}, corner=True)
output_filename = "covariances_estimatedparams.png"
plt.savefig(path+dir+output_filename)
#plt.show()

failed_data = {
    "p1": p1,
    "p2": p2,
    "p3": p3,
    "p4": p4,
    "p5": p5, 
    "p6": p6,
    "p4_eta": p4_eta,
    "p4_cotT": p4_cotT
}
failed_df = pd.DataFrame(failed_data)

sns.pairplot(failed_df, plot_kws={'s': 1}, diag_kws={'bins': 50}, corner=True)
output_filename = "covariances_failedseeds.png"
plt.savefig(path+dir+output_filename)
#plt.show()