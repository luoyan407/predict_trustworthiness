import os, sys
import torch
import torch.nn as nn
import numpy as np
import random
from model import VisionTransformer
from config import *
from checkpoint import *
from data_loaders import *
from utils import setup_device, accuracy, MetricTracker, TensorboardWriter

from oracle import *
from train_oracle import *
from risk_control import risk_control

import torchvision.models as models

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.legend_handler import HandlerLine2D
from matplotlib import rc
rc('font', **{'family':'serif','serif':['Times New Roman'], 'size':12})
rc('text', usetex=True)

def mean(lst): 
    return sum(lst) / len(lst) 

def main():
    show_legend = False
    labels = ['CE', 'Focal', 'TCP', 'SS']
    colors = ['blue', 'black', 'green', 'red']
    rstart = 0.18
    delta = 0.001

    input_npz = ['/path-to-ovit_cvit_ce_experiment_folder/results/val_ep01_oracle_pred.npz',
        '/path-to-ovit_cvit_focal_experiment_folder/results/val_ep01_oracle_pred.npz',
        '/path-to-ovit_cvit_tcp_experiment_folder/results/val_ep01_oracle_pred.npz',
        '/path-to-ovit_cvit_steep_experiment_folder/results/val_ep01_oracle_pred.npz']
    output_svg = 'data4paper/risk/risk_vit_vit.svg'
    show_legend = True
    rstart = 0.18

    # input_npz = ['/path-to-ovit_crsn_ce_experiment_folder/results/val_ep01_oracle_pred.npz',
    #     '/path-to-ovit_crsn_focal_experiment_folder/results/val_ep01_oracle_pred.npz',
    #     '/path-to-ovit_crsn_tcp_experiment_folder/results/val_ep01_oracle_pred.npz',
    #     '/path-to-ovit_crsn_steep_experiment_folder/results/val_ep01_oracle_pred.npz']
    # output_svg = 'data4paper/risk/risk_vit_rsn.svg'
    # rstart = 0.4

    all_risk, all_coverage = [], []
    for i in range(len(input_npz)):
        npzfile = np.load(input_npz[i])

        feat = npzfile['oracle_pred']
        gt = npzfile['oracle_gt']
        feat_tensor = torch.from_numpy(feat).cuda()
        gt_labels = torch.from_numpy(gt).cuda()

        x = 1/(1+np.exp(-feat))
        kappa = np.max(x,1)
        residuals = np.logical_not(gt).astype(int)
        bound_cal = risk_control()

        [theta, b_star, risks, coverages] = bound_cal.bound(rstart,delta,kappa,residuals, split=False)
        all_risk.append(risks)
        all_coverage.append(coverages)

    plt.figure(figsize=(12,7))
    ax = plt.subplot()

    for i in range(len(input_npz)):
        plt.plot(all_coverage[i], all_risk[i], label=labels[i], color=colors[i], linestyle='-', linewidth=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(True, linestyle='--')

    plt.xlabel('coverage', fontsize=45)
    plt.ylabel(r'selective risk (\%)', fontsize=45)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    if show_legend:
        plt.legend(loc=0,fontsize=25)

    plt.savefig(output_svg, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()