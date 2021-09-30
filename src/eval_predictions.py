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

import torchvision.models as models

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def mean(lst): 
    return sum(lst) / len(lst) 

def main():
    tn_th = 0.5 # if evaluating TCP's predictions, tn_th = 1.0/1000, otherwise 0.5
    input_npz = '/path-to-the-experiment-folder/results/val_ep01_oracle_pred.npz'

    npzfile = np.load(input_npz)

    feat = npzfile['oracle_pred']
    gt = npzfile['oracle_gt']
    feat_tensor = torch.from_numpy(feat).cuda()
    gt_labels = torch.from_numpy(gt).cuda()

    print('===> final performance')
    final_perf = compute_performance(feat_tensor, gt_labels, tn_th=tn_th)
    print(final_perf)
    print('<===') 

    print(f' & Acc & FPR-95%-TPR & AURP-Error & AURP-Success & AUC & TPR & TNR \\\\') 
    print(f" & {final_perf['accuracy']*100:02.2f} & {final_perf['fpr_at_95tpr']*100:02.2f} & {final_perf['ap_errors']*100:02.2f} & {final_perf['ap_success']*100:02.2f} & {final_perf['auc']*100:02.2f} & {final_perf['tpr']*100:02.2f} & {final_perf['tnr']*100:02.2f} \\\\")  


    pos_feats = torch.sigmoid(feat_tensor)[gt_labels]
    neg_feats = torch.sigmoid(feat_tensor)[gt_labels==False]
    mu_pos = torch.mean(pos_feats)
    mu_neg = torch.mean(neg_feats)
    std_pos = torch.std(pos_feats)
    std_neg = torch.std(neg_feats)
    Bhattachaya_dist = lambda mu1, sig1, mu2, sig2: .25*torch.log(.25*( (sig1/sig2)**2 + (sig2/sig1)**2 + 2 )) + .25*(((mu1-mu2)**2) / (sig1**2 + sig2**2) )
    b_dist = Bhattachaya_dist(mu_pos,std_pos,mu_neg,std_neg)
    b_coef = 1/torch.exp(b_dist)
    # print(f'Bhattachaya-D: {b_dist.item():.4f}')
    kl_div = lambda mu1, sig1, mu2, sig2: torch.log(sig2/sig1) + (sig1**2 + (mu1-mu2)**2)/(sig2**2) - 0.5
    div_pos_neg = kl_div(mu_pos,std_pos,mu_neg,std_neg)
    div_neg_pos = kl_div(mu_neg,std_neg,mu_pos,std_pos)
    dist = (div_pos_neg + div_neg_pos)/2
    print(f'kl div, Bhattachaya distance')
    print(f' & {dist.item():.4f} & {b_dist.item():.4f} \\\\')


if __name__ == '__main__':
    main()