import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

import dvi.bayes_utils as bu
from dvi.dataset import ToyDataset
from dvi.bayes_models import MLP, AdaptedMLP
from dvi.loss import GLLLoss
from sklearn.metrics import average_precision_score, roc_auc_score, auc

class PrintLogger(object):
    def __init__(self, log_path, mode='a'):
        self.terminal = sys.stdout
        self.log = open(log_path, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.log.flush()
        self.terminal.flush()
        pass

def logistic(x):
    return 1 / (1+torch.exp(-x))

def topk_accuracy(y_hat, y, k=1):
    # k = 1
    _, pred = y_hat.topk(k, 1, True, True)
    pred = pred.t()
    result = pred.eq(y.view(1, -1).expand_as(pred))

    return result[:k].view(-1)

def bhattacharya_distance(mu1, sig1, mu2, sig2):
    dist = .25*torch.log(.25*( (sig1/sig2)**2 + (sig2/sig1)**2 + 2 )) + .25*(((mu1-mu2)**2) / (sig1**2 + sig2**2) )
    return dist

def binary_crossentropy(x, y, eps=1e-7):
    loss = -y*torch.log(x + eps) - (1 - y)*torch.log(1 - x + eps)
    return loss

def f1_loss(y_pred:torch.Tensor, y_true:torch.Tensor, is_training=True) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    # f1.requires_grad = is_training
    return f1, precision, recall

def compute_performance(proba_pred, gt, tp_th=0.5, tn_th=0.5):
    proba_pred = proba_pred.squeeze()
    accurate = gt.cpu().numpy()
    errors = torch.logical_not(gt).cpu().numpy()
    proba_pred = torch.sigmoid(proba_pred).cpu().numpy()
    n_total = gt.shape[0]

    scores = {}

    accuracy = np.sum(accurate) / n_total
    scores["accuracy"] = accuracy

    if len(np.unique(accurate)) == 1:
        auc_score = 1
    else:
        auc_score = roc_auc_score(accurate, proba_pred)
    scores["auc"] = auc_score

    ap_success = average_precision_score(accurate, proba_pred)
    scores["ap_success"] = ap_success

    ap_errors = average_precision_score(errors, proba_pred)
    scores["ap_errors"] = ap_errors

    tpr = len(proba_pred[(accurate == 1) & (proba_pred > tp_th)]) / len(
            proba_pred[(accurate == 1)]
        )
    scores["tpr"] = tpr

    tnr = len(proba_pred[(errors == 1) & (proba_pred < tn_th)]) / len(
            proba_pred[(errors == 1)]
        )
    scores["tnr"] = tnr

    itvl = 0.0005
    for i,delta in enumerate(np.arange(
        proba_pred.min(),
        proba_pred.max(),
        (proba_pred.max() - proba_pred.min()) / 10000,
    )):
        tpr = len(proba_pred[(accurate == 1) & (proba_pred >= delta)]) / len(
            proba_pred[(accurate == 1)]
        )
        if i%100 == 0:
            print(f"Threshold:\t {delta:.6f}")
            print(f"TPR: \t\t {tpr:.4%}")
            print("------")

        if 0.95+itvl >= tpr >= 0.95-itvl:
            print(f"Nearest threshold 95% TPR value: {tpr:.6f}")
            print(f"Threshold 95% TPR value: {delta:.6f}")
            fpr = len(
                proba_pred[(errors == 1) & (proba_pred >= delta)]
            ) / len(proba_pred[(errors == 1)])
            scores["fpr_at_95tpr"] = fpr
            break
    return scores

def oracle_loss_crossentropy(x, y_hat, y, hyperparam=None, class_weight=[1.0,1.0], eps=1e-7):
    losses = []
    stats = []

    neg_class_coef, pos_class_coef = class_weight[0], class_weight[1]

    correct = topk_accuracy(y_hat, y)

    gt = correct.clone().detach()
    gt_float = gt.contiguous().float()
    z = torch.sigmoid(x)

    loss = -gt_float*pos_class_coef*torch.log(z + eps) - (1-gt_float)*neg_class_coef*torch.log(1-z + eps)
    loss = loss.mean()

    output = x[:,0]>0
    stats.append(x.detach())
    stats.append(correct)
    stats.append(output)
    stats.append(torch.logical_and(output, gt))
    stats.append(torch.logical_and(torch.logical_not(output), torch.logical_not(gt)))

    return loss, stats

def oracle_loss_focal(x, y_hat, y, hyperparam=[2.0], class_weight=[1.0,1.0], eps=1e-7):
    losses = []
    stats = []

    gamma = hyperparam[0]
    neg_class_coef, pos_class_coef = class_weight[0], class_weight[1]

    correct = topk_accuracy(y_hat, y)

    gt = correct.clone().detach()
    gt_float = gt.contiguous().float()
    z = torch.sigmoid(x)

    loss = -gt_float*pos_class_coef*((1-z)**gamma)*torch.log(z + eps) - (1-gt_float)*neg_class_coef*(z**gamma)*torch.log(1-z + eps)
    loss = loss.mean()

    output = x[:,0]>0
    stats.append(x.detach())
    stats.append(correct)
    stats.append(output)
    stats.append(torch.logical_and(output, gt))
    stats.append(torch.logical_and(torch.logical_not(output), torch.logical_not(gt)))

    return loss, stats

def oracle_loss_tcp(x, y_hat, y, 
            hyperparam=None, class_weight=[1.0, 1.0], eps=1e-7):
    stats = []
    correct = topk_accuracy(y_hat, y)
    gt = correct.clone().detach()
    gt_float = gt.contiguous().float()

    y_onehot = torch.FloatTensor(y_hat.size()).type(y_hat.type())
    y_onehot.zero_()
    y_onehot.scatter_(1, y.unsqueeze(1), 1)

    output = torch.logical_or(x[:,0]>0, x[:,0]<(1/1000.0)) 

    probs = F.softmax(y_hat, dim=1)
    confidence = torch.sigmoid(x).squeeze()
    
    loss = (confidence - (probs * y_onehot).sum(dim=1)) ** 2

    loss = torch.mean(loss)

    stats.append(x.detach())
    stats.append(correct)
    stats.append(output)
    stats.append(torch.logical_and(output, gt))
    stats.append(torch.logical_and(torch.logical_not(output), torch.logical_not(gt)))

    return loss, stats

def oracle_loss_steepslope(x, y_hat, y, 
            hyperparam=[1.0, 1.0, 0.0, 0.0], class_weight=[1.0, 1.0], eps=1e-7):
    losses = []
    stats = []

    alpha_neg, alpha_pos = hyperparam[0], hyperparam[1]
    beta_neg, beta_pos = hyperparam[2], hyperparam[3]

    correct = topk_accuracy(y_hat, y)

    gt = correct.clone().detach()
    gt_float = gt.contiguous().float()

    output = x[:,0]>0
    loss_neg = torch.exp(alpha_neg*(x[:,0]+beta_neg)/(1+torch.abs(x[:,0]+beta_neg))) - np.exp(-alpha_neg)
    loss_pos = torch.exp(-alpha_pos*(x[:,0]-beta_pos)/(1+torch.abs(x[:,0]-beta_pos))) - np.exp(-alpha_pos)
    loss = gt_float*loss_pos + (1-gt_float)*loss_neg

    loss = loss.mean()

    stats.append(x.detach())
    stats.append(correct)
    stats.append(output)
    stats.append(torch.logical_and(output, gt))
    stats.append(torch.logical_and(torch.logical_not(output), torch.logical_not(gt)))

    return loss, stats

def fBeta_loss(y_pred:torch.Tensor, y_true:torch.Tensor, is_training=True, beta=1.0, epsilon=1e-7) -> torch.Tensor:
    '''Calculate FBeta score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
    

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f_beta = (1+beta**2) * (precision*recall) / ((beta**2) * precision + recall + epsilon)
    return f_beta, precision, recall

class Hyperplane(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=1e-7):
        super(Hyperplane, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.temperature = temperature

    def forward(self, x, isDistance=True):
        if isDistance:
            x = self.linear(x) / (torch.pow(self.linear.weight, 2).sum().sqrt()+self.temperature)
        else:
            x = self.linear(x)
        return x

class Oracle_Main(nn.Module):
    def __init__(self,
                 backbone=None,
                 head=None):
        super(Oracle_Main, self).__init__()

        self.backbone = backbone
        if self.backbone.__class__.__name__ == 'VisionTransformer':
            self.backbone.classifier = torch.nn.Identity()
        elif self.backbone.__class__.__name__ == 'ResNet':
            self.backbone.fc = torch.nn.Identity()

        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return x