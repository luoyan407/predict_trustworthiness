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

import torchvision.models as models

# import torch
import argparse
# import sys
# import os
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.datasets import ImageNet
from torchvision import transforms

import sys
sys.path.insert(1, '../ReColorAdv')

# mister_ed
from recoloradv.mister_ed import loss_functions as lf
from recoloradv.mister_ed import adversarial_training as advtrain
from recoloradv.mister_ed import adversarial_perturbations as ap 
from recoloradv.mister_ed import adversarial_attacks as aa
from recoloradv.mister_ed import spatial_transformers as st
from recoloradv.mister_ed.utils import pytorch_utils as utils

# ReColorAdv
from recoloradv import perturbations as pt
from recoloradv import color_transformers as ct
from recoloradv import color_spaces as cs

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    config = get_train_config()

    training_log = os.path.join(config.result_dir, 'eval.log')
    sys.stdout = PrintLogger(training_log, 'w')

    model = None
    if config.classifier == 'transformer':
        model = VisionTransformer(
                 image_size=(config.image_size, config.image_size),
                 patch_size=(config.patch_size, config.patch_size),
                 emb_dim=config.emb_dim,
                 mlp_dim=config.mlp_dim,
                 num_heads=config.num_heads,
                 num_layers=config.num_layers,
                 num_classes=config.num_classes,
                 attn_dropout_rate=config.attn_dropout_rate,
                 dropout_rate=config.dropout_rate)
        # load checkpoint
        if config.checkpoint_path:
            state_dict = load_checkpoint(config.checkpoint_path)
            if config.num_classes != state_dict['classifier.weight'].size(0):
                del state_dict['classifier.weight']
                del state_dict['classifier.bias']
                print("re-initialize fc layer")
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict)
            print("Load pretrained weights from {}".format(config.checkpoint_path))
    elif config.classifier == 'resnet':
        model = resnet50(pretrained=True, progress=True)
    elif config.classifier == 'efficientnet':
        model = EfficientNet.from_pretrained("efficientnet-b0")


    normalizer = utils.DifferentiableNormalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

    dataset = ImageNet(
        os.path.join(config.data_dir, config.dataset),
        split='val',
        transform=transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
        ]),
    )
    val_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    #--------create and load oracle--------
    print("create oracle model")
    oracle_config = eval("get_{}_config".format(config.oracle_model_arch))(config)
    if config.oracle_type == 'transformer':
        oracle_backbone = VisionTransformer(
                 image_size=(oracle_config.image_size, oracle_config.image_size),
                 patch_size=(oracle_config.patch_size, oracle_config.patch_size),
                 emb_dim=oracle_config.emb_dim,
                 mlp_dim=oracle_config.mlp_dim,
                 num_heads=oracle_config.num_heads,
                 num_layers=oracle_config.num_layers,
                 num_classes=oracle_config.oracle_outdim,
                 attn_dropout_rate=oracle_config.attn_dropout_rate,
                 dropout_rate=oracle_config.dropout_rate)

        # load checkpoint
        if config.oracle_checkpoint_path:
            state_dict = load_checkpoint(config.oracle_checkpoint_path)
            if config.oracle_outdim != state_dict['classifier.weight'].size(0):
                del state_dict['classifier.weight']
                del state_dict['classifier.bias']
                print("re-initialize oracle fc layer")
                oracle_backbone.load_state_dict(state_dict, strict=False)
            else:
                oracle_backbone.load_state_dict(state_dict)
            print("Load pretrained oracle weights from {}".format(config.oracle_checkpoint_path))
    elif config.oracle_type == 'resnet':
        oracle_backbone = models.resnet50(pretrained=True)

    tn_th = 0.5
    if config.oracle_loss == 'ce':
        oracle_loss = oracle_loss_crossentropy
        oracle_head = nn.Linear(config.oracle_feat_dim, config.oracle_outdim)
    elif config.oracle_loss == 'focal':
        oracle_loss = oracle_loss_focal
        oracle_head = nn.Linear(config.oracle_feat_dim, config.oracle_outdim)
    elif config.oracle_loss == 'tcp':
        oracle_loss = oracle_loss_tcp
        oracle_head = nn.Linear(config.oracle_feat_dim, config.oracle_outdim)
        tn_th = 1.0/config.num_classes
    elif config.oracle_loss == 'steep':
        oracle_loss = oracle_loss_steepslope
        oracle_head = Hyperplane(config.oracle_feat_dim, config.oracle_outdim)

    oracle = Oracle_Main(oracle_backbone, oracle_head)

    if config.oracle_pretrained is not None:
        state_dict = load_pretrained_d_oracle(config.oracle_pretrained)
        oracle.load_state_dict(state_dict)
    oracle.eval()
    if torch.cuda.is_available():
        oracle.cuda()
    
    loss_param = config.oracle_loss_hyperparam
    class_weight = config.oracle_class_weight
    if class_weight is None:
        class_weight = [1.0, 1.0]
    hyper_param = [loss_param, class_weight]

    epoch = 1
    stats_npz = 'oracle_pred.npz'
    train_oracle_npz = os.path.join(config.result_dir, f'train_ep{epoch:02d}_{stats_npz}')
    stats_npz = os.path.join(config.result_dir, f'val_ep{epoch:02d}_{stats_npz}')

    cw_loss = lf.CWLossF6(model, normalizer, kappa=float('inf'))
    perturbation_loss = lf.PerturbationNormLoss(lp=2)
    adv_loss = lf.RegularizedLoss(
        {'cw': cw_loss, 'pert': perturbation_loss},
        {'cw': 1.0, 'pert': 0.05},
        negate=True,
    )

    pgd_attack = aa.PGD(
        model,
        normalizer,
        ap.ThreatModel(pt.ReColorAdv, {
            'xform_class': ct.FullSpatial,
            'cspace': cs.CIELUVColorSpace(),
            'lp_style': 'inf',
            'lp_bound': 0.06,
            'xform_params': {
              'resolution_x': 16,
              'resolution_y': 32,
              'resolution_z': 32,
            },
            'use_smooth_loss': True,
        }),
        adv_loss,
    )

    head_str = f'batch, acc1, loss_oracle, tpr, tnr'
    print(head_str)

    losses = []
    acc1s = []
    losses_oracle = []

    all_oracle_pred = torch.empty(0).cuda()
    all_oracle_gt = torch.empty(0, dtype=torch.bool).cuda()
    all_oracle_mask = torch.empty(0, dtype=torch.bool).cuda()
    all_oracle_tpr = torch.empty(0, dtype=torch.bool).cuda()
    all_oracle_tnr = torch.empty(0, dtype=torch.bool).cuda()

    eps = 1e-7
    for batch_idx, (inputs, labels) in enumerate(val_loader):

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        adv_inputs = pgd_attack.attack(
            inputs,
            labels,
            optimizer=optim.Adam,
            optimizer_kwargs={'lr': 0.001},
            signed=False,
            verbose=False,
            num_iterations=(100, 300),
        ).adversarial_tensors()
        with torch.no_grad():
            adv_imgs = normalizer(adv_inputs)
            adv_logits = model(adv_imgs)

            rz_adv_imgs = adv_imgs
            if model.__class__.__name__ == 'VisionTransformer' and oracle.backbone.__class__.__name__ == 'ResNet':
                rz_adv_imgs = F.interpolate(rz_adv_imgs, size=224)
            elif model.__class__.__name__ == 'ResNet' and oracle.backbone.__class__.__name__ == 'VisionTransformer':
                rz_adv_imgs = F.interpolate(rz_adv_imgs, size=384)

            batch_oracle_pred = oracle(rz_adv_imgs)
            batch_pred_detach = adv_logits.detach()
            loss_oracle, states_oracle = oracle_loss(batch_oracle_pred, batch_pred_detach, labels, hyperparam=hyper_param[0], class_weight=hyper_param[1])

        acc1, acc5 = accuracy(adv_logits.detach(), labels, topk=(1, 5))
        acc1s.append(acc1.item())

        all_oracle_pred = torch.cat((all_oracle_pred, states_oracle[0]), dim=0)
        all_oracle_gt = torch.cat((all_oracle_gt, states_oracle[1]), dim=0)
        all_oracle_mask = torch.cat((all_oracle_mask, states_oracle[2]), dim=0)
        all_oracle_tpr = torch.cat((all_oracle_tpr, states_oracle[3]), dim=0)
        all_oracle_tnr = torch.cat((all_oracle_tnr, states_oracle[4]), dim=0)

        losses_oracle.append(loss_oracle.item())

        tpr = all_oracle_tpr.sum().item()/(all_oracle_gt.sum().item()+eps) * 100
        tnr = all_oracle_tnr.sum().item()/(torch.logical_not(all_oracle_gt).sum().item()+eps) * 100

        if batch_idx % 10 == 0:
            out_str = f'{batch_idx:05d}/{len(val_loader):05d}, {np.mean(acc1s):.4f}, {np.mean(losses_oracle):.4f}, {tpr:.2f}, {tnr:.2f}'
            print(out_str)

    acc1 = np.mean(acc1s)
    losses_oracle_avg = np.mean(losses_oracle)

    tpr = all_oracle_tpr.sum().item()/all_oracle_gt.sum().item() * 100
    tnr = all_oracle_tnr.sum().item()/torch.logical_not(all_oracle_gt).sum().item() * 100

    out_str = f'{np.mean(acc1s):.4f}, {np.mean(losses_oracle):.4f}, {tpr:.2f}, {tnr:.2f}'
    print('overall perfs')
    print(out_str)

    final_perf = compute_performance(all_oracle_pred, all_oracle_gt, tn_th=tn_th)
    print('final performance')
    print(final_perf)

    if stats_npz is not None:
        np.savez(stats_npz, oracle_pred=all_oracle_pred.cpu().numpy(), oracle_pred_label=all_oracle_mask.cpu().numpy(), oracle_gt=all_oracle_gt.cpu().numpy())