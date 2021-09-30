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


def main():
    config = get_train_config()

    # device
    device, device_ids = setup_device(config.n_gpu)

    # tensorboard
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss', 'acc1', 'loss_retro', 'tpr', 'tnr']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    training_log = os.path.join(config.result_dir, 'eval.log')
    sys.stdout = PrintLogger(training_log, 'w')

    # create model
    print("create model {}".format(config.classifier))
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
        model = models.resnet50(pretrained=True)

    # send model to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

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

    # send oracle to device
    oracle = oracle.to(device)
    if len(device_ids) > 1:
        oracle = torch.nn.DataParallel(oracle, device_ids=device_ids)

    loss_param = config.oracle_loss_hyperparam
    class_weight = config.oracle_class_weight
    if class_weight is None:
        class_weight = [1.0, 1.0]
    hyper_param = [loss_param, class_weight]
    #--------end of creating oracle--------

    # create dataloader
    print("create dataloaders")
    train_dataloader = eval("{}DataLoader_shufflable".format(config.dataset))(
                    data_dir=os.path.join(config.data_dir, config.dataset),
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='train',
                    shuffle=False)
    valid_dataloader = eval("{}DataLoader_shufflable".format(config.dataset))(
                    data_dir=os.path.join(config.data_dir, config.dataset),
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='val',
                    shuffle=False)
    print(f'training batches {len(train_dataloader)}, validation batches {len(valid_dataloader)}')

    config.train_steps = config.train_epochs * len(train_dataloader)

    # training criterion
    print("create criterion and optimizer")
    criterion = nn.CrossEntropyLoss()

    # start training
    print("start evaluating the oracle on the val set")

    oracle_pred_npz = 'oracle_pred.npz'
    stats_oracle_log = None
    stats_oracle_iter_log = None

    epoch = 1

    train_oracle_npz = os.path.join(config.result_dir, f'train_ep{epoch:02d}_{oracle_pred_npz}')
    val_oracle_npz = os.path.join(config.result_dir, f'val_ep{epoch:02d}_{oracle_pred_npz}')
    log = {}
    model.eval()
    oracle.eval()
    
    result = valid_oracle_epoch(epoch, model, valid_dataloader, criterion, valid_metrics, 
                                oracle, hyper_param, oracle_loss, device, tn_th=tn_th, stats_iter_log=stats_oracle_iter_log, stats_npz=val_oracle_npz)
    log.update(**{'val_' + k: v for k, v in result.items()})
    print(log)

if __name__ == '__main__':
    main()