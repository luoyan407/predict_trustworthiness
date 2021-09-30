import os, sys
import torch
import torch.nn as nn
import numpy as np
import random
from model import VisionTransformer
from config import *
from checkpoint import load_checkpoint
from data_loaders import *
from utils import setup_device, accuracy, MetricTracker, TensorboardWriter

from oracle import *

import torchvision.models as models

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def train_oracle_epoch(epoch, 
                        model, 
                        data_loader, 
                        criterion, 
                        metrics, 
                        oracle,
                        hyper_param,
                        optimizer_oracle,
                        lr_scheduler_oracle,
                        oracle_loss,
                        device=torch.device('cpu'),
                        tn_th=0.5,
                        stats_iter_log=None,
                        stats_npz=None,
                        eps=1e-7):
    metrics.reset()

    all_oracle_pred = torch.empty(0).cuda()
    all_oracle_gt = torch.empty(0, dtype=torch.bool).cuda()
    all_oracle_mask = torch.empty(0, dtype=torch.bool).cuda()
    all_oracle_tpr = torch.empty(0, dtype=torch.bool).cuda()
    all_oracle_tnr = torch.empty(0, dtype=torch.bool).cuda()

    # training loop
    for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        with torch.no_grad():
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)

        optimizer_oracle.zero_grad()
        rz_batch_data = batch_data
        if model.module.__class__.__name__ == 'VisionTransformer' and oracle.module.backbone.__class__.__name__ == 'ResNet':
            rz_batch_data = F.interpolate(batch_data, size=224)
        elif model.module.__class__.__name__ == 'ResNet' and oracle.module.backbone.__class__.__name__ == 'VisionTransformer':
            rz_batch_data = F.interpolate(batch_data, size=384)
        batch_oracle_pred = oracle(rz_batch_data)
        batch_pred_detach = batch_pred.clone().detach()

        loss_oracle, states_oracle = oracle_loss(batch_oracle_pred, batch_pred_detach, batch_target, hyperparam=hyper_param[0], class_weight=hyper_param[1])
        loss_oracle.backward()
        optimizer_oracle.step()
        lr_scheduler_oracle.step()

        all_oracle_pred = torch.cat((all_oracle_pred, batch_oracle_pred.data), dim=0)
        all_oracle_gt = torch.cat((all_oracle_gt, states_oracle[1]), dim=0)
        all_oracle_mask = torch.cat((all_oracle_mask, states_oracle[2]), dim=0)
        all_oracle_tpr = torch.cat((all_oracle_tpr, states_oracle[3]), dim=0)
        all_oracle_tnr = torch.cat((all_oracle_tnr, states_oracle[4]), dim=0)

        acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))

        metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        metrics.update('acc1', acc1.item())

        metrics.update('loss_oracle', loss_oracle.item())
        
        tpr = all_oracle_tpr.sum().item()/(all_oracle_gt.sum().item()+eps) * 100
        tnr = all_oracle_tnr.sum().item()/(torch.logical_not(all_oracle_gt).sum().item()+eps) * 100

        if batch_idx % 100 == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f}, Acc@1: {:.2f}, Loss_oracle: {:.4f}, tpr: {:.2f}, tnr: {:.2f}"
                    .format(epoch, batch_idx, len(data_loader), loss.item(), acc1.item(), loss_oracle.item(), tpr, tnr))

            if stats_iter_log is not None:
                with open(stats_iter_log, 'a') as fs:
                    out_str = 'train, {:d}, {:d}, '.format(epoch, batch_idx)
                    iter_stats = [loss.item(), acc1.item(), loss_oracle.item(), tpr, tnr]
                    for value in iter_stats:
                        out_str = out_str + '{:.4f}, '.format(value)
                    fs.write(out_str+'\n')

    tpr = all_oracle_tpr.sum().item()/(all_oracle_gt.sum().item()+eps) * 100
    tnr = all_oracle_tnr.sum().item()/(torch.logical_not(all_oracle_gt).sum().item()+eps) * 100
    metrics.update('tpr', tpr)
    metrics.update('tnr', tnr)

    final_perf = compute_performance(all_oracle_pred, all_oracle_gt, tn_th=tn_th)
    print(final_perf)

    if stats_npz is not None:
        np.savez(stats_npz, oracle_pred=all_oracle_pred.cpu().numpy(), oracle_pred_label=all_oracle_mask.cpu().numpy(), oracle_gt=all_oracle_gt.cpu().numpy())

    return metrics.result()


def valid_oracle_epoch(epoch, 
                        model, 
                        data_loader, 
                        criterion, 
                        metrics, 
                        oracle,
                        hyper_param,
                        oracle_loss,
                        device=torch.device('cpu'),
                        tn_th=0.5,
                        stats_iter_log=None,
                        stats_npz=None,
                        eps=1e-7):
    metrics.reset()
    losses = []
    acc1s = []
    losses_oracle = []

    all_oracle_pred = torch.empty(0).cuda()
    all_oracle_gt = torch.empty(0, dtype=torch.bool).cuda()
    all_oracle_mask = torch.empty(0, dtype=torch.bool).cuda()
    all_oracle_tpr = torch.empty(0, dtype=torch.bool).cuda()
    all_oracle_tnr = torch.empty(0, dtype=torch.bool).cuda()

    instance_model = None
    instance_oracle = None
    if isinstance(model, nn.DataParallel):
        instance_model = model.module
    if isinstance(oracle, nn.DataParallel):
        instance_oracle = oracle.module

    # validation loop
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
            acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))

            rz_batch_data = batch_data
            if instance_model.__class__.__name__ == 'VisionTransformer' and instance_oracle.backbone.__class__.__name__ == 'ResNet':
                rz_batch_data = F.interpolate(batch_data, size=224)
            elif instance_model.__class__.__name__ == 'ResNet' and instance_oracle.backbone.__class__.__name__ == 'VisionTransformer':
                rz_batch_data = F.interpolate(batch_data, size=384)
            batch_oracle_pred = oracle(rz_batch_data)
            batch_pred_detach = batch_pred.clone().detach()
            loss_oracle, states_oracle = oracle_loss(batch_oracle_pred, batch_pred_detach, batch_target, hyperparam=hyper_param[0], class_weight=hyper_param[1])

            losses.append(loss.item())
            acc1s.append(acc1.item())

            all_oracle_pred = torch.cat((all_oracle_pred, batch_oracle_pred.data), dim=0)
            all_oracle_gt = torch.cat((all_oracle_gt, states_oracle[1]), dim=0)
            all_oracle_mask = torch.cat((all_oracle_mask, states_oracle[2]), dim=0)
            all_oracle_tpr = torch.cat((all_oracle_tpr, states_oracle[3]), dim=0)
            all_oracle_tnr = torch.cat((all_oracle_tnr, states_oracle[4]), dim=0)

            losses_oracle.append(loss_oracle.item())

            tpr = all_oracle_tpr.sum().item()/(all_oracle_gt.sum().item()+eps) * 100
            tnr = all_oracle_tnr.sum().item()/(torch.logical_not(all_oracle_gt).sum().item()+eps) * 100

            if batch_idx % 100 == 0:
                if batch_idx == 0:
                    head_str = f'val, epoch, batch, loss, acc1, loss_oracle, tpr, tnr'
                    print(head_str)
                out_str = f'val, {epoch:d}, {batch_idx:d}, {np.mean(losses):.4f}, {np.mean(acc1s):.4f}, {np.mean(losses_oracle):.4f}, {tpr:.2f}, {tnr:.2f}'
                if stats_iter_log is not None:
                    with open(stats_iter_log, 'a') as fs:
                        fs.write(out_str+'\n')
                print(out_str)
            
    loss = np.mean(losses)
    acc1 = np.mean(acc1s)
    losses_oracle_avg = np.mean(losses_oracle)
    
    tpr = all_oracle_tpr.sum().item()/(all_oracle_gt.sum().item()+eps) * 100
    tnr = all_oracle_tnr.sum().item()/(torch.logical_not(all_oracle_gt).sum().item()+eps) * 100

    metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)
    metrics.update('acc1', acc1)
    metrics.update('loss_oracle', losses_oracle_avg)
    metrics.update('tpr', tpr)
    metrics.update('tnr', tnr)

    final_perf = compute_performance(all_oracle_pred, all_oracle_gt, tn_th=tn_th)
    print(final_perf)

    if stats_npz is not None:
        np.savez(stats_npz, oracle_pred=all_oracle_pred.cpu().numpy(), oracle_pred_label=all_oracle_mask.cpu().numpy(), oracle_gt=all_oracle_gt.cpu().numpy())

    return metrics.result()

def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str(save_dir + 'ep_{:02d}.pth'.format(epoch))
    torch.save(state, filename)


def main():
    config = get_train_config()

    # device
    device, device_ids = setup_device(config.n_gpu)

    # tensorboard
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss', 'acc1', 'loss_oracle', 'tpr', 'tnr']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    training_log = os.path.join(config.result_dir, 'train.log')
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
                 image_size=(oracle_config.vit_image_size, oracle_config.vit_image_size),
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

    # send oracle model to device
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
    train_dataloader = eval("{}DataLoader".format(config.dataset))(
                    data_dir=os.path.join(config.data_dir, config.dataset),
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='train')
    valid_dataloader = eval("{}DataLoader".format(config.dataset))(
                    data_dir=os.path.join(config.data_dir, config.dataset),
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='val')
    print(f'training batches {len(train_dataloader)}, validation batches {len(valid_dataloader)}')

    config.train_steps = config.train_epochs * len(train_dataloader)

    # training criterion
    print("create criterion and optimizer")
    criterion = nn.CrossEntropyLoss()

    optimizer_oracle = torch.optim.SGD(
        params=filter(lambda p: p.requires_grad, oracle.parameters()),
        lr=config.lr,
        weight_decay=config.wd,
        momentum=config.momentum)
    lr_scheduler_oracle = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer_oracle,
        max_lr=config.lr,
        pct_start=config.warmup_steps / config.train_steps,
        total_steps=config.train_steps)

    # start training oracle
    print("start training oracle")
    best_acc = 0.0

    oracle_pred_npz = 'oracle_pred.npz'
    stats_oracle_log = 'stats_oracle.csv'
    stats_oracle_iter_log = os.path.join(config.result_dir, 'stats_oracle_iter.csv')
    if stats_oracle_iter_log is not None:
        record_fields = ['phase', 'n_ep', 'n_iter'] + metric_names
        with open(stats_oracle_iter_log, 'w') as fs:
            out_str = ''
            for field_name in record_fields:
                out_str = out_str + '{}, '.format(field_name)
            fs.write(out_str+'\n')

    epochs = config.train_epochs
    print('there will be {} epochs (i.e., {} steps) for training'.format(epochs, config.train_steps))
    for epoch in range(1, epochs + 1):
        log = {'epoch': epoch}
        train_oracle_npz = os.path.join(config.result_dir, f'train_ep{epoch:02d}_{oracle_pred_npz}')
        val_oracle_npz = os.path.join(config.result_dir, f'val_ep{epoch:02d}_{oracle_pred_npz}')


        # train the oracle
        model.eval()
        oracle.train()
        result = train_oracle_epoch(epoch, model, train_dataloader, criterion, train_metrics, 
                                    oracle, hyper_param, optimizer_oracle, lr_scheduler_oracle, oracle_loss, device, tn_th=tn_th, stats_iter_log=stats_oracle_iter_log, stats_npz=train_oracle_npz)
        log.update(result)

        # validate the oracle
        model.eval()
        oracle.eval()
        result = valid_oracle_epoch(epoch, model, valid_dataloader, criterion, valid_metrics, 
                                    oracle, hyper_param, oracle_loss, device, tn_th=tn_th, stats_iter_log=stats_oracle_iter_log, stats_npz=val_oracle_npz)
        log.update(**{'val_' + k: v for k, v in result.items()})

        # save oracle
        save_model(config.checkpoint_dir, epoch, oracle, optimizer_oracle, lr_scheduler_oracle, device_ids)

        # output the headers of statistics
        if epoch == 1:
            f_stat = os.path.join(config.result_dir, stats_oracle_log)
            with open(f_stat, 'w') as fs:
                out_str = ''
                for key, value in log.items():
                    out_str = out_str + '{}, '.format(str(key))
                fs.write(out_str+'\n')

        # print logged informations to the screen
        f_stat = os.path.join(config.result_dir, stats_oracle_log)
        out_str = ''
        with open(f_stat, 'a') as fs:
            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))
                out_str = out_str + '{:.4f}, '.format(value)
            fs.write(out_str+'\n')

if __name__ == '__main__':
    main()