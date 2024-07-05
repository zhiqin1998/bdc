import copy
import os
import sys
import time
import math
import random
import logging
import argparse
from copy import deepcopy
from threading import Thread

import numpy as np
import yaml
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from torch.cuda import amp
from torch.utils.data import DataLoader
from datasets.datasets import SyntheticDataset, collate_fn, CleanDataset, get_augmentation
from models.yolov7.yolo import Model
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from utils.crowd.aggregator import NoAggregator
from utils.crowd.bdc import BDCAggregator
from utils.crowd.earl import EARLAggregator
from utils.crowd.majority_vote import MVAggregator
from utils.general import init_seeds, one_cycle, check_img_size, labels_to_class_weights, scale_coords
from utils.plots import plot_images
from utils.yolov7.autoanchor import check_anchors
from utils.yolov7.loss import ComputeLoss, ComputeLossOTA, ComputeLossOTAWithLogits
from utils.yolov7.general import transform_to_yolov7_targets, nms_with_logits
from utils.metrics import fitness, compute_map
from utils.torch_utils import intersect_dicts, ModelEMA, time_synchronized
from test_yolov7 import test

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(hyp, opt):
    tb_writer = SummaryWriter(opt.save_dir)
    wandb_run = wandb.init(project=opt.project, config=vars(opt), name=opt.name)
    logger.info('hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, weights = Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights
    init_seeds(opt.seed)
    plots = opt.plots

    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_file = save_dir / 'results.txt'

    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    logger.info('data parameters: ' + ', '.join(f'{k}={v}' for k, v in data_dict.items()))

    nc = data_dict['nc']
    assert len(data_dict['names']) == nc

    device = torch.device(f'cuda:{opt.device}' if opt.device != 'cpu' else opt.device)
    cuda = device.type != 'cpu'
    # model
    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) else []  # exclude keys
        exclude.append('model.105') # exclude last layer
        state_dict = ckpt['model']
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    # optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / opt.batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= opt.batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'):
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):
                pg0.append(v.rbr_dense.vector)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs
        start_epoch = ckpt.get('epoch', -1) + 1
        # if opt.resume:
        #     assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # crowd annotations aggregator
    is_bdc = False
    with open(opt.crowd_aggregator) as f:
        ca_hyp = yaml.load(f, Loader=yaml.SafeLoader)
    logger.info('aggregator parameters: ' + ', '.join(f'{k}={v}' for k, v in ca_hyp.items()))

    if ca_hyp['type'] == 'mv':
        aggregator = MVAggregator(data_dict['n_annotator'], **ca_hyp['parameters'])
    elif ca_hyp['type'] == 'earl':
        aggregator = EARLAggregator(data_dict['n_annotator'], ann_weight=data_dict['earl_ann_weights'])
    elif ca_hyp['type'] == 'bdc':
        is_bdc = True
        aggregator = BDCAggregator(data_dict['n_annotator'], data_dict['nc'], data_dict['nc_ann'], **ca_hyp['parameters'])
    else:
        logger.warning('invalid or no aggregator provided. defaulting to not aggregating crowd labels')
        aggregator = NoAggregator()

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    augments = get_augmentation(hue=data_dict['hsv_h'], sat=data_dict['hsv_s'], val=data_dict['hsv_v'],
                                translate=data_dict['translate'], scale=data_dict['scale'], rotate=data_dict['rotate'],
                                shear=data_dict['shear'], perspective=data_dict['perspective'],
                                fliplr=data_dict['fliplr'], flipud=data_dict['flipud'])

    if 'train' in data_dict:
        dataset = SyntheticDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['train'],
                                   image_size=(imgsz, imgsz), train=True, augments=augments, aggregator=aggregator,
                                   clean_annotations_path=data_dict['clean_train'])
    else:
        logger.info('training on clean dataset')
        dataset = SyntheticDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['clean_train'], expected_ele=5,
                                   image_size=(imgsz, imgsz), train=True, augments=augments, aggregator=aggregator,
                                   clean_annotations_path=data_dict['clean_train'])
    if is_bdc:
        # initialize first aggregated annotations and set to dataset, annotations now becomes logit instead of hard cls
        # annotation is [xyxy, logit, weight]
        dataset.noisy_annotations = dataset.normalize_bbox(dataset.noisy_annotations)
        bdc_annotations, lower_bound = aggregator.initialize_dataset(dataset)
        assert len(dataset.annotations) == len(bdc_annotations), 'this shouldnt happen'
        dataset.annotations = dataset.normalize_bbox(bdc_annotations, inverse=True)

    agg_ap50, agg_ap75, agg_ap = 0., 0., 0.
    if dataset.clean_annotations is not None:
        if is_bdc:
            agg_ap50, agg_ap75, agg_ap = compute_map([label[:, :-1] for label in dataset.annotations],
                                                     dataset.clean_annotations,
                                                     is_score=ca_hyp['type'] == 'earl')
        else:
            # too time-consuming for big dataset (do it manually with utils/metrics.py)
            if isinstance(aggregator, (NoAggregator, EARLAggregator)) and len(dataset) > 10000:
                agg_ap50, agg_ap75, agg_ap = 0., 0., 0.
            else:
                agg_ap50, agg_ap75, agg_ap = compute_map(dataset.annotations, dataset.clean_annotations,
                                                         is_score=ca_hyp['type'] == 'earl')

    logger.info(f'initial aggregation AP: {agg_ap50:.5f}, {agg_ap75:.5f}, {agg_ap:.5f}')

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=opt.workers, pin_memory=True,
                            collate_fn=collate_fn, shuffle=True)
    nb = len(dataloader)  # number of batches
    logger.info(f'training on {len(dataset)} images')
    if 'val' in data_dict:
        test_dataset = CleanDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['val'],
                                    image_size=(imgsz_test, imgsz_test))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=opt.workers, pin_memory=True,
                                     collate_fn=collate_fn, shuffle=False)
        logger.info(f'testing on {len(test_dataset)} images')
    else:
        logger.info('no test or validation dataset found')
        test_dataloader = None

    if plots:
        # plot_labels(labels, names, save_dir, loggers)
        c = np.concatenate(dataset.annotations, 0)[:, 4]
        print(c.shape)
        if tb_writer:
            tb_writer.add_histogram('classes', c, 0)

    if not opt.noautoanchor:
        check_anchors(copy.deepcopy(dataset), model=model, thr=hyp['anchor_t'], imgsz=imgsz)
    model.half().float()  # pre-reduce anchor precision

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.annotations, nc).to(device) * nc  # attach class weights
    model.names = data_dict['names']

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.75, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model, weighted=ca_hyp['type'] == 'earl') if not is_bdc else ComputeLossOTAWithLogits(model, weighted=True)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(4, device=device)  # mean losses
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _, clean_targets) in pbar:  # batch ----------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            targets = transform_to_yolov7_targets(targets, logits=is_bdc)
            if not is_bdc and targets.shape[1] == 7:  # earl
                assert ca_hyp['type'] == 'earl', 'this is only for earl method'
                targets, weights = targets[:, torch.arange(7) != 2], targets[:, 2]  # earl is idx cls_id weight xywh
                weights = weights.to(device)
            elif is_bdc:
                # bdc is idx weight cls_logits xywh
                targets, weights = targets[:, torch.arange(targets.shape[1]) != 1], targets[:, 1]
                weights = weights.to(device)
            else:
                weights = None

            # note: already normalized
            imgs = imgs.to(device, non_blocking=True).float()  # / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / opt.batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in
                          imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    if weights is None:
                        loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                    else:
                        loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs, weights)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            #print
            if plots and i < 1 and epoch < 20:  # plot 1 img per epoch
                if is_bdc:
                    plot_targets = torch.zeros(len(targets), 6)  # convert logits back to cls id
                    plot_targets[:, 0] = targets[:, 0]
                    plot_targets[:, 2:6] = targets[:, -4:]
                    plot_targets[:, 1] = targets[:, 1:1+nc].max(dim=1)[1]
                    targets = plot_targets
                f = save_dir / f'train_ep{epoch}_batch{i}.jpg'  # filename
                Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()

            # end batch ------------------------------------------------------------------------------------------------

        if is_bdc:
            # update bdc now
            # new dataloader with no shuffle
            dataset.train = False
            dataloader = DataLoader(dataset, batch_size=batch_size * 2, num_workers=opt.workers, pin_memory=True,
                                    collate_fn=collate_fn, shuffle=False)
            logger.info('updating with vb')
            out_with_logits = []
            ema.ema.eval()
            t = time_synchronized()
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                with torch.no_grad():
                    for img, _, _, shapes, _ in tqdm(dataloader, total=len(dataloader)):
                        img = img.to(device, non_blocking=True)
                        out, _ = ema.ema(img)
                        out = nms_with_logits(out)  # nms also sorts the prediction by conf already for bcc
                        for si, pred in enumerate(out):
                            predn = pred.clone().detach().cpu()
                            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
                            predn = predn.numpy().astype(np.float32)
                            oh, ow = shapes[si][0]  # normalize box
                            ori_dims = np.asarray([ow, oh, ow, oh], dtype=np.float32)[np.newaxis, :]
                            predn[:, :4] = predn[:, :4] / ori_dims
                            out_with_logits.append(predn)
            model.train()
            new_annotations, lower_bound = aggregator.fit_transform_crowd_labels(dataset.noisy_annotations,
                                                                                 out_with_logits, warmup=opt.bdc_warmup)
            new_annotations = dataset.normalize_bbox(new_annotations, inverse=True)
            t0 = time_synchronized() - t
            logger.info('Aggregator took {:.5f}s to complete'.format(t0))
            # compute stats for q_t if we have clean annotations
            if dataset.clean_annotations is not None:
                agg_ap50_, agg_ap75_, agg_ap_ = compute_map([label[:, :-1] for label in new_annotations], dataset.clean_annotations)

                if agg_ap_ >= agg_ap:
                    agg_ap50, agg_ap75, agg_ap = agg_ap50_, agg_ap75_, agg_ap_
                    dataset.annotations = new_annotations
            # recreate dataloader
            dataset.train = True
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=opt.workers, pin_memory=True,
                                    collate_fn=collate_fn, shuffle=True)
            logger.info(f'epoch {epoch} aggregation AP: {agg_ap50:.5f}, {agg_ap75:.5f}, {agg_ap:.5f}')

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # test, log and save
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
        if test_dataloader is not None and epoch > opt.test_after and epoch % opt.test_interval == 0:
            final_epoch = epoch + 1 == epochs
            results, maps, times = test(data_dict, coco_eval=True,
                                        batch_size=batch_size * 2,
                                        imgsz=imgsz_test,
                                        model=ema.ema,
                                        dataloader=test_dataloader,
                                        save_dir=save_dir,
                                        verbose=nc < 50 and final_epoch,
                                        plots=plots and final_epoch,
                                        compute_loss=compute_loss)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 8 % results + '\n')  # append metrics, val_loss

        # Log
        tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                'train/agg_mAP_0.5', 'train/agg_mAP_0.75', 'train/agg_mAP_0.5:0.95',  # agg metrics
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'metrics/mAP_0.75',
                'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                'x/lr0', 'x/lr1', 'x/lr2']  # params
        stats = list(mloss[:-1]) + [agg_ap50, agg_ap75, agg_ap] + list(results) + lr
        if is_bdc:
            stats.extend([lower_bound])
            tags.extend(['train/vb_lower_bound'])

        wandb_temp = {}
        for x, tag in zip(stats, tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard
            if wandb_run:
                wandb_temp[tag] = x
        if wandb_run:
            wandb_run.log(wandb_temp)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness and ni > nw:
            best_fitness = fi
            # Save
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(model).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'aggregator': aggregator,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': wandb_run.id if wandb_run else None}
            torch.save(ckpt, weights_dir / 'best.pt'.format(epoch))
            del ckpt

        # Save model
        if epoch >= opt.save_after:  # if save
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(model).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'aggregator': aggregator,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': wandb_run.id if wandb_run else None}

            # Save
            if epoch % opt.save_interval == 0:
                torch.save(ckpt, weights_dir / 'epoch_{:03d}.pt'.format(epoch))
            elif epoch == epochs - 1:  # save last
                torch.save(ckpt, weights_dir / 'last.pt'.format(epoch))
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    torch.cuda.empty_cache()
    wandb_run.finish()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov7/config/yolov7.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/voc_2007_ann5_0.7.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='models/yolov7/config/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check', default=True)
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='crowd-annotations', help='save to project/name')
    parser.add_argument('--name', default='yolov7', help='save to project/name')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--save-interval', type=int, default=10, help='save model every save_interval epoch')
    parser.add_argument('--save-after', type=int, default=20, help='save model after save_after epoch')
    parser.add_argument('--test-after', type=int, default=-1, help='test model after test_after epoch')
    parser.add_argument('--test-interval', type=int, default=1, help='test model every test_interval epoch')
    parser.add_argument('--crowd-aggregator', default='config/mv.yaml', help='config path to crowd_aggregator.yaml')
    parser.add_argument('--seed', type=int, default=1, help='seed for random generator')
    parser.add_argument('--bdc-warmup', type=int, default=10, help='warmup steps for bdc')
    parser.add_argument('--plots', action='store_true', help='plot output')
    opt = parser.parse_args()

    opt.save_dir = os.path.join('outputs', opt.project, opt.name)
    os.makedirs(opt.save_dir)

    logger.addHandler(logging.FileHandler(os.path.join(opt.save_dir, 'run.log')))

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    logger.info(opt)
    train(hyp, opt)
