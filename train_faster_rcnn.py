import os
import sys
import time
import math
import logging
import argparse
from copy import deepcopy
from threading import Thread

import numpy as np
import yaml
import wandb
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from torch.cuda import amp
from torch.utils.data import DataLoader
from datasets.datasets import SyntheticDataset, CleanDataset, get_augmentation
from models.faster_rcnn.faster_rcnn import create_model, get_loss_with_logits, get_detection_with_logits, \
    get_loss_with_weights
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from utils.faster_rcnn.general import transform_to_faster_rcnn_targets, collate_fn
from utils.crowd.earl import EARLAggregator
from utils.crowd.aggregator import NoAggregator
from utils.crowd.majority_vote import MVAggregator
from utils.crowd.bdc import BDCAggregator
from utils.general import init_seeds, one_cycle
from utils.metrics import compute_map
from utils.plots import plot_images2
from utils.torch_utils import time_synchronized
from test_faster_rcnn import test

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
    start_epoch = 0
    model = create_model(nc + 1, pretrained=True).to(device)  # add background
    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint
        state_dict = ckpt['model'].float().state_dict()
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if opt.adam:
        optimizer = optim.Adam(params, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(params, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{total_trainable_params:,} training parameters.")
    del params

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained and opt.resume:
        # Optimizer
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

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
        aggregator = BDCAggregator(data_dict['n_annotator'], data_dict['nc'], data_dict['nc_ann'],
                                   **ca_hyp['parameters'])

    else:
        logger.warning('invalid or no aggregator provided. defaulting to not aggregating crowd labels')
        aggregator = NoAggregator()

    # Image sizes
    augments = get_augmentation(hue=data_dict['hsv_h'], sat=data_dict['hsv_s'], val=data_dict['hsv_v'],
                                translate=data_dict['translate'], scale=data_dict['scale'], rotate=data_dict['rotate'],
                                shear=data_dict['shear'], perspective=data_dict['perspective'],
                                fliplr=data_dict['fliplr'], flipud=data_dict['flipud'])

    if 'train' in data_dict:
        dataset = SyntheticDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['train'],
                                   image_size=None, train=True, augments=augments, aggregator=aggregator, normalize_box=False,
                                   clean_annotations_path=data_dict['clean_train'])
    else:
        logger.info('training on clean dataset')
        dataset = SyntheticDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['clean_train'], expected_ele=5,
                                   image_size=None, train=True, augments=augments, aggregator=aggregator,
                                   normalize_box=False,
                                   clean_annotations_path=data_dict['clean_train'])
    # dataset.annotations = [np.concatenate((t[:, :4], np.eye(nc)[t[:, 4].astype(int)]), axis=1) for t in dataset.annotations]
    # lower_bound = 0
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
                                    image_size=None, normalize_box=False)
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

    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.names = data_dict['names']

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)

    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.75, mAP@.5-.95, val_loss(rpn_box, box, obj, cls)
    scaler = amp.GradScaler(enabled=cuda)

    logger.info(f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(5, device=device)  # mean losses
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'rpn_box', 'box', 'obj', 'cls', 'total'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _, clean_targets) in pbar:  # batch ----------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            targets = transform_to_faster_rcnn_targets(targets, logits=is_bdc)
            if 'weights' in targets[0]:
                assert ca_hyp['type'] == 'earl' or is_bdc, 'this is only for earl or bdc method'

            # note: already normalized
            imgs = list(image.to(device) for image in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward
            with amp.autocast(enabled=cuda):
                if is_bdc:
                    loss_dict = get_loss_with_logits(model, imgs, targets)
                else:
                    loss_dict = get_loss_with_weights(model, imgs, targets)  # weight is all 1 if not given
                losses = sum(loss for loss in loss_dict.values())
                loss_items = torch.stack((loss_dict['loss_rpn_box_reg'], loss_dict['loss_box_reg'],
                                          loss_dict['loss_objectness'], loss_dict['loss_classifier'], losses)).detach()

            loss_value = losses.item()
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict)
                sys.exit(1)

            optimizer.zero_grad()
            # Backward
            scaler.scale(losses).backward()

            # Optimize
            scaler.step(optimizer)  # optimizer.step
            scaler.update()

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 5) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss)
            pbar.set_description(s)
            #print
            if plots and i < 1 and epoch < 50:  # plot 1 img per epoch:
                if is_bdc:
                    for ti in range(len(targets)):
                        targets[ti]['labels'] = targets[ti]['labels'].max(dim=1)[1]
                f = save_dir / f'train_ep{epoch}_batch{i}.jpg'  # filename
                Thread(target=plot_images2, args=(imgs, targets, paths, f), daemon=True).start()

            # end batch ------------------------------------------------------------------------------------------------

        if is_bdc:
            # update bdc now
            # new dataloader with no shuffle
            dataset.train = False
            dataloader = DataLoader(dataset, batch_size=batch_size * 2, num_workers=opt.workers, pin_memory=True,
                                    collate_fn=collate_fn, shuffle=False)
            logger.info('updating with vb')
            out_with_logits = []
            model.eval()
            t = time_synchronized()
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                with torch.no_grad():
                    for img, _, _, shapes, _ in tqdm(dataloader, total=len(dataloader)):
                        img = list(image.to(device) for image in img)
                        out = get_detection_with_logits(model, img)  # prediction already sorted
                        for si, pred in enumerate(out):
                            oh, ow = shapes[si][0]  # normalize box
                            ori_dims = np.asarray([ow, oh, ow, oh], dtype=np.float32)[np.newaxis, :]
                            pred_box = pred['boxes'].clone().detach().cpu().numpy().astype(np.float32) / ori_dims
                            logits = pred['logits'].clone().detach().cpu().numpy().astype(np.float32)
                            out_with_logits.append(np.concatenate((pred_box, logits), axis=1))
            model.train()
            new_annotations, lower_bound = aggregator.fit_transform_crowd_labels(dataset.noisy_annotations,
                                                                                 out_with_logits, warmup=opt.bdc_warmup)
            new_annotations = dataset.normalize_bbox(new_annotations, inverse=True)
            t0 = time_synchronized() - t
            logger.info('Aggregator took {:.5f}s to complete'.format(t0))
            # compute stats for q_t if we have clean annotations
            if dataset.clean_annotations is not None:
                agg_ap50_, agg_ap75_, agg_ap_ = compute_map([label[:, :-1] for label in new_annotations],
                                                            dataset.clean_annotations)

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
        if test_dataloader is not None and epoch > opt.test_after and epoch % opt.test_interval == 0:
            final_epoch = epoch + 1 == epochs
            results, maps, times = test(data_dict, coco_eval=True,
                                        batch_size=batch_size * 2,
                                        model=model,
                                        dataloader=test_dataloader,
                                        save_dir=save_dir,
                                        verbose=nc < 50 and final_epoch,
                                        plots=plots and final_epoch,)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 9 % results + '\n')  # append metrics, val_loss

        # Log
        tags = ['train/rpn_box_loss', 'train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                'train/agg_mAP_0.5', 'train/agg_mAP_0.75', 'train/agg_mAP_0.5:0.95',  # agg metrics
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'metrics/mAP_0.75',
                'val/rpn_box_loss', 'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                'x/lr0']  # params
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
        fi = results[2] * 0.1 + results[3] * 0.9  # weighted combination of [mAP@.5, mAP@.5-.95]
        if fi > best_fitness and ni > nw:
            best_fitness = fi
            # Save
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': model,
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
                    'model': model,
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
    parser.add_argument('--data', type=str, default='data/voc_2007_ann5_0.7.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='models/faster_rcnn/config/hyp.faster_rcnn.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='crowd-annotations', help='save to project/name')
    parser.add_argument('--name', default='faster_rcnn', help='save to project/name')
    parser.add_argument('--linear-lr', action='store_true', help='Linear LR')
    parser.add_argument('--save-interval', type=int, default=5, help='save model every save_interval epoch')
    parser.add_argument('--save-after', type=int, default=30, help='save model after save_after epoch')
    parser.add_argument('--test-after', type=int, default=-1, help='test model after test_after epoch')
    parser.add_argument('--test-interval', type=int, default=1, help='test model every test_interval epoch')
    parser.add_argument('--crowd-aggregator', default='config/mv.yaml', help='config path to crowd_aggregator.yaml')
    parser.add_argument('--seed', type=int, default=1, help='seed for random generator')
    parser.add_argument('--bdc-warmup', type=int, default=5, help='warmup steps for bdc')
    parser.add_argument('--plots', action='store_true', help='plot output')
    parser.add_argument('--resume', action='store_true', help='resume training')
    opt = parser.parse_args()

    opt.save_dir = os.path.join('outputs', opt.project, opt.name)
    os.makedirs(opt.save_dir)

    logger.addHandler(logging.FileHandler(os.path.join(opt.save_dir, 'run.log')))

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    logger.info(opt)
    train(hyp, opt)
