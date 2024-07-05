import argparse
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
import numpy as np
from torchvision.models import ResNet101_Weights, ResNet50_Weights, ResNet18_Weights

from tqdm import tqdm

from datasets.datasets import CleanDataset
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from models.rpn_generator import RPNGenerator
from utils.faster_rcnn.general import collate_fn, transform_to_faster_rcnn_targets
from utils.general import init_seeds
from utils.metrics import compute_ar


def train(opt):
    init_seeds(opt.seed)
    save_dir, epochs, batch_size = Path(opt.save_dir), opt.epochs, opt.batch_size
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    if opt.model == 'resnet18':
        weights = ResNet18_Weights.DEFAULT
    elif opt.model == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
    elif opt.model == 'resnet101':
        weights = ResNet101_Weights.DEFAULT
    else:
        raise NotImplementedError
    backbone = resnet_fpn_backbone(backbone_name=opt.model, weights=weights)

    # for param in model.parameters():  # freeze
    #     param.requires_grad = False

    model = RPNGenerator(backbone)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    print('data parameters: ' + ', '.join(f'{k}={v}' for k, v in data_dict.items()))
    dataset = CleanDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['clean_train'],
                           image_size=None, normalize_box=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                            num_workers=opt.workers)
    nb = len(dataloader)
    test_dataset = CleanDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['val'],
                           image_size=None, normalize_box=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                                 num_workers=opt.workers)

    device = torch.device(f'cuda:{opt.device}' if opt.device != 'cpu' else opt.device)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay, gamma=0.1)

    start_epoch = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=nb)
        metrics = torch.zeros(3, device=device)
        for i, (imgs, targets, paths, _) in pbar:
            targets = transform_to_faster_rcnn_targets(targets)

            imgs = list(x.to(device) for x in imgs)
            # if half:
            #     img = [x.half() for x in img]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            proposals, scores, loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()

            # print(loss.detach().item())
            optimizer.step()
            # ar = []
            # for proposal, target in zip(proposals, targets):
            #     ar.append(compute_ar(pred_boxes=proposal.detach().cpu().numpy(), gt_boxes=target['boxes'].cpu().numpy()))
            # ar = np.mean(ar)
            ar = 0.
            loss_items = torch.stack((loss_dict['loss_rpn_box_reg'], loss_dict['loss_objectness'], torch.tensor(ar, device=device))).detach()
            metrics = (metrics * i + loss_items) / (i + 1)

            s = ('%10s' + '%10.4g' * 3) % ('%g/%g' % (epoch, epochs - 1), *metrics)
            pbar.set_description(s)

        val_metrics = test(model, test_dataloader, device)
        print('train box_loss/obj_loss/ar:', metrics.cpu())
        print('val box_loss/obj_loss/ar:', val_metrics)

        ckpt = {'epoch': epoch,
                'model': model}
        torch.save(ckpt, weights_dir / 'epoch_{:03d}.pt'.format(epoch))
        scheduler.step()

    return model


def test(model, test_dataloader, device):
    model = model.eval()
    nb = len(test_dataloader)
    metrics = torch.zeros(3, device=device)
    with torch.no_grad():
        for i, (imgs, targets, paths, _) in enumerate(tqdm(test_dataloader, total=nb)):
            targets = transform_to_faster_rcnn_targets(targets)

            imgs = list(x.to(device) for x in imgs)
            # if half:
            #     img = [x.half() for x in img]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            proposals, scores, loss_dict = model(imgs, targets)
            ar = []
            for proposal, target in zip(proposals, targets):
                ar.append(
                    compute_ar(pred_boxes=proposal.detach().cpu().numpy(), gt_boxes=target['boxes'].cpu().numpy()))
            ar = np.mean(ar)

            metrics += torch.stack((loss_dict['loss_rpn_box_reg'], loss_dict['loss_objectness'], torch.tensor(ar, device=device))).detach()

    return metrics.cpu() / nb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', help='one of resnet18, resnet50, resnet101')
    parser.add_argument('--data', type=str, default='data/coco_clean.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='rpn-coco', help='save to project/name')
    parser.add_argument('--name', default='resnet18', help='save to project/name')
    parser.add_argument('--linear-lr', action='store_true', help='Linear LR')
    parser.add_argument('--eval-only', action='store_true', help='eval model only')
    parser.add_argument('--weight', default='', help='weight path for eval')
    parser.add_argument('--seed', type=int, default=1, help='seed for random generator')
    opt = parser.parse_args()

    if opt.eval_only:
        if os.path.exists(opt.weight):
            model = torch.load(opt.weight, map_location='cpu')['model']
        else:
            model = RPNGenerator.load_pretrained_rpn(opt.model)
        with open(opt.data) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        test_dataset = CleanDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['val'],
                                    image_size=None, normalize_box=False)
        test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn,
                                     num_workers=opt.workers)

        device = torch.device(f'cuda:{opt.device}' if opt.device != 'cpu' else opt.device)
        model.to(device)
        val_metrics = test(model, test_dataloader, device)
        print('val box_loss/obj_loss/ar:', val_metrics)
        sys.exit(0)

    opt.save_dir = os.path.join('outputs', opt.project, opt.name)
    os.makedirs(opt.save_dir, exist_ok=True)

    train(opt)
