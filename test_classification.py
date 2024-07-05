import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from tqdm import tqdm

from datasets.datasets import ClassificationDataset, collate_fn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights, alexnet, AlexNet_Weights, vgg16, VGG16_Weights, efficientnet_b2, EfficientNet_B2_Weights

from utils.general import init_seeds


def eval(opt):
    ckpt = torch.load(opt.weight)
    model = ckpt['model']
    preprocess = ckpt['preprocess']
    model = model.eval()

    dataset = ClassificationDataset(opt.img_dir, opt.train_ann, img_size=preprocess.crop_size[0],
                                    normalize_mean=preprocess.mean, normalize_std=preprocess.std,
                                    train=True, augment=False, min_size=20)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,  # collate_fn=collate_fn,
                            num_workers=opt.workers)

    test_dataset = ClassificationDataset(opt.img_dir, opt.test_ann, img_size=preprocess.crop_size[0],
                                         normalize_mean=preprocess.mean, normalize_std=preprocess.std,
                                         train=True, augment=False, min_size=20)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,  # collate_fn=collate_fn,
                                 num_workers=opt.workers)

    conf_matrix = np.zeros((opt.num_classes, opt.num_classes))
    test_conf_matrix = np.zeros((opt.num_classes, opt.num_classes))
    device = torch.device(f'cuda:{opt.device}' if opt.device != 'cpu' else opt.device)
    model.to(device)

    count = 0
    correct = 0

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(tqdm(dataloader, total=len(dataloader))):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, dim=1)
            for gt_cls, pred_cls in zip(targets.cpu().numpy(), pred.cpu().numpy()):
                conf_matrix[gt_cls][pred_cls] += 1

        for i, (imgs, targets) in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, dim=1)
            for gt_cls, pred_cls in zip(targets.cpu().numpy(), pred.cpu().numpy()):
                conf_matrix[gt_cls][pred_cls] += 1
                test_conf_matrix[gt_cls][pred_cls] += 1
                if gt_cls == pred_cls:
                    correct += 1
                count += 1

    np.save(os.path.join(opt.save_dir, 'conf_matrix.npy'), conf_matrix)
    np.save(os.path.join(opt.save_dir, 'test_conf_matrix.npy'), test_conf_matrix)
    conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    print(conf_matrix)
    print('average diagonal:', conf_matrix.diagonal().mean())
    print('overall accuracy:', correct / count)
    return conf_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='', help='path to weights')
    parser.add_argument('--img-dir', type=str, default='./.data/MSCOCO/images', help='coco images path')
    parser.add_argument('--train-ann', type=str, default='./data/coco_ann5_sim/clean_train', help='clean train annotations')
    parser.add_argument('--test-ann', type=str, default='./data/coco_ann5_sim/test', help='clean test annotations')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--num-classes', type=int, default=80, help='number of classes')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--workers', type=int, default=16, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='classification-coco', help='save to project/name')
    parser.add_argument('--name', default='resnet50', help='save confusion matrix to project/name')
    opt = parser.parse_args()

    opt.save_dir = os.path.join('outputs', opt.project, opt.name)
    opt.num_classes += 1

    eval(opt)
