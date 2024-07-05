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

# torch.autograd.detect_anomaly(True)
def train(opt):
    init_seeds(opt.seed)
    save_dir, epochs, batch_size = Path(opt.save_dir), opt.epochs, opt.batch_size
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    if opt.model == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights if opt.pretrained else None)
        preprocess = weights.transforms()
    elif opt.model == 'alexnet':
        weights = AlexNet_Weights.DEFAULT
        model = alexnet(weights=weights if opt.pretrained else None)
        preprocess = weights.transforms()
    elif opt.model == 'vgg16':
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights=weights if opt.pretrained else None)
        preprocess = weights.transforms()
    elif opt.model == 'efficientdet_b2':
        weights = EfficientNet_B2_Weights.DEFAULT
        model = efficientnet_b2(weights=weights if opt.pretrained else None)
        preprocess = weights.transforms()
    else:
        raise NotImplementedError

    # for param in model.parameters():  # freeze
    #     param.requires_grad = False

    if opt.model == 'resnet50':
        if opt.pretrained:
            for param in model.parameters():  # freeze
                param.requires_grad = False
            for param in model.layer4.parameters():
                param.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, opt.num_classes)
    elif opt.model == 'alexnet':
        model.classifier = nn.Sequential(
            nn.Dropout(p=.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, opt.num_classes),
        )
    elif opt.model == 'vgg16':
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=.5),
            nn.Linear(4096, opt.num_classes),
        )
    elif opt.model == 'efficientdet_b2':
        model.classifier = nn.Sequential(
            nn.Dropout(p=.3, inplace=True),
            nn.Linear(model.classifier[1].in_features, opt.num_classes),
        )
    else:
        raise NotImplementedError

    dataset = ClassificationDataset(opt.img_dir, opt.train_ann, img_size=preprocess.crop_size[0], normalize_mean=preprocess.mean, normalize_std=preprocess.std, train=True, min_size=20)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, #collate_fn=collate_fn,
                            num_workers=opt.workers)
    nb = len(dataloader)
    test_dataset = ClassificationDataset(opt.img_dir, opt.test_ann, img_size=preprocess.crop_size[0], normalize_mean=preprocess.mean, normalize_std=preprocess.std, min_size=20)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, #collate_fn=collate_fn,
                                 num_workers=opt.workers)

    device = torch.device(f'cuda:{opt.device}' if opt.device != 'cpu' else opt.device)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay, gamma=0.1)
    class_weights = dataset.get_class_weights()
    criterion = nn.CrossEntropyLoss(class_weights)
    criterion.to(device)

    start_epoch = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=nb)
        metrics = torch.zeros(2, device=device)
        for i, (imgs, targets) in pbar:
            optimizer.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            _, pred = torch.max(outputs, dim=1)
            correct = torch.sum(pred == targets) / targets.shape[0]

            # print(loss.detach().item())
            optimizer.step()
            metrics = (metrics * i + torch.stack([loss.detach(), correct])) / (i + 1)

            s = ('%10s' + '%10.4g' * 2) % ('%g/%g' % (epoch, epochs - 1), *metrics)
            pbar.set_description(s)

        val_metrics = test(model, criterion, test_dataloader, device)
        print('train loss/acc:', metrics.cpu())
        print('val loss/acc:', val_metrics)

        ckpt = {'epoch': epoch,
                'model': model,
                'preprocess': preprocess}
        torch.save(ckpt, weights_dir / 'epoch_{:03d}.pt'.format(epoch))
        scheduler.step()
    return model


def test(model, criterion, test_dataloader, device):
    model = model.eval()
    nb = len(test_dataloader)
    metrics = torch.zeros(2, device=device)
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(tqdm(test_dataloader, total=nb)):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            _, pred = torch.max(outputs, dim=1)
            correct = torch.sum(pred == targets) / targets.shape[0]
            metrics += torch.stack([loss.detach(), correct])

    return metrics.cpu() / nb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='efficientdet_b2', help='one of alexnet, vgg16, resnet50 or efficientdet_b2')
    parser.add_argument('--img-dir', type=str, default='./.data/MSCOCO/images', help='coco images path')
    parser.add_argument('--train-ann', type=str, default='./data/coco_ann5_sim/clean_train', help='clean train annotations')
    parser.add_argument('--test-ann', type=str, default='./data/coco_ann5_sim/test', help='clean test annotations')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num-classes', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lr-decay', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--workers', type=int, default=16, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='classification-coco', help='save to project/name')
    parser.add_argument('--name', default='alexnet', help='save to project/name')
    parser.add_argument('--linear-lr', action='store_true', help='Linear LR')
    parser.add_argument('--seed', type=int, default=1, help='seed for random generator')
    parser.add_argument('--pretrained', action='store_true', help='pretrained weights')
    opt = parser.parse_args()

    opt.save_dir = os.path.join('outputs', opt.project, opt.name)
    os.makedirs(opt.save_dir, exist_ok=True)

    opt.num_classes += 1  # background

    train(opt)
