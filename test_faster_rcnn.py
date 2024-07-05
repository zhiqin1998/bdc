import argparse
import json
import logging
import sys
import time
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.datasets import CleanDataset
from models.faster_rcnn.faster_rcnn import get_loss_and_detection
from utils.faster_rcnn.general import transform_to_faster_rcnn_targets, collate_fn
from utils.general import coco80_to_coco91_class, check_file, box_iou, xyxy2xywh, set_logging, increment_path
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import plot_images2, output_to_target2
from utils.torch_utils import time_synchronized


def test(data,
         weights=None,
         batch_size=32,
         coco_eval=False,
         verbose=False,
         model=None,
         dataloader=None,
         plots=False,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,  # save auto-label confidences
         half_precision=True,
         is_coco=False):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

        # Directories
        save_dir = Path(increment_path(Path(os.path.join('outputs', opt.project)) / opt.name, exist_ok=opt.exist_ok))  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # Load model
        model = torch.load(weights, map_location=device)['model'].float()
        model.roi_heads.score_thresh = 0.01

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    # if half:
    #     model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    nc = int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        test_dataset = CleanDataset(img_dir=data['image_dir'], annotations_path=data[task],
                                    image_size=None, normalize_box=False)
        dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                                collate_fn=collate_fn, shuffle=False)
    else:
        test_dataset = dataloader.dataset
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map75, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(4, device=device)
    stats, ap, ap_class, wandb_images = [], [], [], []
    coco_pred = {'images': [], 'annotations': []}
    for batch_i, (img, targets, paths, _) in enumerate(tqdm(dataloader, desc=s)):
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0 note: already normalized

        targets = transform_to_faster_rcnn_targets(targets)

        img = list(x.to(device) for x in img)
        # if half:
        #     img = [x.half() for x in img]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        nb = len(img)  # batch size
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=half):
            with torch.no_grad():
                # Run model
                t = time_synchronized()
                loss_dict, detections = get_loss_and_detection(model, img, targets)  # inference and training outputs
                # detections = model(img)  # inference and training outputs
                t0 += time_synchronized() - t

                loss += torch.stack((loss_dict['loss_rpn_box_reg'], loss_dict['loss_box_reg'],
                                     loss_dict['loss_objectness'], loss_dict['loss_classifier'])).detach()

                # convert to proper format list of tensor [xyxy, conf, cls]
                out = []
                for det in detections:
                    temp = torch.zeros((len(det['boxes']), 6), device=device)
                    temp[:, :4] = det['boxes']
                    temp[:, 4] = det['scores']
                    temp[:, 5] = det['labels']
                    out.append(temp)

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[si]
            labels['labels'] = labels['labels'] - 1  # -1 to remove bg
            nl = len(labels['labels'])
            tcls = labels['labels'].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            # Predictions
            predn = pred.clone()

            # Append to text file
            if save_txt:
                for *xyxy, conf, cls in predn.tolist():
                    # xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()  # normalized xywh
                    line = (cls, *xyxy, conf) if save_conf else (cls, *xyxy)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Append to pycocotools JSON dictionary
            if coco_eval:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                img_id = len(coco_pred['images'])
                coco_pred['images'].append({'id': img_id})
                for p, b in zip(pred.tolist(), box.tolist()):
                    coco_pred['annotations'].append({'image_id': img_id,
                                                     'category_id': int(p[5]),
                                                     'bbox': [round(x, 3) for x in b],
                                                     'score': round(p[4], 5),
                                                     'id': len(coco_pred['annotations']),
                                                     'area': b[2] * b[3], 'iscrowd': 0
                                                     })

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels['labels']

                # target boxes
                tbox = labels['boxes']
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((tcls_tensor.unsqueeze(1), tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 2:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images2, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images2, args=(img, output_to_target2(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.75, AP@0.5:0.95
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (batch_size,)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Save JSON
    if coco_eval:
        print('\nEvaluating pycocotools mAP...')
        try:
            import copy
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            from utils.metrics import list_to_coco_obj
            coco_gt = COCO()
            coco_gt.dataset = list_to_coco_obj(test_dataset.annotations)
            coco_gt.createIndex()
            pred_gt = COCO()
            pred_gt.dataset = coco_pred
            pred_gt.dataset['categories'] = copy.deepcopy(coco_gt.dataset['categories'])
            pred_gt.createIndex()
            cocoEval = COCOeval(coco_gt, pred_gt, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50, map75 = cocoEval.stats[:3]
            print(f'coco mAP50: {map50:.5f}\tcoco mAP75: {map75:.5f}\tcoco mAP: {map:.5f}')
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, map75, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/voc_2007_ann5_0.7.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--task', default='test', help='train, val, test')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--coco-eval', action='store_true', help='run coco evaluation')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--plots', action='store_true', help='plot predictions')
    opt = parser.parse_args()
    # opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    # check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.coco_eval,
             opt.verbose,
             save_txt=opt.save_txt,
             save_conf=opt.save_conf,
             plots=opt.plots,
             )
