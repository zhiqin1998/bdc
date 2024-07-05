# Model validation metrics https://github.com/WongKinYiu/yolov7/blob/main/utils/metrics.py
import os
import copy
import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics

from utils.general import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def compute_map(predictions, ground_truths, is_score=False):
    """
    Compute COCO mAP with pycocotools

    Parameters
    ----------
    predictions: list[np.array]
        list of N model predictions with shape (np, 6) [xyxy, cls, conf], or (np, 4+J) [xyxy, class logits]
    ground_truths: list[np.array]
        list of N ground truths with shape (ngt, 5) [xyxy, cls]
    is_score: bool
        Flag to indicate whether prediction is class id + confidence or full class logits (optional, default False)

    Returns
    -------
    float
        mAP50
    float
        mAP75
    float
        mAP
    """
    # ground_truths should be N x nl x 5; [xyxy cls]
    # predictions should be N x nl x 5; [xyxy cls] or may have logits as well
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):  # suppress pycocotools printing
        coco_gt = COCO()
        coco_gt.dataset = list_to_coco_obj(ground_truths)
        coco_gt.createIndex()
        pred_gt = COCO()
        pred_gt.dataset = list_to_coco_obj(predictions, is_score=is_score)
        pred_gt.dataset['categories'] = copy.deepcopy(coco_gt.dataset['categories'])
        pred_gt.createIndex()
        cocoEval = COCOeval(coco_gt, pred_gt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        ap, ap50, ap75 = cocoEval.stats[:3]
    return ap50, ap75, ap


def list_to_coco_obj(labels, is_score=False):
    """
    Convert list of ground truths or predictions into the expected format for pycocotools

    Parameters
    ----------
    labels: list[np.array]
        list of N model predictions with shape (n, 5) [xyxy, cls],
        or (n, 6) [xyxy, cls, conf], or (n, 4+J) [xyxy, class logits]
    is_score: bool
        Flag to indicate whether prediction is class id + confidence or full class logits (optional, default False)

    Returns
    -------
    dict
        Expected dictionary for pycocotools, have keys 'images', 'annotations' and 'categories'
    """
    labels = copy.deepcopy(labels)
    ann_id = 0
    max_category = -1
    d = {'images': [], 'annotations': []}
    for img_id, anns in enumerate(labels):
        d['images'].append({'id': img_id})
        is_logit = anns.shape[1] > 5 and not is_score
        for ann in anns:
            box = ann[:4]
            box[2:] = box[2:] - box[0:2]
            if is_logit:
                cat_id = ann[4:].argmax()
                score = ann[4:][cat_id]
            else:
                cat_id = int(ann[4])
                score = 1.
            if is_score:
                score = ann[5]
            score = min(max(score, 0.), 1.)
            max_category = max(max_category, cat_id)
            d['annotations'].append({'id': ann_id, 'image_id': img_id, 'category_id': cat_id, 'score': score,
                                     'bbox': box.astype(int).tolist(), 'area': box[2] * box[3], 'iscrowd': 0})
            ann_id += 1
    d['categories'] = [{'id': i, 'name': str(i)} for i in range(max_category + 1)]
    return d


if __name__ == '__main__':
    import os
    import sys

    sys.path.insert(0, '../')
    import yaml
    import numpy as np

    os.chdir('../')
    from torch.utils.data import DataLoader
    from datasets.datasets import SyntheticDataset, collate_fn
    from utils.crowd.majority_vote import MVAggregator
    from utils.crowd.earl import EARLAggregator
    from utils.crowd.aggregator import NoAggregator
    from utils.crowd.bdc import BDCAggregator
    from utils.general import init_seeds

    with open('./data/coco_ann1000_mix_disjoint.yaml') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    init_seeds(12)
    with open('./config/bdc.yaml') as f:
        ca_hyp = yaml.load(f, Loader=yaml.SafeLoader)
    aggregator = EARLAggregator(data_dict['n_annotator'])
    # aggregator = MVAggregator(data_dict['n_annotator'], box_iou_thres=0.45, box_combine_method='majority')
    # aggregator = NoAggregator()
    # aggregator = BDCAggregator(data_dict['n_annotator'], data_dict['nc'], data_dict['nc_ann'], **ca_hyp['parameters'])
    dataset = SyntheticDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['train'],
                               image_size=640, train=True, augments=None, aggregator=aggregator,
                               clean_annotations_path=data_dict['clean_train'],)# ref_box_path=data_dict['ref_box_train'])

    if isinstance(aggregator, BDCAggregator):
        print(ca_hyp)
        dataset.noisy_annotations = dataset.normalize_bbox(dataset.noisy_annotations)
        bdc_annotations, lower_bound = aggregator.initialize_dataset(dataset)
        assert len(dataset.annotations) == len(bdc_annotations), 'this shouldnt happen'
        dataset.annotations = dataset.normalize_bbox(bdc_annotations, inverse=True)
        print('prior map')
        print(compute_map([l[:, :-1] for l in dataset.annotations], dataset.clean_annotations, is_score=isinstance(aggregator, EARLAggregator)))
        bdc_annotations, lower_bound = aggregator.initialize_dataset(dataset)
        dataset.annotations = dataset.normalize_bbox(bdc_annotations, inverse=True)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True,
                            collate_fn=collate_fn, shuffle=False)
    if isinstance(aggregator, BDCAggregator):
        print('posterior map')
        print(compute_map([l[:, :-1] for l in dataset.annotations], dataset.clean_annotations, is_score=isinstance(aggregator, EARLAggregator)))
    else:
        print(compute_map(dataset.annotations, dataset.clean_annotations, is_score=isinstance(aggregator, EARLAggregator)))


def ap_per_class(tp, conf, pred_cls, target_cls, v5_metric=False, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], v5_metric=v5_metric)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision, v5_metric=False):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
        v5_metric: Assume maximum recall to be 1.0, as in YOLOv5, MMDetetion etc.
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    if v5_metric:  # New YOLOv5 metric, same as MMDetection and Detectron2 repositories
        mrec = np.concatenate(([0.], recall, [1.0]))
    else:  # Old YOLOv5 metric, i.e. default YOLOv7 metric
        mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py#L81
def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


def compute_ar(pred_boxes, gt_boxes, list_iou_thresholds=np.linspace(0.5, 0.95, 10)):
    AR = []
    for iou_threshold in list_iou_thresholds:
        try:
            recall, _ = compute_recall(pred_boxes, gt_boxes, iou=iou_threshold)
            AR.append(recall)
        except ValueError:
          AR.append(0.0)
          pass

    AUC = 2 * (metrics.auc(list_iou_thresholds, AR))
    return AUC
