import copy

import numpy as np
import ensemble_boxes

from utils.crowd.aggregator import Aggregator


def build_earl_ann_weights(weights):
    """
    Helper function to build the annotator weights for EARLAggregator

    Parameters
    ----------
    weights: list of 2 element tuples
        List of tuples of (ending annotator id, weight)
        e.g. [(25, 0.5), (30, 0.6)] represent 0-24 annotator with weight 0.5 and 25-29 with weight 0.6

    Returns
    -------
    dict{int: float}
        Dictionary mapping of annotator id to its weight
    """
    ann_weights = {}
    start_id = 0
    for end_id, weight in weights:
        for _id in range(start_id, end_id):
            ann_weights[_id] = weight
        start_id = end_id
    return ann_weights


class EARLAggregator(Aggregator):
    """
    Weighted Boxes Fusion - Expert Agreement Re-weighted Loss Aggregator
    https://ieeexplore.ieee.org/document/10041153

    Code referenced from https://github.com/huyhieupham/learning-from-multiple-annotators/blob/main/source_det/processing.py

    Attributes
    ----------
    n_annotator: int
        number of annotators, K
    ann_weight: dict{int: float}
        Dictionary mapping of annotator id to its weight
    """
    def __init__(self, n_annotator, ann_weight=None):
        """
        Class initialisation

        Parameters
        ----------
        n_annotator: int
            number of annotators, K
        ann_weight: list of 2 element tuples
            List of tuples of (ending annotator id, weight)
            e.g. [(25, 0.5), (30, 0.6)] represent 0-24 annotator with weight 0.5 and 25-29 with weight 0.6
        """
        self.n_annotator = n_annotator
        if ann_weight is None:
            ann_weight = {}
        else:
            ann_weight = build_earl_ann_weights(ann_weight)
        self.ann_weight = ann_weight

    def aggregate_single_crowd_label(self, labels):
        """
        Function to aggregate crowdsourced annotations of a single sample by performing WBF
        and calculate weight for EARL

        Parameters
        ----------
        labels: np.array
            Numpy array with shape (nl, 6) which is x1, y1, x2, y2, class_id, ann_id

        Returns
        -------
        np.array
            Numpy array of aggregated annotation
            with shape (nl, 6) which is x1, y1, x2, y2, class_id, weight for EARL
        """
        # group label by annotator id and then perform wbf
        if not len(labels):
            return np.zeros((0, 6), dtype=float)
        labels = copy.deepcopy(labels)
        boxes_list, labels_list, scores_list, weights = [], [], [], []
        max_x = labels[:, [0, 2]].max()
        max_y = labels[:, [1, 3]].max()
        for i in range(self.n_annotator):
            ann_box = copy.deepcopy(labels[labels[:, 5] == i, :5])
            boxes_list.append(ann_box[:, :4] / np.asarray([[max_x, max_y, max_x, max_y]]))
            labels_list.append(ann_box[:, 4])
            scores_list.append(np.ones((len(ann_box),), dtype=np.float32))
            weights.append(self.ann_weight.get(i, 1.))
        # wrong implementation in original code: score and label swapped, iou_thr should be 0.4
        # fused_boxes, fused_labels, fused_scores = ensemble_boxes.weighted_boxes_fusion(
        #     # boxes_list, labels_list, scores_list
        #     boxes_list, scores_list, labels_list
        #     , weights=weights, iou_thr=0.5
        # )
        fused_boxes, fused_scores, fused_labels = ensemble_boxes.weighted_boxes_fusion(
            boxes_list, scores_list, labels_list
            , weights=weights, iou_thr=0.4
        )
        return np.concatenate((fused_boxes * np.asarray([[max_x, max_y, max_x, max_y]]),
                               np.expand_dims(fused_labels, axis=-1).astype(int),
                               # they use count of annotator in wbf as c in earl which is score * n_annotator if all annotator weight is 1
                               np.expand_dims(fused_scores * self.n_annotator, axis=-1).astype(int)),
                              # np.expand_dims(fused_scores, axis=-1).astype(float)),
                              axis=1)

    def pretreat_crowd_labels(self, all_labels):
        """
        Overriding super class function for preprocessing with WBF
        """
        return self.aggregate_crowd_labels(all_labels)


if __name__ == '__main__':
    import os
    import sys

    sys.path.insert(0, '../../')
    os.chdir('../../')
    import yaml
    import torch
    from torch.utils.data import DataLoader
    from datasets.datasets import SyntheticDataset, collate_fn, get_augmentation

    with open('./data/voc_2007_ann25_mix.yaml') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

    aggregator = EARLAggregator(data_dict['n_annotator'], ann_weight=[])
    print(aggregator.ann_weight)

    labels = np.asarray([[185., 106., 337., 190.,   0.,   0.],
                         [185., 106., 337., 190.,   0.,   1.],
                         [185., 106., 337., 190.,   0.,   2.],
                         [185., 106., 337., 190.,   0.,   3.]]
                        )
    print(aggregator.aggregate_single_crowd_label(labels))

    augments = get_augmentation(hue=data_dict['hsv_h'], sat=data_dict['hsv_s'], val=data_dict['hsv_v'],
                                translate=data_dict['translate'], scale=data_dict['scale'], rotate=data_dict['rotate'],
                                shear=data_dict['shear'], perspective=data_dict['perspective'],
                                fliplr=data_dict['fliplr'], flipud=data_dict['flipud'])
    dataset = SyntheticDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['train'],
                               image_size=512, train=True, augments=augments, aggregator=aggregator,
                               clean_annotations_path=data_dict['clean_train'])
    for i in range(len(dataset)):
        ann = dataset.annotations[i]
        if (ann < 0).any():
            print(i)
            print(dataset.noisy_annotations[i])
            print(dataset.clean_annotations[i])
