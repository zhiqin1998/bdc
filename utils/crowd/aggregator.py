import copy
import multiprocessing

import numpy as np
from abc import ABC, abstractmethod


def bbox_iou(box1, box2, x1y1x2y2=True, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = np.maximum(np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1), 0) * \
            np.maximum(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1), 0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou  # IoU


class Aggregator(ABC):
    """
    Abstract class for implementing aggregator methods
    """
    @abstractmethod
    def aggregate_single_crowd_label(self, labels):
        """Method to aggregate crowdsourced labels of a single sample,
           labels of a single sample will have shape (nl, 6) which is x1, y1, x2, y2, class_id, ann_id

           Returns aggregated labels which shapes should be (nl, 5) which is x1, y1, x2, y2, class_id
           Depending on the algorithm, additional elements can be returned too
        """

    def pretreat_crowd_labels(self, all_labels):
        """This method should be implemented if aggregator is used to pretreat data before training,
           else simply return the arguments (default).
           If unsure, you should implement as ```return self.aggregate_crowd_labels(all_labels)```
        """
        return all_labels

    def aggregate_crowd_labels(self, all_labels, num_workers=16):
        """
        Helper function to run aggregate_single_crowd_label on all data

        Parameters
        ----------
        all_labels: list[np.array]
            list of crowdsourced annotations with shape (nl, 6)
        num_workers: int
            number of parallel process to run, only activated if > 1 and len(all_labels) > 10000

        Returns
        -------
        list[np.array]
            list of aggregated annotations with shape (nl, 5)
        """
        if num_workers > 1 and len(all_labels) > 10000:
            with multiprocessing.Pool(num_workers) as pool:
                final = [copy.deepcopy(x) for x in pool.map(self.aggregate_single_crowd_label, all_labels)]
        else:
            final = []
            for labels in all_labels:
                final.append(copy.deepcopy(self.aggregate_single_crowd_label(labels)))
        return final


class NoAggregator(Aggregator):
    """Aggregator class that does nothing"""
    def aggregate_single_crowd_label(self, labels):
        """
        Aggregation is just simply removing the ann_id

        Parameters
        ----------
        labels: np.array
            Numpy array with shape (nl, 6) which is x1, y1, x2, y2, class_id, ann_id

        Returns
        -------
        np.array
            Numpy array of aggregated annotation
            with shape (nl, 5) which is x1, y1, x2, y2, class_id where ann_id is removed
        """
        # simply remove the ann_id from labels
        return labels[:, :5]

    def pretreat_crowd_labels(self, all_labels):
        """
        NoAggregator implements pretreat data before training

        Parameters
        ----------
        all_labels: list[np.array]
            list of crowdsourced annotations with shape (nl, 6)

        Returns
        -------
        list[np.array]
            list of aggregated annotations with shape (nl, 5)
        """
        return self.aggregate_crowd_labels(all_labels, num_workers=0)  # no need for multiprocessing

