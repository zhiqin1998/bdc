import cv2
import numpy as np

from utils.crowd.aggregator import Aggregator, bbox_iou
from scipy import stats


class MVAggregator(Aggregator):
    """
    Majority Voting Aggregator

    Attributes
    ----------
    n_annotator: int
        number of annotators, K
    box_iou_thres: float
        box IOU threshold to consider annotations as belonging to same instance
    box_combine_method: str
        Method of aggregating boxes, possible values are union, intersect, average, majority (default)
        1. union: Union of all boxes (OR)
        2. intersect: Intersection of all boxes (AND)
        3. average: Average xy coordinates of all boxes
        4. majority: The biggest bounding box where > K/2 had annotated
    """
    def __init__(self, n_annotator, box_iou_thres=0.5, box_combine_method='majority'):
        """
        Class initialisation

        Parameters
        ----------
        n_annotator: int
            number of annotators, K
        box_iou_thres: float
            box IOU threshold to consider annotations as belonging to same instance (optional, default 0.5)
        box_combine_method: str
            Method of aggregating boxes (optional), possible values are union, intersect, average, majority (default)
        """
        self.n_annotator = n_annotator
        self.box_iou_thres = box_iou_thres
        self.box_combine_method = box_combine_method.lower()

    def aggregate_single_crowd_label(self, labels):
        """
        Function to aggregate crowdsourced annotations of a single sample by performing MV

        Parameters
        ----------
        labels: np.array
            Numpy array with shape (nl, 6) which is x1, y1, x2, y2, class_id, ann_id

        Returns
        -------
        np.array
            Numpy array of aggregated annotation with shape (n_aggregated, 5) which is x1, y1, x2, y2, class_id
        """
        n_annotator = len(np.unique(labels[:, 5]))
        if n_annotator == 0:
            n_annotator = self.n_annotator
        groups = self.find_same_box(labels)
        return self.reduce_groups(groups, n_annotator)

    def find_same_box(self, labels):
        """
        Function to group similar annotations boxes recursively

        Parameters
        ----------
        labels: np.array
            Numpy array with shape (nl, 6) which is x1, y1, x2, y2, class_id, ann_id

        Returns
        -------
        list[np.array]
            List of Numpy array of found groups with shape (ng, 6)
        """
        def group_box(groups, labels):
            curr_box, labels = labels[0], labels[1:]
            match = False
            curr_ann_id = curr_box[-1]
            for i, group in enumerate(groups):
                if (group[:, -1] == curr_ann_id).any():  # group cannot have multiple labels of same annotator
                    continue
                match = (bbox_iou(curr_box[:4], group[:, :4]) > self.box_iou_thres).sum().item() > \
                        int(len(group) / 2)  # consider matched if over half of box has iou > threshold
                if match:
                    break
            if match:
                groups[i] = np.concatenate((groups[i], np.expand_dims(curr_box, 0)), axis=0)
            else:
                groups.append(np.expand_dims(curr_box, 0))
            return groups, labels

        groups = []  # list of array of array
        while len(labels):
            groups, labels = group_box(groups, labels)
        return groups

    def reduce_groups(self, groups, n_annotator):
        """
        Function to reduce/aggregate grouped annotations, removing if the majority did not annotate

        Parameters
        ----------
        groups: list[np.array]
            List of Numpy array of found groups with shape (ng, 6)
        n_annotator: int
            Number of annotator present in the annotations

        Returns
        -------
        np.array
            Numpy array of aggregated annotation with shape (n_aggregated, 5) which is x1, y1, x2, y2, class_id
        """
        labels = np.zeros((len(groups), 5))
        keep = []
        for i, group in enumerate(groups):
            # add missing annotations so that we skip no majority boxes
            majority_class_id = stats.mode(np.concatenate((group[:, 4],
                                           np.full((n_annotator - len(group),), -1))), keepdims=False)[0]
            if majority_class_id == -1:
                continue
            final_box = np.zeros((4,))
            if self.box_combine_method == 'union':
                final_box[:2] = group[:, :2].min(axis=0)
                final_box[2:4] = group[:, 2:4].max(axis=0)
            elif self.box_combine_method == 'intersect':
                final_box[:2] = group[:, :2].max(axis=0)
                final_box[2:4] = group[:, 2:4].min(axis=0)
            elif self.box_combine_method == 'average':
                final_box[:4] = group[:, :4].mean(axis=0).round()
            elif self.box_combine_method == 'majority':
                counter_box = np.zeros(group[:, 2:4].max(axis=0).astype(int) + 1, dtype=int)  # union box to count
                for box in group:
                    box = box.copy().astype(int)
                    counter_box[box[0]: box[2], box[1]: box[3]] += 1

                cnts, _ = cv2.findContours((counter_box > n_annotator // 2).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = max(cnts, key=lambda x: cv2.contourArea(x))  # get maximum area countour
                y, x, h, w = cv2.boundingRect(cnt)
                final_box[:4] = [x, y, x + w, y + h]
            else:
                raise ValueError(f'unsupported box combine method: {self.box_combine_method}')
            assert final_box[2] > final_box[0] and final_box[3] > final_box[1]
            labels[i, 4] = majority_class_id
            labels[i, :4] = final_box
            keep.append(i)
        return labels[keep, :]

    def pretreat_crowd_labels(self, all_labels):
        """
        Overriding super class function for preprocessing
        """
        return self.aggregate_crowd_labels(all_labels)


if __name__ == '__main__':
    agg = MVAggregator(n_annotator=3)
    labels = np.asarray([[  1.0240, 150.0000, 102.4000, 300.0000,   7.0000,   0.0000],
                        [ 47.1040, 147.6036, 144.3840, 264.4565,   7.0000,   0.0000],
                        [ 29.6960,  86.1021, 112.6400, 166.0540,   7.0000,   0.0000],
                        [400.3840,  87.6396, 484.3520, 190.6546,  19.0000,   0.0000],
                        [165.8880,  29.2132, 238.5920, 106.0901,   7.0000,   0.0000],
                        [430.0800, 152.2162, 512.0000, 273.6817,   7.0000,   0.0000],
                        [437.2480, 227.5556, 512.0000, 381.3093,   7.0000,   0.0000],
                        [163.8400, 206.0300, 311.2960, 355.1712,   7.0000,   1.0000],
                        [  1.0240, 169.1291, 102.4000, 319.8078,   7.0000,   1.0000],
                        [ 47.1040, 147.6036, 144.3840, 264.4565,   7.0000,   1.0000],
                        [ 29.6960,  86.1021, 112.6400, 166.0540,  14.0000,   1.0000],
                        [400.3840,  87.6396, 484.3520, 190.6546,  12.0000,   1.0000],
                        [165.8880,  29.2132, 238.5920, 106.0901,   7.0000,   1.0000],
                        [430.0800, 152.2162, 512.0000, 273.6817,   7.0000,   1.0000],
                        [437.2480, 227.5556, 512.0000, 381.3093,   1.0000,   1.0000],
                        [163.8400, 206.0300, 311.2960, 355.1712,   7.0000,   2.0000],
                        [  1.0240, 169.1291, 102.4000, 319.8078,   7.0000,   2.0000],
                        [ 47.1040, 147.6036, 144.3840, 264.4565,   7.0000,   2.0000],
                        [ 29.6960,  86.1021, 112.6400, 166.0540,  14.0000,   2.0000],
                        [400.3840,  87.6396, 484.3520, 190.6546,   7.0000,   2.0000],
                        [165.8880,  29.2132, 238.5920, 106.0901,   7.0000,   2.0000],
                        [430.0800, 152.2162, 512.0000, 273.6817,   7.0000,   2.0000],
                        [437.2480, 227.5556, 512.0000, 381.3093,   1.0000,   2.0000],
                        [  0.0000,  13.5556,   5.0000,  38.3093,   3.0000,   2.0000],
                        [  0.0000,  14.5556,   4.0000,  40.3093,   3.0000,   1.0000]])
    labels[:, :4] = labels[:, :4]
    print(agg.aggregate_single_crowd_label(labels))
