import copy
import multiprocessing
from itertools import repeat

import torch
import numpy as np
import scipy.special as ss
import torchvision
from scipy import stats
from scipy.special import softmax

from utils.crowd.aggregator import Aggregator, bbox_iou
from utils.general import xyxy2xywh, xywh2xyxy


class BDCAggregator(Aggregator):
    """
    Bayesian Detector Combination: an annotation aggregation method that model the annotator class label and bounding
    box accuracy as multinomial and Gaussian distribution with Dirichlet and Gaussian-Gamma priors

    Examples
    --------
    To use the aggregator, you must first initialise the aggregator with the training dataset
    ```
    dataset.noisy_annotations = dataset.normalize_bbox(dataset.noisy_annotations)  # normalise the bounding boxes first
    bdc_annotations, lower_bound = aggregator.initialize_dataset(dataset)
    dataset.annotations = dataset.normalize_bbox(bdc_annotations, inverse=True)  # unnormalise the bounding boxes
    ```
    Then, during each epoch end, use the model prediction on the training images to update the aggregator parameters
    and aggregate the crowdsourced annotations
    ```
    # out_with_logits contains N model predictions of [xyxy (normalised), J class logits]
    new_annotations, lower_bound = aggregator.fit_transform_crowd_labels(dataset.noisy_annotations,
                                                                         out_with_logits, warmup=opt.bdc_warmup)
    dataset.annotations = dataset.normalize_bbox(new_annotations, inverse=True)  # unnormalise the bounding boxes
    ```

    Attributes
    ----------
    n_annotator: int
        number of annotator, K
    n_true_cls: int
        number of classes, J
    n_ann_cls: int
        number of classes by annotator, L (usually equal to J)
    box_iou_thres: float
        box IOU threshold for first time initialisation only
    conf_thres: float
        confidence threshold to remove low confidence aggregations
    match_iou_cost: float
        the parameter lambda_1 (for IoU cost) in the matching cost function
    match_bbox_cost: float
        the parameter lambda_2 (for bbox L1 cost) in the matching cost function
    step: int
        number of update steps, for logging purpose
    alpha0: np.array
        alpha0 of Dirichlet prior for each annotator, has shape (K, J, L)
    mu0: np.array
        mu0 of Gaussian-Gamma prior of x and y translation and scaling error for each annotator, has shape (K, 4)
        Our initial belief of the mean
    count_prior0: np.array
        count0 of Gaussian-Gamma prior of x and y translation and scaling error for each annotator, has shape (K, 4)
        The strength of the belief of the initial mean
    upsilon0: np.array
        upsilon0 of Gaussian-Gamma prior of x and y translation and scaling error for each annotator, has shape (K, 4)
        variance is beta/upsilon
    beta0: np.array
        beta0 of Gaussian-Gamma prior of x and y translation and scaling error for each annotator, has shape (K, 4)
        variance is beta/upsilon
    alpha: np.array
        alpha of the updated Dirichlet prior for each annotator, has shape (K, J, L)
    mu: np.array
        mu of the updated  Gaussian-Gamma prior of x and y translation and scaling error for each annotator,
        has shape (K, 4)
    upsilon: np.array
        upsilon of the updated  Gaussian-Gamma prior of x and y translation and scaling error for each annotator,
        has shape (K, 4)
    beta: np.array
        beta of the updated  Gaussian-Gamma prior of x and y translation and scaling error for each annotator,
        has shape (K, 4)
    """
    def __init__(self, n_annotator, n_true_cls, n_ann_cls, box_iou_thres=0.45, conf_thres=0.3,
                 alpha0_diag_prior=10, mu0_prior=0., count_prior=10, upsilon0_prior=10, beta0_prior=0.5,
                 match_iou_cost=2, match_bbox_cost=5):
        """
        Class initialisation

        Parameters
        ----------
        n_annotator: int
            number of annotator, K
        n_true_cls: int
            number of classes, J
        n_ann_cls: int
            number of classes by annotator, L (usually equal to J)
        box_iou_thres: float
            box IOU threshold for first time initialisation only (optional, default 0.4)
        conf_thres: float
            confidence threshold to remove low confidence aggregations (optional, default 0.3)
        alpha0_diag_prior: float
            Diagonal value of alpha0 of the Dirichlet prior (optional, default 10)
        mu0_prior: float
            Value of mu0 of the Gaussian-Gamma prior (optional, default 0)
            Our initial belief of the mean
        count_prior: float
            Value of count prior of the Gaussian-Gamma prior (optional, default 10)
            The strength of the belief of the initial mean
        upsilon0_prior: float
            Value of upsilon0 of the Gaussian-Gamma prior (optional, default 10)
            variance is beta/upsilon
        beta0_prior: float
            Value of beta0 of the Gaussian-Gamma prior (optional, default 0.5)
            variance is beta/upsilon
        match_iou_cost: float
            the parameter lambda_1 (for IoU cost) in the matching cost function (optional, default 2)
        match_bbox_cost: float
            the parameter lambda_2 (for bbox L1 cost) in the matching cost function (optional, default 5)
        """
        self.n_annotator = n_annotator  # K
        self.n_true_cls = n_true_cls  # J
        self.n_ann_cls = n_ann_cls  # L
        self.box_iou_thres = box_iou_thres
        self.conf_thres = conf_thres
        self.match_iou_cost = match_iou_cost
        self.match_bbox_cost = match_bbox_cost
        self.step = 0  # for logging

        self.alpha0 = self._initialize_alpha0(alpha0_diag_prior)  # K x J x L
        self.mu0 = self._initialize_prior(mu0_prior)  # K x 4
        self.count_prior0 = self._initialize_prior(count_prior)
        self.upsilon0 = self._initialize_prior(upsilon0_prior)  # K x 4, alpha in murphy notes
        self.beta0 = self._initialize_prior(beta0_prior)  # K x 4
        self.alpha = copy.deepcopy(self.alpha0)
        self.mu = copy.deepcopy(self.mu0)
        self.upsilon = copy.deepcopy(self.upsilon0)
        self.beta = copy.deepcopy(self.beta0)

    def _initialize_alpha0(self, diag_prior):
        """
        Helper function to initialise alpha0

        Parameters
        ----------
        diag_prior: float
            Value of diagonal for alpha0

        Returns
        -------
        np.array
            Numpy array of shape (K, J, L)
        """
        a0 = np.ones((self.n_true_cls, self.n_ann_cls), dtype=np.float64)
        np.fill_diagonal(a0, diag_prior)
        return np.repeat(np.expand_dims(a0, 0), self.n_annotator, 0)

    def _initialize_prior(self, prior):
        """
        Helper function to initialise other priors

        Parameters
        ----------
        prior: float
            Initial value of the prior

        Returns
        -------
        np.array
            Numpy array of shape (K, 4) with value prior
        """
        return np.zeros((self.n_annotator, 4)) + prior

    def aggregate_single_crowd_label(self, labels, model_outputs):
        """
        Function to aggregate single crowdsourced annotations without updating the priors.
        This should be used outside model training. All box coordinates are normalised to [0, 1].

        Parameters
        ----------
        labels: np.array
            Numpy array with shape (nl, 6) which is x1, y1, x2, y2, class_id, ann_id
        model_outputs: np.array
            Numpy array with shape (no, 4+J) which is x1, y1, x2, y2, J classes probabilities from the OD model

        Returns
        -------
        np.array
            Numpy array of aggregated annotation
            with shape (nl, 4+J) which is x1, y1, x2, y2, J aggregated class probabilities
        """
        return self.fit_transform_crowd_labels([labels], [model_outputs], update=False)[0]

    def aggregate_crowd_labels(self, all_labels, model_outputs):
        """
        Helper function to aggregate crowdsourced annotations without updating the priors.
        This should be used outside model training. All box coordinates are normalised to [0, 1].

        Parameters
        ----------
        all_labels: list[np.array]
            list of crowdsourced annotations with shape (nl, 6)
        model_outputs: list[np.array]
            list of model predictions with shape (no, 4+J)

        Returns
        -------
        list[np.array]
            list of aggregated annotations with shape (nl, 4+J)
        """
        return self.fit_transform_crowd_labels(all_labels, model_outputs, update=False)

    def initialize_dataset(self, dataset):
        """Function to initialise dataset with random model outputs. All box coordinates are normalised to [0, 1].

        Parameters
        ----------
        dataset: datasets.datasets.SyntheticDataset
            Dataset object that contains the crowdsourced annotations

        Returns
        -------
        list[np.array]
            list of aggregated annotations with shape (nl, 4+J)
        float
            the lower bound (elbo)
        """
        noisy_annotations = copy.deepcopy(dataset.noisy_annotations)
        random_outputs = self.get_outputs(noisy_annotations)
        return self.fit_transform_crowd_labels(noisy_annotations, random_outputs, warmup=-1)

    def get_single_output(self, noisy_annotation):
        """Function to randomly generate a model prediction given the annotations, only to be used for initialisation

        Parameters
        ----------
        noisy_annotation: np.array
            Numpy array of crowdsourced annotations with shape (nl, 6) which is x1, y1, x2, y2, class_id, ann_id

        Returns
        -------
        np.array
            Numpy array with shape (no, 4+J) which is x1, y1, x2, y2, J classes probabilities
        """
        groups = self.find_same_box(noisy_annotation)  # find group of similar annotations
        output = np.zeros((len(groups), 4 + self.n_true_cls), dtype=np.float32)
        for i, g in enumerate(groups):
            # output[i, :4] = np.concatenate((g[:, :2].min(axis=0), g[:, 2:4].max(axis=0)))  # union
            output[i, :4] = g[:, :4].mean(axis=0)  # mean
            # output[i, 4:] = np.eye(self.n_true_cls)[stats.mode(g[:, 4].astype(int), keepdims=False)[0]]
            output[i, 4:] = np.random.random(self.n_true_cls) + np.eye(self.n_true_cls)[
                stats.mode(g[:, 4].astype(int), keepdims=False)[0]]
            output[i, 4:] /= output[i, 4:].max()  # normalise
        return output

    def get_outputs(self, noisy_annotations, num_workers=16):
        """Helper function to multiprocess self.get_single_output function, only to be used for initialisation

        Parameters
        ----------
        noisy_annotations: list[np.array]
            list of Numpy array of crowdsourced annotations with shape (nl, 6)
        num_workers: int
            number of parallel process to run, only activated if > 1 and len(all_labels) > 10000

        Returns
        -------
        list[np.array]
            list of Numpy array of random predictions with shape (no, 4+J)
        """
        if num_workers > 1 and len(noisy_annotations) > 10000:  # multiprocessing for large dataset
            with multiprocessing.Pool(num_workers) as pool:
                random_outputs = pool.map(self.get_single_output, noisy_annotations)
        else:
            random_outputs = []
            for j, noisy_annotation in enumerate(noisy_annotations):
                random_outputs.append(self.get_single_output(noisy_annotation))
        return random_outputs

    def fit_transform_crowd_labels(self, all_labels, model_outputs, update=True, warmup=10):
        """
        Function to aggregate crowdsourced annotations and update the prior distributions by:
        1. Matching crowdsourced annotations to model predictions
        2. Aggregate the bounding boxes
        3. Aggregate the class labels
        4. Update the prior distributions
        All box coordinates are normalised to [0, 1]. Also returns the lower bound for debugging/logging purpose

        TODO: may need to batch this function to reduce memory usage, or use float32, or use sparse array

        Parameters
        ----------
        all_labels: list[np.array]
            list of N Numpy array of crowdsourced annotations with shape (nl, 6); [xyxy cls_id ann_id]
        model_outputs: list[np.array]
            list of N Numpy array of model predicitons with shape (no, 4+J) [xyxy class_logits]
            (sorted by desc confidence)
        update: bool
            Flag of whether to update the prior distributions (optional, default True)
        warmup: int
            Number of steps for warming up the model (optional, default 10)

        Returns
        -------
        list[np.array]
            list of N aggregated annotations with shape (nl, 4+J)
        float
            the lower bound (elbo)
        """
        all_labels = copy.deepcopy(all_labels)
        if self.step >= warmup:
            model_outputs = copy.deepcopy(model_outputs)
        else:
            model_outputs = self.get_outputs(copy.deepcopy(all_labels))

        assert len(all_labels) == len(model_outputs), 'this shouldnt happen'
        matched_labels, matched_model_outputs, img_indexes, weights, labelled_masks = self.match_boxes(all_labels, model_outputs)
        b_n, ann_err = self.estimate_true_box(matched_labels, matched_model_outputs[:, :4], labelled_masks)
        q_t, expected_ln_pi, njl, ln_rho = self.estimate_true_labels(matched_labels, matched_model_outputs[:, 4:], labelled_masks)
        lower_bound = 0
        if update:
            self.update_alpha(njl)
            self.update_gauss_gamma(ann_err)
            self.step += 1
            lower_bound = self.compute_lower_bound(q_t, expected_ln_pi, ln_rho, matched_model_outputs[:, 4:])
        restored_q_t = self._restore_labels(q_t, b_n, img_indexes, weights, len(all_labels), self.conf_thres)
        assert len(all_labels) == len(model_outputs) == len(restored_q_t), 'this shouldnt happen'
        return restored_q_t, lower_bound

    def match_boxes(self, all_labels, model_outputs):
        """
        Function to match crowdsourced annotations to model predictions. All box coordinates are normalised to [0, 1].
        The function also flatten the list of arrays into np.array with shape[0] = N x n_match

        Parameters
        ----------
        all_labels: list[np.array]
            list of N crowdsourced annotations with shape (nl, 6)
        model_outputs: list[np.array]
            list of N model predictions with shape (no, 4+J)

        Returns
        -------
        list[np.array]
            K list of matched annotations for each annotator
        np.array
            matched (flatten) model outputs with shape (N x n_match, 4+J)
        np.array
            image indexes of the matched crowdsourced annotations and model outputs with length N x n_match
            this used to reconstruct the array after aggregation back to the original format
        np.array
            weights of the matched pairs with length N x n_match, used for reweighting the loss function
        np.array
            boolean mask of shape (N x n_match, K) to indicate which annotator is involved in the matched pairs
            this is implemented to reduce the memory footprint when K is large
        """
        # matched_labels should be n_annotator x (N x n_match) x 5: [xyxy cls]
        # matched_model_outputs should be (N x n_match) x (4+J) [xyxy, class_logits]
        matched_model_outputs, img_indexes, weights, labelled_masks = [], [], [], []
        matched_labels = [[] for _ in range(self.n_annotator)]
        for i, (labels, outputs) in enumerate(zip(all_labels, model_outputs)):
            if not len(labels):  # skip if no label
                continue
            if not len(outputs):
                # if no output, model must be in early stage (use random output)
                outputs = self.get_outputs([labels])[0]
            # n annotator in image
            n_annotator_img = len(np.unique(labels[:, 5]))
            n_annotator_img = n_annotator_img if n_annotator_img > 0 else self.n_annotator

            # calculate cost matrix
            iou_cost = -torchvision.ops.box_iou(torch.tensor(outputs[:, :4]), torch.tensor(labels[:, :4])).numpy()
            l1_cost = torch.cdist(torch.tensor(outputs[:, :4]), torch.tensor(labels[:, :4]), p=1).numpy()
            output_probs = softmax(outputs[:, 4:], axis=1)
            cls_cost = -output_probs[:, labels[:, 4].astype(int)]
            cost_matrix = (cls_cost + self.match_iou_cost * iou_cost + self.match_bbox_cost * l1_cost).T  # nl x no
            # find one to many matching with minimum cost
            ann_assigned_matrix = np.zeros((len(outputs), self.n_annotator), dtype=bool)
            output_to_label_matching = [[] for _ in range(len(outputs))]
            min_cost_outputs = cost_matrix.argmin(axis=1)
            for j, (label, min_output_idx) in enumerate(zip(labels, min_cost_outputs)):
                if not ann_assigned_matrix[min_output_idx, int(label[-1])]:  # output can only match one per annotator
                    output_to_label_matching[min_output_idx].append(label)
                elif len(outputs) > 1:
                    sorted_idxs = np.argsort(cost_matrix[j])
                    smallest_idx = 1
                    min_output_idx = sorted_idxs[smallest_idx]
                    while ann_assigned_matrix[min_output_idx, int(label[-1])] and smallest_idx + 1 < len(outputs):
                        smallest_idx += 1
                        min_output_idx = sorted_idxs[smallest_idx]
                    if not ann_assigned_matrix[min_output_idx, int(label[-1])]:
                        output_to_label_matching[min_output_idx].append(label)
                ann_assigned_matrix[min_output_idx, int(label[-1])] = True

            # build required retval
            for j, curr_match in enumerate(output_to_label_matching):
                if len(curr_match):
                    curr_match = np.stack(curr_match)

                    labelled_mask = np.zeros((self.n_annotator,), dtype=bool)
                    labelled_mask[curr_match[:, 5].astype(int)] = True

                    curr_output = outputs[j]
                    weight = len(curr_match) / n_annotator_img

                    # matched_labels.append(curr_labels)
                    for match in curr_match:
                        matched_labels[int(match[5])].append(match[:5])
                    matched_model_outputs.append(curr_output)
                    img_indexes.append(i)
                    weights.append(weight)
                    labelled_masks.append(labelled_mask)
        matched_labels = [np.stack(x) if len(x) else np.zeros((0, 5)) for x in matched_labels]
        return matched_labels, np.stack(matched_model_outputs), np.asarray(img_indexes, dtype=np.int32), np.asarray(weights), np.stack(labelled_masks)

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
            List of Numpy array of found group with shape (n_group, 6)
        """
        # only used on first timestep for initialisation
        def group_box(groups, labels):
            curr_box, labels = labels[0], labels[1:]
            found_i = -1
            found_avg_iou = -1
            curr_ann_id = curr_box[5]
            for i, group in enumerate(groups):
                if (group[:, 5] == curr_ann_id).any():  # group cannot have multiple labels of same annotator
                    continue
                curr_box_iou = bbox_iou(curr_box[:4], group[:, :4])
                match = (curr_box_iou > self.box_iou_thres).sum().item() > int(len(group) / 2 - 1) # consider matched if over half of box has iou > threshold
                if match:
                    if found_i >= 0:
                        curr_mean_iou = np.mean(curr_box_iou)
                        if curr_mean_iou > found_avg_iou:
                            found_i = i
                            found_avg_iou = np.mean(curr_box_iou)
                    else:
                        found_i = i
                        found_avg_iou = np.mean(curr_box_iou)

            if found_i >= 0:
                groups[found_i] = np.concatenate((groups[found_i], np.expand_dims(curr_box, 0)), axis=0)
            else:
                groups.append(np.expand_dims(curr_box, 0))
            return groups, labels

        if not len(labels):
            return []

        # groups = []  # list of array of array
        ann_mask = labels[:, 5] == labels[:, 5].min()  # initialize group with one annotator
        min_annotations = labels[ann_mask]
        groups = [np.expand_dims(x, 0) for x in min_annotations]
        labels = labels[~ann_mask]
        while len(labels):
            groups, labels = group_box(groups, labels)
        return groups

    def estimate_true_box(self, matched_labels, matched_model_outputs, labelled_masks):
        """
        Function to aggregate and compute the box error with the matched annotations-predictions from self.match_boxes
        All box coordinates are normalised to [0, 1].

        Parameters
        ----------
        matched_labels: list[np.array]
            K list of matched annotations for each annotator
        matched_model_outputs: np.array
            matched (flatten) model outputs with shape (N x n_match, 4+J)
        labelled_masks: np.array
            boolean mask of shape (N x n_match, K) to indicate which annotator is involved in the matched pairs
            this is implemented to reduce the memory footprint when K is large

        Returns
        -------
        np.array
            aggregated bounding box of shape (N x n_match, 4)
        list[np.array]
            bounding box error in terms of x and y translational and scaling error for each annotator
            K list of numpy array with shape (n_annotated, 4)
        """
        output_boxes = xyxy2xywh(matched_model_outputs[:, :4].copy())  # n_match x 4 [xyxy]
        true_box = np.zeros((labelled_masks.shape[0], 4))
        weights = np.zeros((labelled_masks.shape[0], 4))
        ann_errs = []  # K x Nk x 4
        # correct box with expected value
        for k in range(self.n_annotator):
            expected_mu = self.mu[k]
            precision = self._expected_precision(k)
            labelled_mask = labelled_masks[:, k]
            if sum(labelled_mask):  # only do something if the annotator label something
                xywh = xyxy2xywh(matched_labels[k][:, :4])
                # get error relative to width and height
                ann_err = np.zeros((sum(labelled_mask), 4), dtype=np.float32)
                # translational error
                ann_err[:, :2] = (xywh[:, :2] - output_boxes[labelled_mask, :2]) / output_boxes[labelled_mask, 2:]
                # scaling error
                ann_err[:, 2:] = (xywh[:, 2:] / output_boxes[labelled_mask, 2:]) - 1
                ann_errs.append(ann_err)

                # correcting boxes based on posterior Gaussian mean
                xywh[:, :2] -= expected_mu[:2] * output_boxes[labelled_mask, 2:]
                xywh[:, 2:] = xywh[:, 2:] / (1 + expected_mu[2:])
                true_box[labelled_mask] += xywh * precision
                weights[labelled_mask] += precision
            else:
                ann_errs.append(np.zeros((0, 4), dtype=np.float32))
        # weighted average with precision of each annotator as weights
        true_box = true_box / weights
        true_box = np.clip(xywh2xyxy(true_box), 0., 1.)
        # assert (true_box[:, 0] < true_box[:, 2]).all() and (true_box[:, 1] < true_box[:, 3]).all()
        return true_box, ann_errs

    def update_gauss_gamma(self, ann_errs):
        """
        Function to update the Gaussian-Gamma priors of each annotator given the bounding box errors
        Derivation from https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

        Parameters
        ----------
        ann_errs: list[np.array]
            bounding box error in terms of x and y translational and scaling error for each annotator
            K list of numpy array with shape (n_annotated, 4)
        """
        for k in range(self.n_annotator):
            # update prior for each annotator
            samples = ann_errs[k]
            sample_count = len(samples)
            if sample_count:
                sample_mean = samples.mean(axis=0)
                self.mu[k] = (self.mu0[k] * self.count_prior0[k] + sample_mean * sample_count) / (self.count_prior0[k] + sample_count)
                self.upsilon[k] = self.upsilon0[k] + sample_count / 2
                first_term = 0.5 * ((samples - sample_mean) ** 2).sum(axis=0)
                second_term = ((sample_count * self.count_prior0[k]) / (self.count_prior0[k] + sample_count)) * \
                              ((sample_mean - self.mu0[k]) ** 2 / 2)
                self.beta[k] = self.beta0[k] + first_term + second_term
        # clip to sensible values
        self.mu[:, :2] = np.clip(self.mu[:, :2], -0.5, 0.5)
        self.mu[:, 2:] = np.clip(self.mu[:, 2:], -0.9, 0.9)

    def _expected_precision(self, k=None):
        """
        Function to get the expected precision of annotators

        Parameters
        ----------
        k: int
            index of annotator to return its expected precision
            (optional, default None will return precision of all annotators)

        Returns
        -------
        np.array
            return array of length 4 if k is not None, else return Numpy array of shape (K, 4)
        """
        if k is not None:
            return self.upsilon[k] / self.beta[k]
        return self.upsilon / self.beta

    @staticmethod
    def _ln_B(a):
        """
        Helper function to compute the ln of (sum of gamma(i) for all i in a/gamma of sum(a))

        Parameters
        ----------
        a: np.array
            Numpy array of shape (K, J, L) for the Dirichlet alpha

        Returns
        -------
        float
            ln of (sum of gamma(i) for all i in a/gamma of sum(a))
        """
        # ln of (sum of gamma(i) for all i in a/gamma of sum(a))
        return np.sum(ss.gammaln(a)) - ss.gammaln(np.sum(a))

    def _expected_ln_pi(self):
        """
        Function to compute the expected ln pi of the current alpha. For usage in self.estimate_true_labels

        Returns
        -------
        np.array
            The expected ln pi for each annotator, has shape (K, J, L)
        """
        # pi[k, j, l] = psi(alpha[k, j, l]) - psi(sum(alpha[k, j, over all l]))
        return ss.psi(self.alpha) - ss.psi(np.sum(self.alpha, axis=2, keepdims=True))

    def estimate_true_labels(self, crowd_labels, model_outputs, labelled_masks):
        """
        Function to aggregate class labels and count the Njl required for updating the Dirichlet prior

        Parameters
        ----------
        crowd_labels: list[np.array]
            K list of matched class labels for each annotator
        model_outputs: np.array
            matched (flatten) model class probabilities predictions with shape (N x n_match, J)
        labelled_masks: np.array
            boolean mask of shape (N x n_match, K) to indicate which annotator is involved in the matched pairs
            this is implemented to reduce the memory footprint when K is large

        Returns
        -------
        np.array
            aggregated class label probabilities (N x n_match, J)
        np.array
            The expected ln pi for each annotator, has shape (K, J, L)
        np.array
            Njl count, that is used to update the Dirichlet prior, has shape (K, J, L)
        np.array
            unnormalised value of ln rho, that is used to compute lower bound, has shape (N x n_match, J)
        """
        expected_ln_pi = self._expected_ln_pi()
        ln_rho = copy.deepcopy(model_outputs)
        crowd_labels = [x[:, 4].astype(int) for x in crowd_labels]

        for k in range(self.n_annotator):
            labelled_mask = labelled_masks[:, k]  # -1 is missing crowd annotation for the instance
            ln_rho[labelled_mask, :] = ln_rho[labelled_mask, :] + expected_ln_pi[k, :, crowd_labels[k]]

        ln_rho = ln_rho - np.max(ln_rho, axis=1, keepdims=True)

        # normalizing step and avoid div by 0
        q_t = np.exp(ln_rho) / np.maximum(np.sum(np.exp(ln_rho), axis=1, keepdims=True), 1e-10)

        return q_t, expected_ln_pi, self.count_njl(q_t, crowd_labels, labelled_masks), ln_rho

    @staticmethod
    def _unique_to_index(values):
        """
        Helper function to get the indexes of all unique value in the array.
        This is used to speed up repetitive indexing

        Parameters
        ----------
        values: np.array or list[int]
            input array to get indexes of all unique value

        Returns
        -------
        dict{int: list[int]}
            Dictionary mapping of unique values to their corresponding indexes in parameter values
        """
        unq, unq_inv, unq_cnt = np.unique(values, return_inverse=True, return_counts=True)
        index = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
        return {k: v for k, v in zip(unq, index)}

    def count_njl(self, q_t, crowd_labels, labelled_masks, num_workers=16):
        """
        Function to compute Njl required for updating the Dirichlet prior

        Parameters
        ----------
        q_t: np.array
            aggregated class label probabilities (N x n_match, J)
        crowd_labels: list[np.array]
            K list of matched class labels for each annotator
        labelled_masks: np.array
            boolean mask of shape (N x n_match, K) to indicate which annotator is involved in the matched pairs
            this is implemented to reduce the memory footprint when K is large
        num_workers: int
            number of parallel process to run, only activated if > 1 and K > 100

        Returns
        -------
        np.array
            Njl count, that is used to update the Dirichlet prior, has shape (K, J, L)
        """
        if num_workers > 1 and self.n_annotator >= 100:
            with multiprocessing.Pool(num_workers) as pool:
                njl = pool.starmap(self.count_njl_k, zip(repeat(q_t), crowd_labels, labelled_masks.T))
            njl = np.stack(njl)
        else:
            njl = np.zeros((self.n_annotator, self.n_true_cls, self.n_ann_cls), dtype=np.float32)
            for k in range(self.n_annotator):
                njl[k] = self.count_njl_k(q_t, crowd_labels[k], labelled_masks[:, k])
        return njl

    def count_njl_k(self, q_t, crowd_label, labelled_mask):
        """
        Helper function to compute Njl for a single annotator

        Parameters
        ----------
        q_t: np.array
            aggregated class label probabilities (N x n_match, J)
        crowd_label: np.array
            matched class labels for single annotator
        labelled_mask: np.array
            boolean mask of length (N x n_match) to indicate which matched pairs is annotated by the annotator
            this is implemented to reduce the memory footprint when K is large

        Returns
        -------
        np.array
            Njl count, that is used to update the Dirichlet prior, has shape (K, J, L)
        """
        njl = np.zeros((self.n_true_cls, self.n_ann_cls), dtype=np.float32)
        idx_dict = self._unique_to_index(crowd_label)
        for l in range(self.n_ann_cls):
            idx = idx_dict.get(l, np.empty((0,), dtype=int))
            njl[:, l] = np.sum(q_t[labelled_mask][idx, :], axis=0)
        return njl

    def update_alpha(self, njl):
        """
        Helper function to update the Dirichlet prior, alpha0

        Parameters
        ----------
        njl: np.array
            Njl count obtained from self.count_njl, has shape (K, J, L)
        """
        self.alpha = self.alpha0 + njl

    def compute_lower_bound(self, q_t, e_ln_pi, ln_rho, model_outputs):
        """
        Function to compute lower bound (elbo) for debugging/logging purpose, not used to update the parameters

        Parameters
        ----------
        q_t: np.array
            aggregated class label probabilities (N x n_match, J)
        e_ln_pi: np.array
            The expected ln pi for each annotator, has shape (K, J, L)
        ln_rho: np.array
            unnormalised value of ln rho, that is used to compute lower bound, has shape (N x n_match, J)
        model_outputs: np.array
            matched (flatten) model class probabilities predictions with shape (N x n_match, J)

        Returns
        -------
        float
            the lower bound (elbo)
        """
        expected_ln_t = np.sum(np.log(np.sum(np.exp(ln_rho), axis=1)), axis=0) - np.sum(q_t * ln_rho)
        expected_ln_pi = sum([-self._ln_B(a0) + self._ln_B(a) +
                              np.sum((a0 - 1) * el) - np.sum((a - 1) * el)
                              for a0, a, el in zip(self.alpha0, self.alpha, e_ln_pi)])
        # ? normalize to max 0 so that exp^0 become 1
        model_outputs = model_outputs - model_outputs.max(axis=1, keepdims=True)
        expected_ln_model = np.sum(q_t * model_outputs) - np.sum(np.log(np.sum(np.exp(model_outputs), axis=1)), axis=0)
        return expected_ln_t + expected_ln_pi + expected_ln_model

    @staticmethod
    def _restore_labels(q_t, b_n, img_indexes, weights, max_len, conf_thres=0.):
        """
        Function to restore the flattened annotations back into its original format.
        All box coordinates are normalised to [0, 1].

        Parameters
        ----------
        q_t: np.array
            aggregated class label probabilities (N x n_match, J)
        b_n: np.array
            aggregated bounding box of shape (N x n_match, 4)
        img_indexes: np.array
            image indexes of the aggregated annotations with length N x n_match
        weights: np.array
            weights of the matched pairs with length N x n_match, used for reweighting the loss function
        max_len: int
            expected number of images (for when there are images with no annotations)
        conf_thres: float
            confidence threshold to remove low confidence aggregations (optional, default 0)

        Returns
        -------
        list[np.array]
            N list of aggregated annotations with shape (n_aggregated, 4 + J + 1), [xyxy, J class probabilities, weight]
        """
        # restore clean label based on estimated true class labels and box from BCC
        # and rearrange based on img_indexes (which should be sorted already)

        # xyxy from model and logits from bcc
        restored_labels = np.concatenate((b_n[:, :4], q_t, np.expand_dims(weights, 1)), axis=1)
        final = []
        idx_dict = BDCAggregator._unique_to_index(img_indexes)
        for img_idx in range(max_len):
            idx = idx_dict.get(img_idx, np.empty((0,), dtype=int))
            curr_labels = restored_labels[idx]
            # drop invalid box
            final.append(curr_labels[(curr_labels[:, 0] < curr_labels[:, 2]) & (curr_labels[:, 1] < curr_labels[:, 3]) & (curr_labels[:, -1] > conf_thres)])
        return final


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, '../../')
    os.chdir('../../')
    import yaml
    import torch
    from torch.utils.data import DataLoader
    from datasets.datasets import SyntheticDataset, collate_fn
    with open('./data/vincxr_ann17.yaml') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    with open('./config/bdc.yaml') as f:
        ca_hyp = yaml.load(f, Loader=yaml.SafeLoader)
    aggregator = BDCAggregator(data_dict['n_annotator'], data_dict['nc'], data_dict['nc_ann'], **ca_hyp['parameters'])
    dataset = SyntheticDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['train'],
                               image_size=(640, 640), train=True, augments=None, aggregator=aggregator,
                               clean_annotations_path=data_dict['clean_train'])

    dataset.noisy_annotations = dataset.normalize_bbox(dataset.noisy_annotations)
    bdc_annotations, lower_bound = aggregator.initialize_dataset(dataset)
    assert len(dataset.annotations) == len(bdc_annotations), 'this shouldnt happen'
    dataset.annotations = dataset.normalize_bbox(bdc_annotations, inverse=True)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True,
                            collate_fn=collate_fn, shuffle=False)

    noisy_outputs = aggregator.get_outputs(copy.deepcopy(dataset.noisy_annotations)[:5])
    noisy_outputs[0] = np.concatenate(noisy_outputs[0:2], axis=0)
    noisy_outputs[1] = np.zeros((0, 24))
    print(noisy_outputs)
    bdc_annotations, lower_bound = aggregator.fit_transform_crowd_labels(dataset.noisy_annotations[:5], noisy_outputs, update=False)
    print(bdc_annotations)

    # torch.save({'aggregator': aggregator}, 'test.pt')
    # imgs, targets, paths, shapes, clean_box = next(iter(dataloader))
    # print(targets)
    # print(clean_box)
    bdc_annotations, lower_bound = aggregator.initialize_dataset(dataset)
    bdc_annotations, lower_bound = aggregator.initialize_dataset(dataset)
    dataset.annotations = dataset.normalize_bbox(bdc_annotations, inverse=True)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True,
                            collate_fn=collate_fn, shuffle=False)

    imgs, targets, paths, shapes, clean_box = next(iter(dataloader))
    print(targets)
    print(clean_box)
