import json
import os
import numpy as np
import pandas as pd
from PIL import Image
from utils.general import bbox_iou


def generate_random_conf_matrix(n_annotator, n_classes, acc_mean=0.6, acc_std=0.05, seed=1234, related_classes=None):
    """
    Function to generate confusion matrix by filling diagonal with samples from a Gaussian distribution
    This is used to synthesise class label with the typical approach

    Parameters
    ----------
    n_annotator: int
        number of annotator, K
    n_classes: int
        number of classes, J
    acc_mean: float
        mean of Gaussian for the diagonal (optional, default 0.6)
    acc_std: float
        std of Gaussian for the diagonal (optional, default 0.05)
    seed: int
        pseudorandom seed (optional, default 1234)
    related_classes: list[list]
        list of related classes for each classes, related classes with have higher synthesis chance compared to
        non-related classes (optional, default None)

    Returns
    -------
    np.array
        confusion matrix of shape (K, J, J)
    """
    # function to generate confusion matrix by filling diagonal with normal
    generator = np.random.default_rng(seed=seed)
    conf_matrix = np.zeros((n_annotator, n_classes, n_classes))
    if related_classes is None:
        related_classes = [[] for _ in range(n_classes)]
    for i in range(n_annotator):
        np.fill_diagonal(conf_matrix[i], np.clip(generator.normal(acc_mean, acc_std, n_classes), 0, 1))
        for j in range(n_classes):
            class_acc = conf_matrix[i, j, j]
            related_class = related_classes[j]
            if not len(related_class):
                other = generator.random(n_classes - 1)
                conf_matrix[i, j, np.arange(n_classes) != j] = other / other.sum() * (1 - class_acc)
            else:  # distribute 75% remaining (up to class_acc) around related classes first
                related_p = generator.random(len(related_class))
                conf_matrix[i, j, related_class] = related_p / related_p.sum() * min((1 - class_acc) * 0.75, class_acc)
                other = generator.random(n_classes - len(related_class) - 1)
                other_class = [l for l in np.arange(n_classes) if l != j and l not in related_class]
                conf_matrix[i, j, other_class] = other / other.sum() * (
                            1 - (conf_matrix[i, j, related_class].sum() + class_acc))
    return conf_matrix


def generate_dl_conf_matrix(n_annotator, dirichlet_conf, seed=1234):
    """
    Function to generate confusion matrix by drawing samples from a Dirichlet matrix obtained from classification
    deep learning model (our method)

    Parameters
    ----------
    n_annotator: int
        number of annotator, K
    dirichlet_conf: np.array
        Dirichlet matrix obtained from classification deep learning model with shape (J, J)
    seed: int
        pseudorandom seed (optional, default 1234)

    Returns
    -------
    np.array
        confusion matrix of shape (K, J, J)
    """
    # function to generate confusion matrix by using confusion matrix of a classification model as dirichlet
    generator = np.random.default_rng(seed=seed)
    assert dirichlet_conf.shape[0] == dirichlet_conf.shape[1]
    n_classes = dirichlet_conf.shape[0]
    conf_matrix = np.zeros((n_annotator, n_classes, n_classes))
    for i in range(n_classes):
        class_acc = generator.dirichlet(dirichlet_conf[i], n_annotator)  # K x J
        conf_matrix[:, i, :] = class_acc
    return conf_matrix


def generate_box_parameters(n_annotator, translation_err_mean=(0.05, 0.05), translation_err_std=(0.05, 0.05),
                            scale_err_mean=(0.05, 0.05), scale_err_std=(0.05, 0.05), seed=1234):
    """
    Function to generate xy translation scaling Gaussian parameters from given range uniformly
    This is used to synthesise bounding box with the typical approach

    Parameters
    ----------
    n_annotator: int
        number of annotator, K
    translation_err_mean: tuple[int]
        tuple of xy translation error mean (optional, default (0.05, 0.05))
    translation_err_std: tuple[int]
        tuple of xy translation error std (optional, default (0.05, 0.05))
    scale_err_mean: tuple[int]
        tuple of xy scaling error mean (optional, default (0.05, 0.05))
    scale_err_std: tuple[int]
        tuple of xy scaling error std (optional, default (0.05, 0.05))
    seed: int
        pseudorandom seed (optional, default 1234)

    Returns
    -------
    np.array
        xy translation scaling Gaussian parameters of shape (K, 4, 2)
    """
    # mean and std for trans and scale error in x and y axis respectively
    # mean is drawn uniformly from (-x, x) while std is drawn uniformly from (0, x)
    # class agnostic
    # return n_ann x 4 x 2, mean and standard of gaussian for translation, scale error for x and y separately
    generator = np.random.default_rng(seed=seed)
    box_params = np.zeros((n_annotator, 4, 2))
    # box_params[:, :, 1] = generator.random((n_annotator, 4)) / 20.
    for i in range(n_annotator):
        # box_params[i, :, 0] = generator.normal(translation_err_mean + scale_err_mean, translation_err_std + scale_err_std)
        box_params[i, :, 0] = generator.uniform(
            (-translation_err_mean[0], -translation_err_mean[1]) + (-scale_err_mean[0], -scale_err_mean[1]),
            translation_err_mean + scale_err_mean)
        box_params[i, :, 1] = generator.uniform((0., 0., 0., 0.,), translation_err_std + scale_err_std)

    return box_params


def xyxy2xywh(x):
    """Helper function to convert xyxy coordinate to x-center, y-center, width, height format"""
    y = np.asarray(x, dtype=np.float32).copy()
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    """Helper function to convert x-center, y-center, width, height format to xyxy coordinate"""
    y = np.asarray(x, dtype=np.float32).copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def correct_boxes(boxes):
    """Helper function to clip bounding boxes between [0, 1]"""
    boxes[:, 0] = np.maximum(boxes[:, 0], 0)
    boxes[:, 1] = np.maximum(boxes[:, 1], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], 1)
    boxes[:, 3] = np.minimum(boxes[:, 3], 1)
    return boxes


def generate_synthetic_data(dataset, n_annotator=None, n_classes=None, annotator_acc_mean=0.7, annotator_acc_std=0.05,
                            conf_matrix=None, pixel_offset=1, annotator_coverage=None, related_classes=None, fp_prob=1.,
                            fixed_box=False, rpn_proposals=None,  # dict of ann_id to proposals
                            translation_err_mean=(0.1, 0.1), translation_err_std=(0.1, 0.1),
                            scale_err_mean=(0.1, 0.1), scale_err_std=(0.1, 0.1), box_params=None, seed=1234):
    """
    Function to synthesise crowdsourced annotations

    Parameters
    ----------
    dataset: torch.dataset
        must have attributes 'classes', 'images', 'gt'
    n_annotator: int
        number of annotator, K (optional, default None to infer from conf_matrix
    n_classes: int
        number of classes (optional, default None to infer from dataset)
    annotator_acc_mean: float
        mean of Gaussian for the diagonal to generate confusion matrix (optional, default 0.7)
    annotator_acc_std: float
        std of Gaussian for the diagonal to generate confusion matrix (optional, default 0.05)
    conf_matrix: np.array
        class label confusion matrix of shape (K, J, J), overrides annotator_acc_mean and annotator_acc_std if given
        (optional, default None)
    pixel_offset: int
        pixel coordinate offset (optional, default 1 for VOC dataset)
    annotator_coverage: list[float]
        list of length K that indicate the coverage of each annotator (optional, default None where each annotator has
        full coverage)
    related_classes: list[list]
        list of related classes for each classes, related classes with have higher synthesis chance compared to
        non-related classes (optional, default None to infer from dataset)
    fp_prob: float
        probability of false positives (optional, default 0.5)
    fixed_box: bool
        Flag for whether to fix or synthesise bounding boxes (optional, default False)
        Function will ignore the 6 parameters below if True
    rpn_proposals: dict{int: list[np.array]}
        Dictionary mapping of annotator id to rpn_proposals (list of np.array of shape (n_proposals, 4))
        Function will ignore the 5 parameters below if rpn_proposals is given
    translation_err_mean: tuple[int]
        tuple of xy translation error mean to generate bounding box Gaussian parameters (optional, default (0.1, 0.1))
    translation_err_std: tuple[int]
        tuple of xy translation error std to generate bounding box Gaussian parameters (optional, default (0.1, 0.1))
    scale_err_mean: tuple[int]
        tuple of xy scaling error mean to generate bounding box Gaussian parameters (optional, default (0.1, 0.1))
    scale_err_std: tuple[int]
        tuple of xy scaling error std to generate bounding box Gaussian parameters (optional, default (0.1, 0.1))
    box_params: np.array
        xy translation scaling Gaussian parameters of shape (K, 4, 2), overrides translation_err_mean, translation_err_std,
        scale_err_mean and scale_err_std if given (optional, default None)
    seed: int
        pseudorandom seed (optional, default 1234)

    Returns
    -------
    list[np.array]
        list of synthesised crowdsourced annotations of shape (n, 6) [xyxy, class_id, annotator_id]
    conf_matrix: np.array
        class label confusion matrix of shape (K, J, J)
    box_params: np.array
        xy translation scaling Gaussian parameters of shape (K, 4, 2), None if rpn_proposals is given
    """
    if related_classes is None:
        related_classes = getattr(dataset, 'related_classes', None)
    if conf_matrix is None:
        assert n_annotator is not None, 'parameter n_annotator must be provided when confusion matrix is None'
        # assert annotator_acc_mean is not None, 'parameter annotator_acc_mean must be provided when confusion matrix is None'
        # assert annotator_acc_std is not None, 'parameter annotator_acc_std must be provided when confusion matrix is None'
        if n_classes is None:
            n_classes = len(dataset.classes)
        conf_matrix = generate_random_conf_matrix(n_annotator, n_classes, annotator_acc_mean, annotator_acc_std, seed=seed,
                                           related_classes=related_classes)
    else:
        n_annotator = conf_matrix.shape[0]
        assert conf_matrix.shape[1] == conf_matrix.shape[2], 'invalid confusion matrix shape'
        n_classes = conf_matrix.shape[1]

    if not fixed_box:
        if box_params is None:
            assert n_annotator is not None, 'parameter n_annotator must be provided when box_params is None'
            box_params = generate_box_parameters(n_annotator, translation_err_mean=translation_err_mean,
                                                 translation_err_std=translation_err_std,
                                                 scale_err_mean=scale_err_mean, scale_err_std=scale_err_std, seed=seed)
        else:
            assert n_annotator == box_params.shape[0]
            assert box_params.shape[1] == 4 and box_params.shape[2] == 2, 'invalid box params shape'
    if rpn_proposals is not None:
        assert isinstance(rpn_proposals, dict), 'invalid proposals provided, expected dict'
        for _, proposals in rpn_proposals.items():
            assert len(proposals) == len(dataset), 'invalid proposals provided, incorrect length'

    generator = np.random.default_rng(seed=seed)
    img_to_annotators = dict()
    if annotator_coverage is not None:
        assert len(annotator_coverage) == n_annotator
        n_data = len(dataset)
        all_indexes = generator.permutation(n_data)
        for ann_id, ann_p in enumerate(annotator_coverage):
            n_seen = int(n_data * ann_p)
            seen_idx, all_indexes = all_indexes[:n_seen], all_indexes[n_seen:]
            if len(all_indexes) < n_data // 2:
                all_indexes = np.concatenate([all_indexes, generator.permutation(n_data)])
            for idx in seen_idx:
                if idx in img_to_annotators:
                    img_to_annotators[idx].append(ann_id)
                else:
                    img_to_annotators[idx] = [ann_id]
        if len(img_to_annotators) < n_data:
            print('warning: some data has no annotator, assuming all annotator annotate it by default')

    # generate target label of shape [n_data, n_annotator, n_annotations (variable size)]
    generator = np.random.default_rng(seed=seed)
    final_labels = []
    for i, (img, gt) in enumerate(zip(dataset.images, dataset.gt)):
        crowd_annotations = [[] for _ in range(n_annotator)]
        dims = None  # only load dims if needed
        for annotator in img_to_annotators.get(i, range(n_annotator)):
            gt_labels = [box[4] for box in gt]
            # synthesize labels
            annotated_labels = np.sum(np.cumsum(conf_matrix[annotator, gt_labels, :],
                                                axis=1) < generator.random((len(gt_labels), 1)), axis=1)
            annotated_boxes = [[b - pixel_offset for b in box[:4]] for box in gt]
            if not fixed_box and len(annotated_boxes):
                if rpn_proposals is None:
                    # synthesize boxes
                    if dims is None:
                        width, height = Image.open(img).size
                        dims = np.asarray([width, height, width, height], dtype=np.float32)[np.newaxis, :]
                    annotated_boxes = np.asarray([box[:4] for box in annotated_boxes], dtype=np.float32) / dims
                    annotated_boxes = xyxy2xywh(annotated_boxes)
                    annotated_boxes[:, 0] += generator.normal(*box_params[annotator][0], size=len(gt)) * annotated_boxes[:,
                                                                                                         2]
                    annotated_boxes[:, 1] += generator.normal(*box_params[annotator][1], size=len(gt)) * annotated_boxes[:,
                                                                                                         3]
                    annotated_boxes[:, 2] *= np.clip(generator.normal(*box_params[annotator][2], size=len(gt)) + 1., .1, 2.)
                    annotated_boxes[:, 3] *= np.clip(generator.normal(*box_params[annotator][3], size=len(gt)) + 1., .1, 2.)
                    annotated_boxes = (np.clip(xywh2xyxy(annotated_boxes), 0., 1.) * dims).astype(int).tolist()
                else:
                    final_boxes = []
                    for box in annotated_boxes:
                        iou = np.clip(bbox_iou(box, rpn_proposals[annotator][i][:, :4], CIoU=True).numpy(), 0, 1)
                        prob = iou * rpn_proposals[annotator][i][:, 4]
                        prob = prob / (prob.sum() + 1e-8)  # adding to prevent div by 0
                        choices = rpn_proposals[annotator][i][:, :4]
                        prob = np.append(prob, 1.)
                        final_boxes.append(generator.choice(choices, p=prob / prob.sum(), replace=False, shuffle=False).astype(int).tolist())
                    annotated_boxes = final_boxes

            if generator.random() < fp_prob:
                random_bg_label = generator.random()  # false positive case
                random_label = np.sum(np.cumsum(conf_matrix[annotator, 0, :], ) < random_bg_label)
                if random_label != 0:
                    if rpn_proposals is None:
                        if dims is None:
                            width, height = Image.open(img).size
                        x1, y1 = generator.integers(0, width - 50), generator.integers(0, height - 50)
                        x2, y2 = generator.integers(x1 + 10, min(x1 + int(0.5 * width), width - 1)), \
                            generator.integers(y1 + 10, min(y1 + int(0.5 * height), height - 1))
                        random_box = [x1, y1, x2, y2]
                        annotated_labels = np.append(annotated_labels, random_label)
                    else:
                        prob = rpn_proposals[annotator][i][:, 4]
                        random_box = generator.choice(rpn_proposals[annotator][i], p=prob / prob.sum(), replace=False, shuffle=False).astype(int).tolist()
                    annotated_boxes.append(random_box)

            annotations = [annotated_box + [annotated_label] for annotated_box, annotated_label in
                           zip(annotated_boxes, annotated_labels)
                           if
                           annotated_label and annotated_box[2] > annotated_box[0] and \
                           annotated_box[3] > annotated_box[1]]
            # remove annotation if label = 0 (background) or zero width height box
            # crowd_annotations.append(annotations)
            crowd_annotations[annotator].extend(annotations)
        final_labels.append(crowd_annotations)
    return final_labels, conf_matrix, box_params


def crowd_labels2df(crowd_labels, dataset, remove_bg_cls=True):
    """
    Function to convert crowdsourced labels into dataframe

    Parameters
    ----------
    crowd_labels: list[list]
        list of crowdsourced annotations which is list of length K containing each annotator annotations (xyxy, class_id)
    dataset: torch.dataset
        must have attribute 'images' that contains the image file names
    remove_bg_cls: bool
        flag to remove background classes by subtracting 1 from the class id (optional, default True)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'img_path', 'x1', 'y1', 'x2', 'y2', 'class_id', 'annotator_id'
    """
    temp_data = []
    for crowd_label, img_file in zip(crowd_labels, dataset.images):
        img_file = os.path.basename(img_file)
        for ann_id, annotations in enumerate(crowd_label):
            for box in annotations:
                temp_data.append([img_file, *box, ann_id])

    df = pd.DataFrame(temp_data, columns=['img_path', 'x1', 'y1', 'x2', 'y2', 'class_id', 'annotator_id'])
    if remove_bg_cls:
        df['class_id'] = df['class_id'] - 1
    return df


def crowd_labels_df2txt(df, path='../data/voc2007'):
    """
    Function to convert dataframe into csv files for each image file

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with columns: 'img_path', 'x1', 'y1', 'x2', 'y2', 'class_id', 'annotator_id'
    path: str
        Directory path to save the csv files
    """
    os.makedirs(path, exist_ok=True)
    for filename, group in df.groupby('img_path'):
        filename += '.txt'
        group.drop(columns='img_path').to_csv(os.path.join(path, filename), index=False, header=False)
