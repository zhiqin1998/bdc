import copy
import os
import random

import torch
import albumentations as A
import numpy as np
import torchvision.transforms.functional as FT

from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
from utils.crowd.aggregator import bbox_iou
from sklearn.utils import compute_class_weight


def get_augmentation(hue=0.01, sat=0.7, val=0.4, translate=0.1, scale=0.1, rotate=10,
                     shear=0, perspective=0.0, fliplr=0.5, flipud=0.0):
    """
    Helper function to get the albumentations transformation for image augmentation

    Parameters
    ----------
    hue: float
        Hue value to jitter (optional, default 0.01)
    sat: float
        Saturation value to jitter (optional, default 0.7)
    val: float
        Value value to jitter (optional, default 0.4)
    translate: float
        Percentage of translation augmentation [-translate, translate] (optional, default 0.1)
    scale: float
        Percentage of scale augmentation [1-scale, 1+scale] (optional, default 0.1)
    rotate: float
        Degree of rotation augmentation [-rotate, rotate] (optional, default 10)
    shear: float
        Degree of shear augmentation [-shear, shear] (optional, default 0)
    perspective: float
        Percentage of four point perspective augmentation (optional, default 0)
    fliplr: float
        Probability of horizontal flip (optional, default 0.5)
    flipud: float
        Probability of vertical flip (optional, default 0)

    Returns
    -------
    albumentations.transforms
        albumentations transformation for image augmentation
    """
    transform = A.Compose([
        A.ColorJitter(brightness=val, saturation=sat, hue=hue),
        A.HorizontalFlip(p=fliplr),
        A.VerticalFlip(p=flipud),
        A.Affine(scale=(1-scale, 1+scale), translate_percent=(-translate, translate), rotate=(-rotate, rotate),
                 shear=(-shear, shear), keep_ratio=True),
        A.Perspective(scale=perspective)
    ], bbox_params=A.BboxParams(format='pascal_voc'), additional_targets={'clean_bboxes': 'bboxes'})
    return transform


def collate_fn(batch):
    """
    Collate function of torch dataloader when images are of the same size
    Returns torch tensor images but keep other elements as list
    """
    imgs, *others = zip(*batch)
    return torch.stack(imgs, 0), *others


class CleanDataset(Dataset):
    """
    Dataset Class for dataset with clean ground truth annotations

    Attributes
    ----------
    img_dir: str
        Path to image directory
    annotations_path: str
        Path to annotation directory
    image_size: int or list[int] or None
        Target image size, if length is longer than 2 then one random image size is chosen from it
        elif None then no resizing is performed
    annotation_files: list[str]
        list of annotations files in annotations_path
    annotations: list[np.array]
        list of loaded annotations of shape (n, 5)
    transform: torchvision.transforms
        torchvision transformation for image preprocessing
    normalize_box: bool
        Flag for whether to normalise bounding box to [0, 1]
    max_size: int
        maximum longer side for resizing when len(image_size) > 2
    normalize_mean: list[float]
        mean of RGB for normalizing
    normalize_std: list[float]
        std of RGB for normalizing
    """
    def __init__(self, img_dir, annotations_path, image_size=512, transform=None, normalize_box=True, expected_ele=5,
                 max_size=None, normalize_mean=None, normalize_std=None, skip_empty=False):
        """
        Class initialisation

        Parameters
        ----------
        img_dir: str
            Path to image directory
        annotations_path: str
            Path to annotation directory
        image_size: int or list[int] or None
            Target image size, if length is longer than 2 then one random image size is chosen from it
            elif None then no resizing is performed (optional, default 512)
        transform: torchvision.transforms
            torchvision transformation for image preprocessing (optional, default None)
        normalize_box: bool
            Flag for whether to normalise bounding box to [0, 1] (optional, default True)
        expected_ele: int
            The expected number of element in each annotation, used when there are images without annotations
            (optional, default 5)
        max_size: int
            maximum longer side for resizing when len(image_size) > 2 (optional, default None)
        normalize_mean: list[float]
            mean of RGB for normalizing (optional, default None)
        normalize_std: list[float]
            std of RGB for normalizing (optional, default None)
        skip_empty: bool
            Flag to skip images without annotations (optional, default False)
        """
        self.img_dir = img_dir
        self.annotations_path = annotations_path
        # if isinstance(image_size, int):
        #     image_size = (image_size, image_size)
        self.image_size = image_size
        self.annotation_files = os.listdir(self.annotations_path)
        if skip_empty:
            self.annotation_files = [f for f in self.annotation_files if len(open(os.path.join(annotations_path, f)).readlines())]
        self.annotations = self.read_annotations(self.annotation_files, annotations_path, expected_ele=expected_ele)
        assert len(self.annotations)
        self.transform = transform
        self.normalize_box = normalize_box
        self.max_size = max_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    @staticmethod
    def read_annotations(annotation_files, annotations_path, expected_ele=5):
        """
        Function to load all annotations

        Parameters
        ----------
        annotation_files: list[str]
            list of annotations files in annotations_path
        annotations_path: str
            Path to annotation directory
        expected_ele: int
            The expected number of element in each annotation, used when there are images without annotations
            (optional, default 5)

        Returns
        -------
        list[np.array]
            list of N ground truth annotations with shape (nl, expected_ele)
        """
        annotations = []
        for ann in annotation_files:
            ann_filename = os.path.join(annotations_path, ann)
            with open(ann_filename) as f:
                # x1, y1, x2, y2, cls_id, (ann_id)
                bbox = np.asarray([list(map(int, x.strip().split(','))) for x in f.readlines()], dtype=np.float32)
                if not len(bbox):
                    bbox = np.zeros((0, expected_ele), dtype=np.float32)
            annotations.append(bbox)
        return annotations

    def __len__(self):
        return len(self.annotations)

    def _normalize_img(self, img):
        """Function to normalize image with mean and std"""
        if self.normalize_mean and self.normalize_std:
            return FT.normalize(img, mean=self.normalize_mean, std=self.normalize_std)
        else:
            return img

    def _resize_to_tensor(self, img, bbox):
        """
        Function to resize and/or normalize images and bounding box with self.image_size and convert to torch tensor

        Parameters
        ----------
        img: np.array or PIL.Image
            input image
        bbox: np.array
            bounding box annotations with shape (nl, 5)

        Returns
        -------
        torch.Tensor
            resized image tensor
        np.array
            resized or normalized bounding box annotations
        """
        new_image = FT.to_tensor(img)
        new_bbox = np.asarray(bbox, dtype=np.float32)

        # resize image and box
        if self.image_size is None:  # no resize
            dims = (img.height, img.width)
        elif isinstance(self.image_size, (list, tuple)):  # resize to exact size
            if len(self.image_size) == 2:
                dims = self.image_size
                new_image = FT.resize(new_image, dims)
            else:
                size = random.choice(self.image_size)  # random resize
                dims = get_size_with_aspect_ratio((img.width, img.height), size, max_size=self.max_size)
                new_image = FT.resize(new_image, dims)
        elif isinstance(self.image_size, int):  # if int, then make sure image is smaller than it
            dims = get_size_with_aspect_ratio((img.height, img.width), self.image_size, self.image_size)
            new_image = FT.resize(new_image, dims)
        else:
            raise NotImplementedError

        # Resize bounding boxes
        old_dims = np.asarray([img.width, img.height, img.width, img.height], dtype=np.float32)[np.newaxis, :]
        new_bbox[:, :4] = new_bbox[:, :4] / old_dims  # percent coordinates

        if not self.normalize_box:
            new_bbox[:, :4] = new_bbox[:, :4] * np.asarray([dims[1], dims[0], dims[1], dims[0]],
                                                           dtype=np.float32)[np.newaxis, :]
        new_image = self._normalize_img(new_image)
        return new_image, new_bbox

    def __getitem__(self, idx):
        """
        Function to get single data with index

        Parameters
        ----------
        idx: int
            index

        Returns
        -------
        torch.Tensor
            pytorch image tensor
        np.array
            annotation array
        str
            path to the image file
        tuple
            tuple of image info with format (original height, original width), ((y gain, x gain), (y pad, x pad))
        """
        img_filename = os.path.join(self.img_dir, self.annotation_files[idx].rstrip('.txt'))
        img = Image.open(img_filename).convert("RGB")
        ow, oh = img.size

        bbox = copy.deepcopy(self.annotations[idx])  # avoid modifying inplace

        if self.transform:
            img, bbox = self.transform(img, bbox)
        img, bbox = self._resize_to_tensor(img, bbox)

        if self.image_size is not None:
            shapes = (oh, ow), ((img.shape[1] / oh, img.shape[2] / ow), (0., 0.))
        else:
            shapes = (oh, ow), ((1., 1.), (0., 0.))
        return img, bbox, img_filename, shapes


class SyntheticDataset(CleanDataset):
    """
    Dataset Class for dataset with noisy crowdsourced annotations

    Attributes
    ----------
    img_dir: str
        Path to image directory
    annotations_path: str
        Path to annotation directory
    image_size: int or list[int] or None
        Target image size, if length is longer than 2 then one random image size is chosen from it
        elif None then no resizing is performed
    annotation_files: list[str]
        list of annotations files in annotations_path
    annotations: list[np.array]
        list of aggregated annotations of shape (n, 5) or (n, n_class) for logits
    transform: torchvision.transforms
        torchvision transformation for image preprocessing
    normalize_box: bool
        Flag for whether to normalise bounding box to [0, 1]
    train: bool
        Flag to indicate whether training or testing
    augments: albumentations.transforms
        albumentations transformation for image augmentation
    aggregator: utils.crowd.aggregator.Aggregator
        Aggregator to use to aggregate crowdsourced annotations
    noisy_annotations: list[np.array]
        list of loaded crowdsourced annotations of shape (n, 6)
    clean_annotations: list[np.array]
        list of clean ground truth annotations of shape (n, 5) (if available to calculate AP)
    n_annotator: int
        number of annotator, K
    all_dims: list[tuple]
        list of [width, height] of all images for normalising bounding box purpose
    max_size: int
        maximum longer side for resizing when len(image_size) > 2
    normalize_mean: list[float]
        mean of RGB for normalizing
    normalize_std: list[float]
        std of RGB for normalizing

    TODO: rename class to something else
    """
    def __init__(self, img_dir, annotations_path, image_size=512, transform=None, normalize_box=True,
                 train=True, augments=None, aggregator=None, clean_annotations_path=None,
                 max_size=None, normalize_mean=None, normalize_std=None, expected_ele=6, skip_empty=False):
        """
        Class initialisation

        Parameters
        ----------
        img_dir: str
            Path to image directory
        annotations_path: str
            Path to annotation directory
        image_size: int or list[int] or None
            Target image size, if length is longer than 2 then one random image size is chosen from it
            elif None then no resizing is performed (optional, default 512)
        transform: torchvision.transforms
            torchvision transformation for image preprocessing (optional, default None)
        normalize_box: bool
            Flag for whether to normalise bounding box to [0, 1] (optional, default True)
        train: bool
            Flag to indicate whether training or testing (optional, default True)
        augments: albumentations.transforms
            albumentations transformation for image augmentation (optional, default None)
        aggregator: utils.crowd.aggregator.Aggregator
            Aggregator to use to aggregate crowdsourced annotations (optional, default None)
        clean_annotations_path: str
            Path to clean ground truth annotations directory if available for AP calculation (optional, default None)
        max_size: int
            maximum longer side for resizing when len(image_size) > 2 (optional, default None)
        normalize_mean: list[float]
            mean of RGB for normalizing (optional, default None)
        normalize_std: list[float]
            std of RGB for normalizing (optional, default None)
        expected_ele: int
            The expected number of element in each annotation, used when there are images without annotations
            (optional, default 6)
        skip_empty: bool
            Flag to skip images without annotations (optional, default False)
        """
        self.train = train
        self.augments = augments
        self.aggregator = aggregator
        super().__init__(img_dir, annotations_path, image_size, transform, normalize_box, expected_ele=expected_ele,
                         max_size=max_size, normalize_mean=normalize_mean, normalize_std=normalize_std,
                         skip_empty=skip_empty)
        self.noisy_annotations = copy.deepcopy(self.annotations)
        self.all_dims = None
        self.n_annotator = self._get_n_annotators()
        self._pretreat_crowd_labels()
        if clean_annotations_path is not None:
            self.clean_annotations = self.read_annotations(self.annotation_files, clean_annotations_path, expected_ele=5)
        else:
            self.clean_annotations = None

    def _augment(self, image, bbox, clean_bbox=None):
        """
        Function to augment image and bounding boxes

        Parameters
        ----------
        image: PIL.Image
            input image
        bbox: np.array
            aggregated annotations
        clean_bbox: np.array
            clean ground truth annotations (optional, default None)

        Returns
        -------
        PIL.Image
            augmented image
        np.array
            augmented annotations
        np.array
            augmented ground truth annotations
        """
        if self.augments is not None:
            if clean_bbox is None:
                clean_bbox = []
            ele_len = bbox.shape[-1]
            transformed = self.augments(image=np.array(image), bboxes=bbox, clean_bboxes=clean_bbox)
            image, bbox = Image.fromarray(transformed['image']), np.asarray(transformed['bboxes'], dtype=np.float32)
            clean_bbox = np.asarray(transformed['clean_bboxes'], dtype=np.float32)
            if not len(bbox):
                bbox = bbox.reshape(0, ele_len)
            if not len(clean_bbox):
                clean_bbox = clean_bbox.reshape(0, 5)
        return image, bbox, clean_bbox
    
    def _get_n_annotators(self):
        """Helper function to get number of annotators"""
        max_ann_id = -1
        for bbox in self.noisy_annotations:
            annotation_ids = bbox[:, -1]  # x1, y1, x2, y2, cls_id, ann_id
            if len(annotation_ids):
                max_ann_id = max(max_ann_id, max(annotation_ids))
        assert max_ann_id >= 0
        return max_ann_id + 1

    def _resize_to_tensor(self, img, bbox, clean_bbox):
        """
        Function to resize and/or normalize images and bounding box with self.image_size and convert to torch tensor
        Override super class method to also process clean ground truth bounding boxes
        """
        new_image = FT.to_tensor(img)
        new_bbox = np.asarray(bbox, dtype=np.float32)
        new_clean_bbox = np.asarray(clean_bbox, dtype=np.float32)

        # resize image and box
        if self.image_size is None:
            dims = (img.height, img.width)
        elif isinstance(self.image_size, (list, tuple)):
            if len(self.image_size) == 2:
                dims = self.image_size
                new_image = FT.resize(new_image, dims)
            else:
                size = random.choice(self.image_size)  # random resize
                dims = get_size_with_aspect_ratio((img.width, img.height), size, max_size=self.max_size)
                new_image = FT.resize(new_image, dims)
        elif isinstance(self.image_size, int):  # if int, then make sure longest edge is it
            dims = get_size_with_aspect_ratio((img.height, img.width), self.image_size, self.image_size)
            new_image = FT.resize(new_image, dims)
        else:
            raise NotImplementedError

        # Resize bounding boxes
        old_dims = np.asarray([img.width, img.height, img.width, img.height], dtype=np.float32)[np.newaxis, :]
        new_bbox[:, :4] = new_bbox[:, :4] / old_dims  # percent coordinates
        new_clean_bbox[:, :4] = new_clean_bbox[:, :4] / old_dims

        if not self.normalize_box:
            new_bbox[:, :4] = new_bbox[:, :4] * np.asarray([dims[1], dims[0], dims[1], dims[0]],
                                                           dtype=np.float32)[np.newaxis, :]
            new_clean_bbox[:, :4] = new_clean_bbox[:, :4] * np.asarray([dims[1], dims[0], dims[1], dims[0]],
                                                                       dtype=np.float32)[np.newaxis, :]
        new_image = self._normalize_img(new_image)
        return new_image, new_bbox, new_clean_bbox

    def __getitem__(self, idx):
        """
        Function to get single data with index

        Parameters
        ----------
        idx: int
            index

        Returns
        -------
        torch.Tensor
            pytorch image tensor
        np.array
            annotation array
        str
            path to the image file
        tuple
            tuple of image info with format (original height, original width), ((y gain, x gain), (y pad, x pad))
        np.array
            clean ground truth annotation array
        """
        img_filename = os.path.join(self.img_dir, self.annotation_files[idx].rstrip('.txt'))
        img = Image.open(img_filename).convert("RGB")
        ow, oh = img.size

        bbox, clean_bbox = copy.deepcopy(self.annotations[idx]), np.zeros((0, 5))  # avoid modifying inplace
        if self.clean_annotations:
            clean_bbox = copy.deepcopy(self.clean_annotations[idx])

        if self.train:
            img, bbox, clean_bbox = self._augment(img, bbox, clean_bbox)

        if self.transform:
            img, bbox, clean_bbox = self.transform(img, bbox, clean_bbox)
        img, bbox, clean_bbox = self._resize_to_tensor(img, bbox, clean_bbox)

        if self.image_size is not None:
            shapes = (oh, ow), ((img.shape[1] / oh, img.shape[2] / ow), (0., 0.))
        else:
            shapes = (oh, ow), ((1., 1.), (0., 0.))
        return img, bbox, img_filename, shapes, clean_bbox

    def _pretreat_crowd_labels(self):
        """
        Function to preprocess crowdsourced labels with the aggregator
        """
        if self.aggregator is not None:
            self.annotations = self.aggregator.pretreat_crowd_labels(self.noisy_annotations)

    def _populate_all_dims(self):
        """
        Function to populate all_dims attribute with image width and height
        """
        self.all_dims = []
        for idx in range(len(self)):
            img_filename = os.path.join(self.img_dir, self.annotation_files[idx].rstrip('.txt'))
            img = Image.open(img_filename).convert("RGB")
            self.all_dims.append([img.width, img.height])

    def normalize_bbox(self, annotations, inverse=False):
        """
        Helper function to normalize or unnormalize bounding box coordinates

        Parameters
        ----------
        annotations: list[np.array]
            list of annotations with shape [nl, >4]
        inverse: bool
            Flag to indicate whether to inverse transform/unnormalize the bounding box

        Returns
        -------
        list[np.array]
            list of annotations with shape [nl, >4] after normalize or unnormalize
        """
        annotations = copy.deepcopy(annotations)
        if self.all_dims is None:
            self._populate_all_dims()
        for idx in range(len(self)):
            width, height = self.all_dims[idx]
            dims = np.asarray([width, height, width, height], dtype=np.float32)[np.newaxis, :]
            if inverse:
                annotations[idx][:, :4] = annotations[idx][:, :4] * dims
            else:
                annotations[idx][:, :4] = annotations[idx][:, :4] / dims
        return annotations


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    """Helper function to get target image size while maintaining the aspect ratio"""
    h, w = image_size
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


class ClassificationDataset(Dataset):
    """
    Dataset Class for dataset with clean ground truth annotations to train Classification network

    Attributes
    ----------
    img_dir: str
        Path to image directory
    annotations_path: str
        Path to annotation directory
    img_size: list[int]
        Target image size
    annotation_files: list[str]
        list of annotations files in annotations_path
    annotations: list[np.array]
        list of loaded annotations of shape (n, 5)
    normalize_mean: list[float]
        mean of RGB for normalizing
    normalize_std: list[float]
        std of RGB for normalizing
    train: bool
        Flag to indicate training or testing
    augment: bool
        Flag for whether to perform image augmentation

    """
    def __init__(self, img_dir, annotations_path, img_size, normalize_mean, normalize_std, train=False, augment=True, min_size=50, fp_prob=.05):
        """
        Class initialisation

        Parameters
        ----------
        img_dir: str
            Path to image directory
        annotations_path: str
            Path to annotation directory
        img_size: list[int]
            Target image size
        normalize_mean: list[float]
            mean of RGB for normalizing
        normalize_std: list[float]
            std of RGB for normalizing
        train: bool
            Flag to indicate training or testing (optional, default False)
        augment: bool
            Flag for whether to perform image augmentation (optional, default True)
        min_size: int
            minimum cropped image size (optional, default 50)
        fp_prob: float
            false positive (background) probability (optional, default 0.05)
        """
        self.img_dir = img_dir
        self.annotations_path = annotations_path
        self.annotation_files = os.listdir(self.annotations_path)
        self.img_size = [img_size, img_size]
        self.annotations = self.read_annotations(self.annotation_files, annotations_path, min_size, fp_prob)
        assert len(self.annotations)
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.train = train
        if self.train:
            if augment:
                self.augment = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomApply([
                        transforms.ColorJitter(hue=0.1, saturation=0.4, brightness=0.4),
                        transforms.GaussianBlur((5, 5))
                        ])
                ])
            else:
                self.augment = transforms.Compose([])

    def read_annotations(self, annotation_files, annotations_path, min_size=50, fp_prob=1/80):
        """
        Function to load annotations for classification with random background classes

        Parameters
        ----------
        annotation_files: list[str]
            list of annotations files in annotations_path
        annotations_path: str
            Path to annotation directory
        min_size: int
            Minimum cropped image size (optional, default 50)
        fp_prob: float
            false positive (background) probability (optional, default 1/80)

        Returns
        -------
        list[tuple]
            List of tuple of (image_path, np.array of cropping xy coordinate and class id)
        """
        annotations = []
        for ann in annotation_files:
            ann_filename = os.path.join(annotations_path, ann)
            with open(ann_filename) as f:
                # x1, y1, x2, y2, cls_id, (ann_id)
                bbox = np.asarray([list(map(int, x.strip().split(','))) for x in f.readlines()], dtype=int)
                if not len(bbox):
                    bbox = np.zeros((0, 5))
                bbox[:, 4] += 1
                for box in bbox:
                    if box[2] - box[0] > min_size and box[3] - box[1] > min_size:
                        box[0] = max(box[0] - 3, 0)
                        box[1] = max(box[1] - 3, 0)
                        box[2] = box[2] + 3
                        box[3] = box[3] + 3
                        annotations.append((ann.rstrip('.txt'), box))
            if random.random() < fp_prob:
                img_filename = os.path.join(self.img_dir, ann.rstrip('.txt'))
                img = Image.open(img_filename).convert("RGB")
                w, h = img.size
                if w > self.img_size[0] and h > self.img_size[1]:
                    i = 0
                    while i < 5:
                        size = random.randrange(100, self.img_size[0])
                        random_box = [random.randrange(w - size), random.randrange(h - size)]
                        random_box = [random_box[0], random_box[1], random_box[0] + size, random_box[1] + size]
                        if len(bbox):
                            if bbox_iou(random_box, bbox).max() < 0.01:
                                box = random_box
                                annotations.append((ann.rstrip('.txt'), np.asarray(box + [0])))
                                break
                        i += 1
        return annotations

    def __len__(self):
        return len(self.annotations)

    def get_class_weights(self):
        """Function to compute class weights to re-weight imbalance loss"""
        all_classes = np.asarray([a[1][4] for a in self.annotations])
        class_weights = compute_class_weight('balanced', classes=np.unique(all_classes), y=all_classes)
        class_weights = np.clip(class_weights, 0.01, 5)
        return torch.tensor(class_weights, dtype=torch.float)

    def __getitem__(self, idx):
        """
        Function to get single data with index

        Parameters
        ----------
        idx: int
            index

        Returns
        -------
        torch.Tensor
            pytorch image tensor
        int
            class index
        """
        img_filename, box = self.annotations[idx]
        img_filename = os.path.join(self.img_dir, img_filename)
        img = Image.open(img_filename).convert("RGB")
        box, cls = box[:4], box[4]

        img = img.crop(box)
        transformed_img = FT.resize(img, self.img_size)
        if self.train:
            transformed_img = self.augment(transformed_img)
        transformed_img = FT.pil_to_tensor(transformed_img)
        transformed_img = FT.normalize(FT.convert_image_dtype(transformed_img, torch.float),
                                       self.normalize_mean, self.normalize_std)
        return transformed_img, cls


if __name__ == '__main__':
    import os
    os.chdir('../')
    import yaml
    from torch.utils.data import DataLoader
    from utils.crowd.majority_vote import MVAggregator
    from utils.crowd.aggregator import NoAggregator
    from utils.eva.general import collate_fn, transform_to_eva_targets

    with open('./data/vincxr_ann17.yaml') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    with open('./config/mv.yaml') as f:
        ca_hyp = yaml.load(f, Loader=yaml.SafeLoader)
    ca_hyp['parameters']['box_combine_method'] = 'average'
    augments = get_augmentation()
    # aggregator = MVAggregator(data_dict['n_annotator'], **ca_hyp['parameters'])
    aggregator = NoAggregator()
    dataset = SyntheticDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['train'],
                               # clean_annotations_path=data_dict['clean_train'],
                               image_size=512, train=True, augments=None, aggregator=aggregator)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, pin_memory=True,
                            collate_fn=collate_fn, shuffle=False)

    nb = len(dataloader)  # number of batches

    # for i in range(len(dataset)):
    #     img, bbox, img_filename, shapes, clean_bbox = dataset.__getitem__(i)
    #     if (bbox[:, 0] >= bbox[:, 2]).any() or (bbox[:, 1] >= bbox[:, 3]).any():
    #         print(img_filename, bbox)
    # print(augments(image=img.numpy().transpose(1, 2, 0), bboxes=box))

    print(nb)
    imgs, targets, paths, shapes, clean_box = next(iter(dataloader))
    print(imgs.shape)
    print(targets)
    # print(paths)
    # print(shapes)
    print(clean_box)

    test_dataset = CleanDataset(img_dir=data_dict['image_dir'], annotations_path=data_dict['test'],
                                image_size=512)
    test_dataloader = DataLoader(test_dataset, batch_size=2, num_workers=2, pin_memory=True,
                                 collate_fn=collate_fn, shuffle=False)

    imgs, targets, paths, shapes = next(iter(test_dataloader))
    print(imgs.shape)
    print(targets)
