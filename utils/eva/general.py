import os
import torch

from detectron2.structures import Instances, Boxes


# changed detectron2.modeling.box_regression._dense_box_regression_loss \
# changed detectron2.modeling.roi_heads.cascade_rcnn.CascadeROIHeads \
# changed detectron2.modeling.roi_heads.roi_heads.ROIHeads \
# changed detectron2.modeling.FastRCNNOutputLayers to process weights and logits
def transform_to_eva_targets(imgs, targets, paths, shapes, logits=False):
    """
    Helper function to transform the output of dataloader of dataset classes in datasets.datasets.py
    to the expected format of EVA

    Parameters
    ----------
    imgs: torch.Tensor
        pytorch tensor of images
    targets: list[np.array]
        list of arrays that contains [xyxy, class_id/class logits, (weights)]
    paths: list[str]
        list of image file paths
    shapes: list[tuples]
        list of tuples with format (original height, original width), ((y gain, x gain), (y pad, x pad))
    logits: bool
        Flag to indicate whether target contains class id or logits (optional, default False)

    Returns
    -------
    list[dict]
        list of dictionary with keys 'file_name', 'height', 'width', 'image_id', 'image' and 'instances'
    """
    # input targets is list of array: n_img x n_boxes x [x1, y1, x2, y2, cls_id/logits, box_score (for earl only), weight (for bdc)]
    # eva expect logits of n+1 classes (last class is bg)
    datas = []

    for img, target, path, shape in zip(imgs, targets, paths, shapes):
        boxes = torch.from_numpy(target[:, :4]).float()
        temp = {'gt_boxes': Boxes(boxes)}
        if logits:
            gt_logits = torch.zeros((target.shape[0], target.shape[1] - 5 + 1),
                                    dtype=torch.float32)  # minus 4 box coor and 1 weight, add 1 bg cls
            gt_logits[:, :-1] = torch.from_numpy(target[:, 4:-1])

            labels = gt_logits.max(dim=1)[1].long()
            temp['gt_logits'] = gt_logits
        else:
            labels = torch.from_numpy(target[:, 4]).long()  # class id
        temp['gt_classes'] = labels
        if logits:  # bdc
            temp['weights'] = torch.from_numpy(target[:, -1])
        if not logits and target.shape[1] == 6:
            temp['weights'] = torch.from_numpy(target[:, 5])
        oh, ow = shape[0]
        img = (img * 255).to(torch.uint8)
        instances = Instances((oh, ow), **temp)
        data = {'file_name': path, 'height': oh, 'width': ow, 'image_id': os.path.basename(path)[:-4],
                'image': img, 'instances': instances}
        datas.append(data)
    return datas


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))
