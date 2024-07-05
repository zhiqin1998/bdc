import torch


def transform_to_faster_rcnn_targets(targets, logits=False):
    """
    Helper function to transform the targets output of dataloader of dataset classes in datasets.datasets.py
    to the expected format of Faster RCNN

    Parameters
    ----------
    targets: list[np.array]
        list of arrays that contains [xyxy, class_id/class logits, (weights)]
    logits: bool
        Flag to indicate whether target contains class id or logits (optional, default False)

    Returns
    -------
    list[dict]
        list of dictionary with keys 'boxes', 'labels', and optionally 'weights'
    """
    # input targets is list of array: n_img x n_boxes x [x1, y1, x2, y2, cls_id/logits, box_score (for earl only), weight (for bdc)] cls_id + 1 to add background
    # output targets for fasterrcnn should be list of dict: [{'boxes': tensor([x1, y1, x2, y2]), 'labels': tensor([1])}]
    final_targets = []

    for target in targets:
        boxes = torch.from_numpy(target[:, :4]).float()
        if logits:
            labels = torch.zeros((target.shape[0], target.shape[1] - 5 + 1), dtype=torch.float32)  # minus 4 box coor and 1 weight, add 1 bg cls
            labels[:, 1:] = torch.from_numpy(target[:, 4:-1])
        else:
            labels = torch.from_numpy(target[:, 4]).long() + 1  # class id
        final_targets.append({'boxes': boxes, 'labels': labels})
        if logits: # bdc
            final_targets[-1]['weights'] = torch.from_numpy(target[:, -1])
        if not logits and target.shape[1] == 6:
            final_targets[-1]['weights'] = torch.from_numpy(target[:, 5])
    return final_targets


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))
