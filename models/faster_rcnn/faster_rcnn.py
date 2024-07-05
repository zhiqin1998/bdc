import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any

from torch import Tensor


import torch
import torchvision
import torch.nn.functional as F

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.ops import boxes as box_ops


def create_model(num_classes, pretrained=True):
    # Load Faster RCNN pre-trained model
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
    #     weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
    #     weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
    # )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
    )
    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if not pretrained:
        # quick fix to avoid nan loss https://github.com/pytorch/vision/issues/4147#issuecomment-900539548
        model.rpn.min_size = 0.0

    return model


# functions below are implemented because FasterRCNN does not return both loss and detection in one forward pass
# they are modified from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py
def _old_get_loss_and_detection(model, imgs, targets):
    # naive way to get loss and detections is to do two forward passes (slow)
    is_training = model.training
    with torch.no_grad():
        model.train()
        loss = model(imgs, targets)
        model.eval()
        out = model(imgs, targets)

    if is_training:
        model.train()
    return loss, out


def get_loss_and_detection(model, images, targets):
    # override forward pass of faster rcnn to get both loss and detection
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models). labels - 1 to remove bg
    """
    if targets is None:
        torch._assert(False, "targets should not be none when in training mode")
    else:
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                torch._assert(
                    len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                    f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                )
            else:
                torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    # override forward for rpn and roi
    _rpn_training = model.rpn.training
    model.rpn.training = True
    proposals, proposal_losses = model.rpn(images, features, targets)
    model.rpn.training = _rpn_training

    # ------------------- override roi_heads forward -----------------------
    # detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    if targets is not None:
        for t in targets:
            # TODO: https://github.com/pytorch/pytorch/issues/26731
            floating_point_types = (torch.float, torch.double, torch.half)
            if not t["boxes"].dtype in floating_point_types:
                raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
            if not t["labels"].dtype == torch.int64:
                raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
            if model.roi_heads.has_keypoint():
                if not t["keypoints"].dtype == torch.float32:
                    raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

    training_proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)

    training_box_features = model.roi_heads.box_roi_pool(features, training_proposals, image_shapes)
    training_box_features = model.roi_heads.box_head(training_box_features)
    training_class_logits, training_box_regression = model.roi_heads.box_predictor(training_box_features)

    detections: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    if labels is None:
        raise ValueError("labels cannot be None")
    if regression_targets is None:
        raise ValueError("regression_targets cannot be None")
    loss_classifier, loss_box_reg = fastrcnn_loss(training_class_logits, training_box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = model.roi_heads.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)
    class_logits_list = class_logits.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    all_class_logits = []
    for boxes, scores, class_logit, image_shape in zip(pred_boxes_list, pred_scores_list, class_logits_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]
        class_logit = class_logit[:, 1:].unsqueeze(1).repeat(1, num_classes - 1, 1)  # repeat logit for each instance

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        class_logit = class_logit.reshape(-1, num_classes - 1)

        # remove low scoring boxes
        inds = torch.where(scores > model.roi_heads.score_thresh)[0]
        boxes, scores, labels, class_logit = boxes[inds], scores[inds], labels[inds], class_logit[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels, class_logit = boxes[keep], scores[keep], labels[keep], class_logit[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, model.roi_heads.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[: model.roi_heads.detections_per_img]
        boxes, scores, labels, class_logit = boxes[keep], scores[keep], labels[keep], class_logit[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_class_logits.append(class_logit)

    num_images = len(all_boxes)
    for i in range(num_images):
        detections.append(
            {
                "boxes": all_boxes[i],
                "labels": all_labels[i] - 1,  # remove bg class
                "scores": all_scores[i],
                "logits": all_class_logits[i]
            }
        )
    # ----------------end override-------------------

    detections = model.transform.postprocess(detections, image_shapes,
                                             original_image_sizes)  # type: ignore[operator]

    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections


def get_detection_with_logits(model, images):
    # this function is built for bcc, override forward to get detection with logits
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models). labels - 1 to remove bg
    """
    targets = None

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    # override forward for rpn and roi
    proposals, proposal_losses = model.rpn(images, features, targets)

    # ------------------- override roi_heads forward -----------------------
    # detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes

    detections: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}

    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    # boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)

    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = model.roi_heads.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)
    class_logits_list = class_logits.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    all_class_logits = []
    for boxes, scores, class_logit, image_shape in zip(pred_boxes_list, pred_scores_list, class_logits_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]
        class_logit = class_logit[:, 1:].unsqueeze(1).repeat(1, num_classes - 1, 1)  # repeat logit for each instance

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        class_logit = class_logit.reshape(-1, num_classes - 1)

        # remove low scoring boxes
        inds = torch.where(scores > 0.1)[0]
        boxes, scores, labels, class_logit = boxes[inds], scores[inds], labels[inds], class_logit[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels, class_logit = boxes[keep], scores[keep], labels[keep], class_logit[keep]

        # non-maximum suppression, only for vb we perform class agnostic nms
        keep = box_ops.batched_nms(boxes, scores, torch.zeros_like(labels), model.roi_heads.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[: model.roi_heads.detections_per_img // 2]
        boxes, scores, labels, class_logit = boxes[keep], scores[keep], labels[keep], class_logit[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_class_logits.append(class_logit)

    num_images = len(all_boxes)
    for i in range(num_images):
        detections.append(
            {
                "boxes": all_boxes[i],
                "labels": all_labels[i] - 1,  # remove bg class
                "scores": all_scores[i],
                "logits": all_class_logits[i]
            }
        )
    # ----------------end override-------------------

    detections = model.transform.postprocess(detections, image_shapes,
                                             original_image_sizes)  # type: ignore[operator]

    return detections


def get_loss_with_logits(model, images, targets):
    # this function is built for bcc, override original forward to calculate loss with logits and weights
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Dict[Any, Any]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models). labels - 1 to remove bg
    """
    num_classes = model.nc + 1
    if targets is None:
        torch._assert(False, "targets should not be none when in training mode")
    else:
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                torch._assert(
                    len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                    f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                )
            else:
                torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    # override forward for rpn and roi
    proposals, proposal_losses = model.rpn(images, features, targets)

    # ------------------- override roi_heads forward -----------------------
    # detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    if targets is not None:
        for t in targets:
            if 'weights' not in t:
                t['weights'] = torch.ones(t['labels'].shape[0], dtype=torch.float32, device=t['labels'].device)
            # TODO: https://github.com/pytorch/pytorch/issues/26731
            floating_point_types = (torch.float, torch.double, torch.half)
            if not t["boxes"].dtype in floating_point_types:
                raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
            if not t["weights"].dtype in floating_point_types:
                raise TypeError(f"target weights must of float type, instead got {t['weights'].dtype}")
            if not t["labels"].dtype in floating_point_types:
                raise TypeError(f"target labels/logits must of float type, instead got {t['labels'].dtype}")
            if model.roi_heads.has_keypoint():
                if not t["keypoints"].dtype == torch.float32:
                    raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

    # proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    model.roi_heads.check_targets(targets)
    if targets is None:
        raise ValueError("targets should not be None")
    dtype = proposals[0].dtype
    device = proposals[0].device

    gt_boxes = [t["boxes"].to(dtype) for t in targets]
    gt_labels = [t["labels"] for t in targets]
    gt_weights = [t["weights"] for t in targets]

    # append ground-truth bboxes to propos
    proposals = model.roi_heads.add_gt_proposals(proposals, gt_boxes)

    # get matching gt indices for each proposal
    # matched_idxs, labels = model.roi_heads.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
    matched_idxs = []
    labels = []
    actual_labels = []
    weights = []
    for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_weights_in_image in zip(proposals, gt_boxes, gt_labels, gt_weights):

        if gt_boxes_in_image.numel() == 0:
            # Background image
            device = proposals_in_image.device
            clamped_matched_idxs_in_image = torch.zeros(
                (proposals_in_image.shape[0],), dtype=torch.int64, device=device
            )
            labels_in_image = torch.zeros((proposals_in_image.shape[0], num_classes), dtype=torch.float32, device=device)
            labels_in_image[:, 0] = 1.  # bg class
            actual_label = labels_in_image.max(dim=1)[1]
            weights_in_image = torch.ones((proposals_in_image.shape[0],), dtype=torch.float32, device=device)
        else:
            #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
            match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = model.roi_heads.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.float32)
            weights_in_image = gt_weights_in_image[clamped_matched_idxs_in_image]

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == model.roi_heads.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds, 0] = 1.
            labels_in_image[bg_inds, 1:] = 0.

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == model.roi_heads.proposal_matcher.BETWEEN_THRESHOLDS
            actual_label = labels_in_image.max(dim=1)[1]
            actual_label[ignore_inds] = -1
            # labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

        matched_idxs.append(clamped_matched_idxs_in_image)
        labels.append(labels_in_image)
        actual_labels.append(actual_label)
        weights.append(weights_in_image)

    # sample a fixed proportion of positive-negative proposals
    sampled_inds = model.roi_heads.subsample(actual_labels)  # convert back to class index

    matched_gt_boxes = []
    num_images = len(proposals)
    for img_id in range(num_images):
        img_sampled_inds = sampled_inds[img_id]
        proposals[img_id] = proposals[img_id][img_sampled_inds]
        labels[img_id] = labels[img_id][img_sampled_inds]
        matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
        weights[img_id] = weights[img_id][img_sampled_inds]

        gt_boxes_in_image = gt_boxes[img_id]
        if gt_boxes_in_image.numel() == 0:
            gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
        matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

    regression_targets = model.roi_heads.box_coder.encode(matched_gt_boxes, proposals)

    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    detections: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    if labels is None:
        raise ValueError("labels cannot be None")
    if regression_targets is None:
        raise ValueError("regression_targets cannot be None")
    loss_classifier, loss_box_reg = fastrcnn_loss_with_weights(class_logits, box_regression, labels, regression_targets, weights, is_logits=True)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses


def fastrcnn_loss_with_weights(class_logits, box_regression, labels, regression_targets, weights=None, is_logits=False):
    # function to compute fastrcnn loss with logits and/or weights
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    if weights is not None:
        weights = torch.cat(weights, dim=0)
        classification_loss = torch.mean(F.cross_entropy(class_logits, labels, reduction='none') * weights)
    else:
        classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    if is_logits:
        # convert back to class indices
        labels = labels.max(dim=1)[1]
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    if weights is not None:
        box_loss = torch.sum(torch.sum(F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="none",
        ), dim=1) * weights[sampled_pos_inds_subset])
    else:
        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def get_loss_with_weights(model, images, targets):
    # override original forward to calculate loss with weights
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Dict[Any, Any]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models). labels - 1 to remove bg
    """
    if targets is None:
        torch._assert(False, "targets should not be none when in training mode")
    else:
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                torch._assert(
                    len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                    f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                )
            else:
                torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    # override forward for rpn and roi
    proposals, proposal_losses = model.rpn(images, features, targets)

    # ------------------- override roi_heads forward -----------------------
    # detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    if targets is not None:
        for t in targets:
            if 'weights' not in t:
                t['weights'] = torch.ones(t['labels'].shape, dtype=torch.float32, device=t['labels'].device)
            # TODO: https://github.com/pytorch/pytorch/issues/26731
            floating_point_types = (torch.float, torch.double, torch.half)
            if not t["boxes"].dtype in floating_point_types:
                raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
            if not t["weights"].dtype in floating_point_types:
                raise TypeError(f"target weights must of float type, instead got {t['weights'].dtype}")
            if not t["labels"].dtype == torch.int64:
                raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
            if model.roi_heads.has_keypoint():
                if not t["keypoints"].dtype == torch.float32:
                    raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

    # proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    model.roi_heads.check_targets(targets)
    if targets is None:
        raise ValueError("targets should not be None")
    dtype = proposals[0].dtype
    device = proposals[0].device

    gt_boxes = [t["boxes"].to(dtype) for t in targets]
    gt_labels = [t["labels"] for t in targets]
    gt_weights = [t["weights"] for t in targets]

    # append ground-truth bboxes to propos
    proposals = model.roi_heads.add_gt_proposals(proposals, gt_boxes)

    # get matching gt indices for each proposal
    # matched_idxs, labels = model.roi_heads.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
    matched_idxs = []
    labels = []
    weights = []
    for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_weights_in_image in zip(proposals, gt_boxes, gt_labels, gt_weights):

        if gt_boxes_in_image.numel() == 0:
            # Background image
            device = proposals_in_image.device
            clamped_matched_idxs_in_image = torch.zeros(
                (proposals_in_image.shape[0],), dtype=torch.int64, device=device
            )
            labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            weights_in_image = torch.ones((proposals_in_image.shape[0],), dtype=torch.float32, device=device)
        else:
            #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
            match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = model.roi_heads.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)
            weights_in_image = gt_weights_in_image[clamped_matched_idxs_in_image]

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == model.roi_heads.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == model.roi_heads.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

        matched_idxs.append(clamped_matched_idxs_in_image)
        labels.append(labels_in_image)
        weights.append(weights_in_image)

    # sample a fixed proportion of positive-negative proposals
    sampled_inds = model.roi_heads.subsample(labels)  # convert back to class index

    matched_gt_boxes = []
    num_images = len(proposals)
    for img_id in range(num_images):
        img_sampled_inds = sampled_inds[img_id]
        proposals[img_id] = proposals[img_id][img_sampled_inds]
        labels[img_id] = labels[img_id][img_sampled_inds]
        matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
        weights[img_id] = weights[img_id][img_sampled_inds]

        gt_boxes_in_image = gt_boxes[img_id]
        if gt_boxes_in_image.numel() == 0:
            gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
        matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

    regression_targets = model.roi_heads.box_coder.encode(matched_gt_boxes, proposals)

    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    detections: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    if labels is None:
        raise ValueError("labels cannot be None")
    if regression_targets is None:
        raise ValueError("regression_targets cannot be None")
    loss_classifier, loss_box_reg = fastrcnn_loss_with_weights(class_logits, box_regression, labels, regression_targets, weights)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses


if __name__ == '__main__':
    from utils.general import init_seeds
    nc = 2
    model = create_model(num_classes=nc + 1, pretrained=True)
    model.nc = nc
    # print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    x = [torch.rand((3, 500, 600))]
    with torch.no_grad():
        init_seeds(1)
        print(model(x, [{'boxes': torch.tensor([[1, 50, 50, 100], [400, 300, 430, 330]]),
                         'labels': torch.tensor([1, 2]), 'weights': torch.tensor([1., 1.5]),}]))
        init_seeds(1)
        print(get_loss_with_weights(model, x, [{'boxes': torch.tensor([[1, 50, 50, 100], [400, 300, 430, 330]]),
                                                'labels': torch.tensor([1, 2]), 'weights': torch.tensor([1., 1.5])}]))
        init_seeds(1)
        loss, out = get_loss_and_detection(model, x,
                                           [{'boxes': torch.tensor([[1, 50, 50, 100], [400, 300, 430, 330]]),
                                             'labels': torch.tensor([1, 2])}])
    print(loss)
    # print(out)
    assert len(out[0]['boxes']) == len(out[0]['labels']) == len(out[0]['scores']) == len(out[0]['logits'])
    assert out[0]['labels'].min().item() >= 0
    with torch.no_grad():
        init_seeds(1)
        loss = get_loss_with_logits(model, x,
                                         [{'boxes': torch.tensor([[1, 50, 50, 100], [400, 300, 430, 330]]),
                                          'labels': torch.tensor([[0., 1., 0.], [0., 0., 1.]]), 'weights': torch.tensor([1., 1.5])}])
    print(loss)
