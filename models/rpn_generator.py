from typing import List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork, concat_box_prediction_layers
from torchvision.models.detection.faster_rcnn import _default_anchorgen, fasterrcnn_resnet50_fpn, \
    fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn, \
    FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, \
    FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes


class RPNGenerator(nn.Module):
    """
    RPN Model Class modified from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py
    but removing the box head

    """
    def __init__(
            self,
            backbone,
            # transform parameters
            min_size=800,
            max_size=1333,
            image_mean=None,
            image_std=None,
            # RPN parameters
            rpn_anchor_generator=None,
            rpn_head=None,
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            rpn_score_thresh=0.0,):
        super().__init__()

        self.backbone = backbone
        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            rpn_anchor_generator = _default_anchorgen()
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    @staticmethod
    def load_pretrained_rpn(model=None):
        if isinstance(model, str):
            if model == 'fasterrcnn_resnet50_fpn':  # 58.14
                pt_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,)
            elif model == 'fasterrcnn_resnet50_fpn_v2':  # 64.73
                pt_model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
            elif model == 'fasterrcnn_mobilenet_v3_large_fpn':  # 54.86
                pt_model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
            elif model == 'fasterrcnn_mobilenet_v3_large_320_fpn':  # 43.93
                pt_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
            else:
                raise ValueError('unknown model')
            model = RPNGenerator(pt_model.backbone)
            model.backbone = pt_model.backbone
            model.rpn = pt_model.rpn
            model.transform = pt_model.transform
            return model
        raise NotImplementedError('not implemented yet')

    def forward(self, images, targets=None):
        if self.training:
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

        images, targets = self.transform(images, targets)

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

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        # proposals, proposal_losses = self.rpn(images, features, targets)
        features = list(features.values())
        objectness, pred_bbox_deltas = self.rpn.head(features)
        anchors = self.rpn.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if targets is None:
            if self.rpn.training:
                raise ValueError("targets should not be None")
        else:
            labels, matched_gt_boxes = self.rpn.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.rpn.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.rpn.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }

        results = []
        for box, im_s, o_im_s in zip(boxes, images.image_sizes, original_image_sizes):
            results.append(resize_boxes(box, im_s, o_im_s))
        return results, scores, losses


if __name__ == '__main__':
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.resnet import ResNet50_Weights
    from utils.general import init_seeds
    rpn = RPNGenerator(backbone=resnet_fpn_backbone('resnet50', ResNet50_Weights.DEFAULT))
    rpn = RPNGenerator.load_pretrained_rpn('fasterrcnn_resnet50_fpn')

    x = [torch.rand((3, 500, 600))]
    with torch.no_grad():
        init_seeds(1)
        results, scores, losses = rpn(x, [{'boxes': torch.tensor([[1, 50, 50, 100], [400, 300, 430, 330]]),
                                           'labels': torch.tensor([1, 2]), 'weights': torch.tensor([1., 1.5]), }])

        print(results[0].shape)
        print(scores[0].shape)
        print(losses)

    print(torch.stack((losses['loss_objectness'], torch.tensor(0.5))))
