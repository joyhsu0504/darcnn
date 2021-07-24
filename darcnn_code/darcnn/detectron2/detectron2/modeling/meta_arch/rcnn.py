# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn
from functools import partial
from torch.nn import functional as F
from torch.autograd import Variable
import cv2
import random
import copy

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads, build_single_head_darcnn_roi_heads
from .build import META_ARCH_REGISTRY
import copy

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class DomainSeparationDARCNN(nn.Module):


    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.coco_backbone = copy.deepcopy(backbone)
        self.cryo_backbone = copy.deepcopy(backbone)
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.coco_roi_heads = copy.deepcopy(roi_heads)
        
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_single_head_darcnn_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
            
    def forward(self, batched_inputs):
        
        if not self.training:
            return self.inference(batched_inputs)
        
        storage = get_event_storage()
        if storage.iter == 0:
            self.coco_backbone = copy.deepcopy(self.backbone)
            self.cryo_backbone = copy.deepcopy(self.backbone)
            self.coco_roi_heads = copy.deepcopy(self.roi_heads)

        # Extract coco and cryo images
        batched_coco_inputs, batched_cryo_inputs = self.filter_domain(batched_inputs)
        
        coco_images = self.preprocess_image(batched_coco_inputs)
        cryo_images = self.preprocess_image(batched_cryo_inputs)
        
        if "instances" in batched_coco_inputs[0]:
            coco_gt_instances = [x["instances"].to(self.device) for x in batched_coco_inputs]
            
        else:
            coco_gt_instances = None
             
        # Pass through encoder backbone
        coco_shared_features = self.backbone(coco_images.tensor)
        cryo_shared_features = self.backbone(cryo_images.tensor)
        
        coco_private_features = self.coco_backbone(coco_images.tensor)
        cryo_private_features = self.cryo_backbone(cryo_images.tensor)
        
        # Domain adaptation losses
        last_key = 'p6'
        shared_sim_loss = self.da_sim_loss(coco_shared_features[last_key],
                                           cryo_shared_features[last_key])
        
        coco_diff_loss = self.da_diff_loss(coco_shared_features[last_key],
                                           coco_private_features[last_key])
        cryo_diff_loss = self.da_diff_loss(cryo_shared_features[last_key],
                                           cryo_private_features[last_key])


        # Proposal network
        coco_proposals, coco_proposal_losses = self.proposal_generator(coco_images,
                                                                       coco_shared_features,
                                                                       coco_gt_instances)
        self.proposal_generator.training = False
        cryo_proposals, _ = self.proposal_generator(cryo_images,
                                                    cryo_shared_features,
                                                    None)
        self.proposal_generator.training = True
        
        _, coco_detector_losses = self.coco_roi_heads(
                                              True,
                                              coco_images,
                                              coco_private_features,
                                              coco_shared_features,
                                              coco_proposals,
                                              coco_gt_instances)
        _, cryo_detector_losses = self.roi_heads(
                                              False,
                                              cryo_images,
                                              cryo_private_features,
                                              cryo_shared_features,
                                              cryo_proposals,
                                              _)
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        storage = get_event_storage()
        
        if storage.iter >= 10:
            alpha = 5
        else:
            alpha = (1 / (10-storage.iter)) * 5

        shared_sim_loss = {'sim_loss': shared_sim_loss['sim_loss'] * alpha}
                
        losses = {}
        
        losses.update(cryo_detector_losses)
        losses.update(coco_detector_losses)
        losses.update(coco_proposal_losses)
        losses.update(shared_sim_loss)
        losses.update(coco_diff_loss)
        losses.update(cryo_diff_loss)
        return losses
    
    def da_diff_loss(self, shared_features, specific_features, weight=1):
        # Orthogonality loss https://github.com/fungtion/DSN/blob/master/functions.py
        feat_size = 128
        shared_features = shared_features[:, :feat_size, :, :]
        specific_features = specific_features[:, :feat_size, :, :]
        reduced_shared_features = nn.Conv2d(feat_size, 1, kernel_size=1).to(torch.device("cuda:0"))(shared_features)
        reduced_specific_features = nn.Conv2d(feat_size, 1, kernel_size=1).to(torch.device("cuda:0"))(specific_features)
                
        batch_size = reduced_shared_features.size(0)
        input1 = reduced_shared_features.view(batch_size, -1)
        input2 = reduced_specific_features.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        diff_loss *= weight
        return {'diff_loss': diff_loss * 1}
    
    def da_sim_loss(self, coco_shared_features, cryo_shared_features, weight=1):
        # MMD loss https://github.com/CuthbertCai/pytorch_DAN/blob/master/utils.py
        min_x = min(coco_shared_features.size(2), cryo_shared_features.size(2))
        min_y = min(coco_shared_features.size(3), cryo_shared_features.size(3))
        crop_coco_shared_features = coco_shared_features[:, :, :min_x, :min_y]
        crop_cryo_shared_features = cryo_shared_features[:, :, :min_x, :min_y]
        
        reduced_coco_features = nn.Conv2d(256, 1, kernel_size=1).to(torch.device("cuda:0"))(crop_coco_shared_features)
        reduced_cryo_features = nn.Conv2d(256, 1, kernel_size=1).to(torch.device("cuda:0"))(crop_cryo_shared_features)
        
        reduced_coco_features = torch.squeeze(reduced_coco_features)
        reduced_cryo_features = torch.squeeze(reduced_cryo_features)
                        
        sigmas = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]
        
        def pairwise_distance(x, y):
            if not len(x.shape) == len(y.shape) == 2:
                raise ValueError('Both inputs should be matrices.')
            if x.shape[1] != y.shape[1]:
                raise ValueError('The number of features should be the same.')
            x = x.view(x.shape[0], x.shape[1], 1)
            y = torch.transpose(y, 0, 1)
            output = torch.sum((x - y) ** 2, 1)
            output = torch.transpose(output, 0, 1)
            return output
        
        def gaussian_kernel_matrix(x, y, sigmas):
            sigmas = sigmas.view(sigmas.shape[0], 1)
            beta = 1. / (2. * sigmas)
            dist = pairwise_distance(x, y).contiguous()
            dist_ = dist.view(1, -1)
            s = torch.matmul(beta, dist_)    
            return torch.sum(torch.exp(-s), 0).view_as(dist)
        
        def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
            cost = torch.mean(kernel(x, x))
            cost += torch.mean(kernel(y, y))
            cost -= 2 * torch.mean(kernel(x, y))
            return cost

        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas=Variable(torch.cuda.FloatTensor(sigmas))
        )
        
        batch = reduced_coco_features.size(0)
        mmd_losses = [maximum_mean_discrepancy(reduced_coco_features[i],
                                               reduced_cryo_features[0],
                                               kernel=gaussian_kernel)
                      for i in range(batch)]
        mmd_loss = sum(mmd_losses)
        mmd_loss *= weight
        return {'sim_loss': mmd_loss * 1}

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        shared_features = self.backbone(images.tensor)
        cryo_features = self.cryo_backbone(images.tensor)
        
        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, shared_features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                
            results, _ = self.roi_heads(False, images, cryo_features, shared_features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
        
    def filter_domain(self, batched_inputs):
        coco = []
        cryo = []
        for x in batched_inputs:
            if "domain" in x and x["domain"] == "cryo":
                cryo.append(x)
            else:
                coco.append(x)
        min_len = min(len(coco), len(cryo))
        coco = coco[:min_len]
        cryo = cryo[:min_len]

        return coco, cryo

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    


@META_ARCH_REGISTRY.register()
class PseudolabelTargetOnlyDARCNN(nn.Module):
    """
    Domain Adaptation R-CNN.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        coco_backbone: Backbone,
        cryo_backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        coco_roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.coco_backbone = coco_backbone
        self.cryo_backbone = cryo_backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.coco_roi_heads = coco_roi_heads
        
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        coco_backbone = copy.deepcopy(backbone)
        cryo_backbone = copy.deepcopy(backbone)
        
        return {
            "backbone": backbone,
            "coco_backbone": coco_backbone,
            "cryo_backbone": cryo_backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_single_head_darcnn_roi_heads(cfg, backbone.output_shape()),
            "coco_roi_heads": build_single_head_darcnn_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
            
    def forward(self, batched_inputs):
        
        if not self.training:
            return self.inference(batched_inputs)
        
        batched_coco_inputs = batched_inputs
        
        coco_images = self.preprocess_image(batched_coco_inputs)
        
        if "instances" in batched_coco_inputs[0]:
            coco_gt_instances = [x["instances"].to(self.device) for x in batched_coco_inputs]
            
        else:
            coco_gt_instances = None
            
        
        # Pass through encoder backbone
        coco_shared_features = self.backbone(coco_images.tensor)
        
        coco_private_features = self.cryo_backbone(coco_images.tensor)

        # Proposal network
        coco_proposals, coco_proposal_losses = self.proposal_generator(coco_images,
                                                                       coco_shared_features,
                                                                       coco_gt_instances)
        
        _, coco_detector_losses = self.roi_heads(
                                              True,
                                              coco_images,
                                              coco_private_features,
                                              coco_shared_features,
                                              coco_proposals,
                                              coco_gt_instances)
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

                
        losses = {}
        
        losses.update(coco_detector_losses)
        losses.update(coco_proposal_losses)

        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training
                
        images = self.preprocess_image(batched_inputs)
        shared_features = self.backbone(images.tensor)
        cryo_features = self.cryo_backbone(images.tensor)
        
        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, shared_features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(False, images, cryo_features, shared_features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
        
    def filter_domain(self, batched_inputs):
        coco = []
        cryo = []
        for x in batched_inputs:
            if "domain" in x and x["domain"] == "cryo":
                cryo.append(x)
            else:
                coco.append(x)
        min_len = min(len(coco), len(cryo))
        coco = coco[:min_len]
        cryo = cryo[:min_len]

        return coco, cryo

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    



