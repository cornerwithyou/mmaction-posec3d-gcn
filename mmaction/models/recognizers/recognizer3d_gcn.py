# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import OptSampleList
from .base import BaseRecognizer,BaseRecognizer_fusion
import time
import numpy as np
runtime_ntu60_test=[]
FPS_ntu60_test =[]

@MODELS.register_module()
class Recognizer3D_GCN(BaseRecognizer):
    """3D recognizer model framework."""

    def extract_feat(self,
                     inputs: Tensor,
                     stage: str = 'neck',
                     data_samples: OptSampleList = None,
                     test_mode: bool = False) -> tuple:
        """Extract features of different stages.

        Args:
            inputs_pose (torch.Tensor): The input data.
            stage (str): Which stage to output the feature.
                Defaults to ``'neck'``.
            data_samples (list[:obj:`ActionDataSample`], optional): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode (bool): Whether in test mode. Defaults to False.

        Returns:
                torch.Tensor: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline. These keys are usually included:
                    ``loss_aux``.
        """

        # Record the kwargs required by `loss` and `predict`
        inputs_pose, inputs_gcn = inputs
        loss_predict_kwargs = dict()
        #assert inputs_pose.shape[1] == inputs_gcn.shape[1]
        num_segs = inputs_pose.shape[1]
        # [N, num_crops, C, T, H, W] ->
        # [N * num_crops, C, T, H, W]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        inputs_pose = inputs_pose.view((-1,) + inputs_pose.shape[2:])

        bs, nc = inputs_gcn.shape[:2]
        inputs_gcn = inputs_gcn.reshape((bs * nc,) + inputs_gcn.shape[2:])

        # Check settings of test
        start_time = time.time()
        if test_mode:
            if self.test_cfg is not None:
                loss_predict_kwargs['fcn_test'] = self.test_cfg.get(
                    'fcn_test', False)
            if self.test_cfg is not None and self.test_cfg.get(
                    'max_testing_views', False):
                max_testing_views = self.test_cfg.get('max_testing_views')
                assert isinstance(max_testing_views, int)

                total_views = inputs_pose.shape[0]
                assert num_segs == total_views, (
                    'max_testing_views is only compatible '
                    'with batch_size == 1')
                view_ptr = 0
                feats = []
                while view_ptr < total_views:
                    batch_imgs = inputs_pose[view_ptr:view_ptr + max_testing_views]
                    feat = self.backbone(batch_imgs)
                    if self.with_neck:
                        feat, _ = self.neck(feat)
                    feats.append(feat)
                    view_ptr += max_testing_views

                def recursively_cat(feats):
                    # recursively traverse feats until it's a tensor,
                    # then concat
                    out_feats = []
                    for e_idx, elem in enumerate(feats[0]):
                        batch_elem = [feat[e_idx] for feat in feats]
                        if not isinstance(elem, torch.Tensor):
                            batch_elem = recursively_cat(batch_elem)
                        else:
                            batch_elem = torch.cat(batch_elem)
                        out_feats.append(batch_elem)

                    return tuple(out_feats)

                if isinstance(feats[0], tuple):
                    x_pose = recursively_cat(feats)
                else:
                    x_pose = torch.cat(feats)
            else:
                x_pose, x_gcn = self.backbone((inputs_pose,inputs_gcn))
                #x_gcn = self.backbone_gcn(inputs_gcn)
                if self.with_neck:
                    x_pose, _ = self.neck(x_pose)
                    #x_gcn = self.neck(x_gcn)


            endtimes = time.time()
            runtimes = endtimes - start_time
            FPS = inputs_pose.shape[0] * inputs_pose.shape[2] / runtimes
            runtime_ntu60_test.append(runtimes)
            FPS_ntu60_test.append(FPS)
            #print(f'内部runtimes = {runtimes}, FPS = {FPS},AVG_runtimes = {np.mean(np.array(runtime_ntu60_test[1:]))}, AVG_FPS={np.mean(np.array(FPS_ntu60_test[1:]))},')

            return tuple([x_pose, x_gcn ,inputs_gcn]), loss_predict_kwargs
        else:
            # Return features extracted through backbone
            x_pose,x_gcn = self.backbone(tuple([inputs_pose, inputs_gcn]))
            # x_gcn = self.backbone_gcn(inputs_gcn)
            if stage == 'backbone':
                return x_pose, loss_predict_kwargs

            loss_aux = dict()
            if self.with_neck:
                x_pose, loss_aux = self.neck(x_pose, data_samples=data_samples)

            # Return features extracted through neck
            loss_predict_kwargs['loss_aux'] = loss_aux
            if stage == 'neck':
                return tuple([x_pose, x_gcn ,inputs_gcn]), loss_predict_kwargs

            # Return raw logits through head.
            if self.with_cls_head and stage == 'head':
                x_pose = self.cls_head(tuple([x_pose, x_gcn ,inputs_gcn]), **loss_predict_kwargs)
                return x_pose, loss_predict_kwargs
