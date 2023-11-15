# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList

from mmaction.registry import MODELS
from .focal_func import SpMiddleResNetFHDFocal

@MODELS.register_module()
class spfocal(BaseModule):
    def __init__(self,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.model = SpMiddleResNetFHDFocal(num_input_features=17,
                    ds_factor=8,
                    TOPK=True,
                    USE_IMG=True,
                    SKIP_LOSS=True).to('cuda')

    def forward(self, x: torch.Tensor):
        y = self.model(x)
        return y
