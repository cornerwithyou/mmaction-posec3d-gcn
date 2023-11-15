# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList

from mmaction.registry import MODELS
from .focal_func import SubMConv,Sparse,spsresnet,spsresnet_test,spsconvall,spnet

@MODELS.register_module()
class spconv(BaseModule):
    def __init__(self,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.model = spsconvall(num_input_features=17,
                    ds_factor=8,
                    TOPK=True,
                    USE_IMG=True,
                    SKIP_LOSS=True).to('cuda')

    def forward(self, x: torch.Tensor):
        y = self.model(x)
        return y



@MODELS.register_module()
class SubM(BaseModule):
    def __init__(self,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.model = SubMConv(num_input_features=17,
                    ds_factor=8,
                    TOPK=True,
                    USE_IMG=True,
                    SKIP_LOSS=True).to('cuda')

    def forward(self, x: torch.Tensor):
        y = self.model(x)
        return y

@MODELS.register_module()
class SpsResnet(BaseModule):
    def __init__(self,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        print(kwargs)
        self.model = spsresnet(num_input_features=17,
                    stem=kwargs['stem'] if 'stem' in kwargs else 'sub',
                    conv1=kwargs['conv1'] if 'conv1' in kwargs else 'spss',
                    conv2=kwargs['conv2'] if 'conv2' in kwargs else 'spss',
                    conv3=kwargs['conv3'] if 'conv3' in kwargs else 'spss',
                    num_classes=kwargs['num_classes'] if 'num_classes' in kwargs else 60,
                    ds_factor=8,
                    TOPK=True,
                    USE_IMG=True,
                    SKIP_LOSS=True).to('cuda')

    def forward(self, x: torch.Tensor):
        y = self.model(x)
        return y

@MODELS.register_module()
class SpsResnet_test(BaseModule):
    def __init__(self,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.model = spsresnet_test(num_input_features=17,
                    ds_factor=8,
                    TOPK=True,
                    USE_IMG=True,
                    SKIP_LOSS=True).to('cuda')

    def forward(self, x: torch.Tensor):
        y = self.model(x)
        return y

@MODELS.register_module()
class Spnet(BaseModule):
    def __init__(self,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.model = spnet(num_input_features=17,
                                    ).to('cuda')

    def forward(self, x):
        y = self.model(x)
        return y
