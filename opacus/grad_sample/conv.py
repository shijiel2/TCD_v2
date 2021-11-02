#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
from opacus.utils.tensor_utils import unfold3d

from .utils import register_grad_sampler
import torch.nn.functional as F


@register_grad_sampler([nn.Conv1d, nn.Conv2d, nn.Conv3d])
def compute_conv_grad_sample(
    layer: Union[nn.Conv2d, nn.Conv1d],
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for convolutional layers

    args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    # TODO: No clue what is up when stride is not (1, 1)
    if layer.stride == (1, 1):
        # Things are hardcoded for conv2d at the moment
        assert len(layer.stride) == 2
        assert len(layer.padding) == 2
        assert len(layer.dilation) == 2

        batch_size = activations.shape[0]
        I = activations.shape[1]
        O = backprops.shape[1]

        activations_ = activations.transpose(0, 1)
        backprops_ = backprops.view(backprops.shape[0] * backprops.shape[1], 1, backprops.shape[2], backprops.shape[3])

        weight_grad_sample = F.conv2d(activations_, backprops_, None, layer.stride, layer.padding, layer.dilation, groups=batch_size)
        weight_grad_sample = weight_grad_sample.view(I, batch_size, O, *weight_grad_sample.shape[-2:])
        weight_grad_sample = weight_grad_sample.movedim(0, 2)


        ret = {
            layer.weight: weight_grad_sample
        }
        if layer.bias is not None:
            ret[layer.bias] = torch.sum(backprops, dim=[-1, -2])

        return ret
