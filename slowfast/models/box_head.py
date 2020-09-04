import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, Linear, ShapeSpec
from detectron2.modeling.roi_heads import ROI_BOX_HEAD_REGISTRY


@ROI_BOX_HEAD_REGISTRY.register()
class HOIRCNNConvFCHead(nn.Module):
    """
    A head with several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_fc: the number of fc layers
            fc_dim: the dimension of the fc layers
        """
        super().__init__()

        # fmt: off
        num_fc = cfg.MODEL.HOI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.HOI_BOX_HEAD.FC_DIM
        # fmt: on
        assert num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self.fcs = []
        for k in range(num_fc):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))
        return x

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])


def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)


def build_hoi_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.HOI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.HOI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)