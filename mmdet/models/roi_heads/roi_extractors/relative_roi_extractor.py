import torch

from mmdet.core import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class RelativeRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides=[1.0]):
        super(RelativeRoIExtractor, self).__init__(roi_layer, out_channels,
                                                 featmap_strides)
    def compute_relative_rois(self, rois, base_rois, feature_shape):
        #rois[:,1][rois[:,1] < base_rois[:,1]] = base_rois[:,1]
        #rois[:,2][rois[:,2] < base_rois[:,2]] = base_rois[:,2]
        #rois[:,3][rois[:,3] > base_rois[:,3]] = base_rois[:,3]
        #rois[:,4][rois[:,4] > base_rois[:,4]] = base_rois[:,4]
        base_w = base_rois[:,3] - base_rois[:,1]
        base_h = base_rois[:,4] - base_rois[:,2]
        relative_rois = torch.zeros_like(rois)
        relative_rois[:,0] = torch.arange(len(relative_rois)).to(dtype=relative_rois.dtype)
        relative_rois[:,1] = (rois[:,1] - base_rois[:,1]) / base_w
        relative_rois[:,3] = (rois[:,3] - base_rois[:,1]) / base_w
        relative_rois[:,2] = (rois[:,2] - base_rois[:,2]) / base_h
        relative_rois[:,4] = (rois[:,4] - base_rois[:,2]) / base_h
        relative_rois[:,1] = relative_rois[:,1]*feature_shape[1]
        relative_rois[:,3] = relative_rois[:,3]*feature_shape[1]
        relative_rois[:,2] = relative_rois[:,2]*feature_shape[0]
        relative_rois[:,4] = relative_rois[:,4]*feature_shape[0]
        return relative_rois

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, base_rois):
        """Forward function"""
        out_size = self.roi_layers[0].out_size
        feature_shape = feats.shape[-2:]
        relative_rois = self.compute_relative_rois(rois,base_rois,feature_shape)
        if len(rois) == 0:
            return feats.new_zeros(
                rois.size(0), self.out_channels, *out_size)
        return self.roi_layers[0](feats, relative_rois)
    
