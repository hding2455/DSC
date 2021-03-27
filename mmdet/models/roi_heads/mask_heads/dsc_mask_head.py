from mmcv.cnn import ConvModule, build_conv_layer
import torch.nn as nn

from mmdet.models.builder import HEADS
from .fcn_mask_head import FCNMaskHead


@HEADS.register_module()
class DSCMaskHead(FCNMaskHead):

    def __init__(self, with_conv_res=True, *args, **kwargs):
        super(DSCMaskHead, self).__init__(*args, **kwargs)
        self.with_conv_res = with_conv_res
        if self.with_conv_res:
            norm_cfg = dict(type='BN', requires_grad=True) 
            self.conv_res = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=norm_cfg)

    def init_weights(self):
        super(DSCMaskHead, self).init_weights()
        if self.with_conv_res:
            self.conv_res.init_weights()

    def forward(self, x, res_feat=None, return_logits=True, return_feat=True):
        if res_feat is not None:
            assert self.with_conv_res
            res_feat = self.conv_res(res_feat)
            res_feat = nn.functional.adaptive_avg_pool2d(res_feat, x.shape[-2:])
            x = x + res_feat
        for conv in self.convs:
            x = conv(x)
        res_feat = x
        outs = []
        if return_logits:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
            mask_pred = self.conv_logits(x)
            outs.append(mask_pred)
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]
