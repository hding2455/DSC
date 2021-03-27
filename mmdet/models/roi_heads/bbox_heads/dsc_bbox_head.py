import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.core import multi_apply, force_fp32
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead


@HEADS.register_module()
class DSCBBoxHead(ConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 conv_res=None,
                 *args,
                 **kwargs):
        super(DSCBBoxHead, self).__init__(num_shared_convs=num_shared_convs,
                 num_shared_fcs=num_shared_fcs,
                 num_cls_convs=num_cls_convs,
                 num_cls_fcs=num_cls_fcs,
                 num_reg_convs=num_reg_convs,
                 num_reg_fcs=num_reg_fcs,
                 conv_out_channels=conv_out_channels,
                 fc_out_channels=fc_out_channels,
                 conv_cfg=conv_cfg,
                 norm_cfg=norm_cfg,
                 *args, 
                 **kwargs)
        self.with_conv_res = False
        if conv_res is not None:
            self.with_conv_res = True
            norm_cfg = dict(type='BN', requires_grad=True)
            self.conv_res = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels,
                conv_res,
                padding= int(conv_res / 2),
                conv_cfg=self.conv_cfg,
                norm_cfg=norm_cfg)

    def init_weights(self):
        super(DSCBBoxHead, self).init_weights()
        if self.with_conv_res:
            self.conv_res.init_weights()

    def forward(self, x, res_feat=None):
        if res_feat is not None and self.with_conv_res:
            res_feat = self.conv_res(res_feat)
            res_feat = nn.functional.adaptive_avg_pool2d(res_feat, x.shape[-2:])
            x = x + res_feat
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        res_feat = x
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, res_feat

    def _get_unsampled_target_single(self, bboxes, pos_inds, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        num_samples = bboxes.size(0)
        num_pos = bboxes[pos_inds].size(0) 

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = bboxes.new_ones(num_samples)
        bbox_targets = bboxes.new_zeros(num_samples, 4)
        bbox_weights = bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    bboxes[pos_inds], pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_unsampled_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        bboxes_list = [res.unsampled_bboxes for res in sampling_results]
        pos_inds_list = [res.pos_inds for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_unsampled_target_single,
            bboxes_list,
            pos_inds_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def enlarge_bboxes(self, bboxes, img_shape, ratio=2):
        new_bboxes = torch.zeros_like(bboxes)
        new_bboxes.copy_(bboxes)
        w = new_bboxes[:,2] - new_bboxes[:,0]
        h = new_bboxes[:,3] - new_bboxes[:,1]
        new_bboxes[:,0] -= w * (ratio - 1) / 2
        new_bboxes[:,2] += w * (ratio - 1) / 2
        new_bboxes[:,1] -= h * (ratio - 1) / 2
        new_bboxes[:,3] += h * (ratio - 1) / 2
        new_bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
        new_bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
        return new_bboxes

    @force_fp32(apply_to=('bbox_pred', ))
    def refine_bboxes(self, rois, enlarged_rois, next_enlarge_ratio, labels, bbox_preds, pos_is_gts, img_metas):
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)
        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            enlarged_rois_ = enlarged_rois[inds, :]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]
            pos_inds = label_ < label_.max()
            bboxes = self.regress_by_class(bboxes_, enlarged_rois_, next_enlarge_ratio,label_, bbox_pred_,
                                           img_meta_)
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[pos_inds] = pos_keep
            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])
        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, enlarged_rois, next_enlarge_ratio, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            bboxes = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])

        new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        #bboxes = new_rois[:,1:]
        enlarged_bboxes = enlarged_rois[:, 1:]
        next_enlarge_bboxes = self.enlarge_bboxes(bboxes, img_meta['img_shape'], next_enlarge_ratio)
        clamp_low_inds = next_enlarge_bboxes[:,[0,2]] < enlarged_bboxes[:,[0,2]]
        next_enlarge_bboxes[:,[0,2]][clamp_low_inds] = enlarged_bboxes[:,[0,2]][clamp_low_inds]
        clamp_high_inds = next_enlarge_bboxes[:,[1,3]] > enlarged_bboxes[:,[1,3]]
        next_enlarge_bboxes[:,[1,3]][clamp_high_inds] = enlarged_bboxes[:,[1,3]][clamp_high_inds]
        clamped_bboxes = self.enlarge_bboxes(next_enlarge_bboxes, img_meta['img_shape'], 1.0/next_enlarge_ratio)

        clamped_rois = torch.cat((new_rois[:, [0]], clamped_bboxes), dim=1)
        if rois.size(1) == 4:
            return clamped_bboxes
        else:
            return clamped_rois

@HEADS.register_module()
class Shared2FCDSCBBoxHead(DSCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCDSCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class Shared4Conv1FCDSCBBoxHead(DSCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCDSCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
