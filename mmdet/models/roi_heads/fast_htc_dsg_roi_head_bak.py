import torch
import torch.nn.functional as F
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class FastHybridTaskCascadeDSGRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 relative_roi_extractor=None,
                 mpn=None,
                 mpn_roi_extractor=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 enlarge_ratio=[2.0, 1.41, 1.0],
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        #if mpn is not None:
        #    self.init_mpn(mpn_roi_extractor, mpn)
        super(FastHybridTaskCascadeDSGRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        if relative_roi_extractor is not None:
            self.relative_roi_extractor = build_roi_extractor(relative_roi_extractor)
        if mpn is not None:
            self.init_mpn(mpn_roi_extractor, mpn)
        if semantic_head is not None:
            self.semantic_roi_extractor = build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = build_head(semantic_head)
        self.enlarge_ratio = enlarge_ratio

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mpn(self, mpn_roi_extractor, mpn):
        self.mpn = nn.ModuleList()
        if not isinstance(mpn, list):
            mpn = [mpn for _ in range(self.num_stages)]
        assert len(mpn) == self.num_stages
        for head in mpn:
            self.mpn.append(build_head(head))
        self.mpn_roi_extractor = nn.ModuleList()
        if not isinstance(mpn_roi_extractor, list):
            mpn_roi_extractor = [
                mpn_roi_extractor for _ in range(self.num_stages)
            ]
        assert len(mpn_roi_extractor) == self.num_stages
        for roi_extractor in mpn_roi_extractor:
            self.mpn_roi_extractor.append(
                build_roi_extractor(roi_extractor))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = build_head(mask_head)
        self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage"""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for rcnn_train_cfg in self.train_cfg:
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.bbox_sampler.append(build_sampler(rcnn_train_cfg.sampler))

    @property
    def with_mpn(self):
        return hasattr(self, 'mpn') and self.mpn is not None
    
    @property
    def with_semantic(self):
        return hasattr(self, 'semantic_head') and self.semantic_head is not None

    def init_weights(self, pretrained):
        """Initialize the weights in head

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mpn:
                self.mpn_roi_extractor[i].init_weights()
                self.mpn[i].init_weights()
        if self.with_mask:
            self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()
        if self.with_semantic:
            self.semantic_head.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function"""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'], )
        return outs
    def enlarge_rois(self, rois, img_shape, ratio=2):
        new_rois = torch.zeros_like(rois)
        new_rois.copy_(rois)
        w = new_rois[:,3] - new_rois[:,1]
        h = new_rois[:,4] - new_rois[:,2]
        new_rois[:,1] -= w * (ratio - 1) / 2
        new_rois[:,3] += w * (ratio - 1) / 2
        new_rois[:,2] -= h * (ratio - 1) / 2
        new_rois[:,4] += h * (ratio - 1) / 2
        new_rois[:, [1, 3]].clamp_(min=0, max=img_shape[1])
        new_rois[:, [2, 4]].clamp_(min=0, max=img_shape[0])
        return new_rois

    def _bbox_forward(self, stage, x, rois,  semantic_feat,  img_shape, enlarge=False):
        """Box head forward function used in both training and testing"""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        probability_map = None
        res_feat = None
        if enlarge:
            enlarged_rois = self.enlarge_rois(rois, img_shape, ratio=self.enlarge_ratio[stage])
        else:
            enlarged_rois = rois

        #base_rois = mpn_results['base_rois']
        #res_feat = mpn_results['res_feat']
        #res_feat = self.relative_roi_extractor(res_feat, enlarged_rois, base_rois)

        mpn_feats = self.mpn_roi_extractor[stage](
            x[:self.mpn_roi_extractor[stage].num_inputs], enlarged_rois)
        for i in range(stage):
            tmp_res_feat = self.mpn[i](mpn_feats, res_feat, return_logits=False)
            #if res_feat is not None:
            #    res_feat = res_feat + tmp_res_feat
            #else:
            res_feat = tmp_res_feat
        probability_map_pred, tmp_res_feat = self.mpn[stage](mpn_feats, res_feat)
        #if res_feat is not None:
        #    res_feat = res_feat + tmp_res_feat
        #else:
        res_feat = tmp_res_feat
        with torch.no_grad():
            probability_map = probability_map_pred.sigmoid()
        
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        enlarged_rois, masks = probability_map)

        if self.with_semantic:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat

        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats, res_feat)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg, semantic_feat, img_metas):
        """Run forward function and calculate loss for box head in training"""
        img_shape = img_metas[0]['img_shape']
        rois = bbox2roi([res.bboxes for res in sampling_results])
        if stage == 0:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        else:
            rois = bbox2roi([res.unsampled_bboxes for res in sampling_results])
            bbox_targets = self.bbox_head[stage].get_unsampled_targets(
                sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        bbox_results = self._bbox_forward(stage, x, rois,semantic_feat, img_shape, enlarge=(stage != (self.num_stages-1)))
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward(self, stage, x, rois, semantic_feat, img_shape, enlarge=False):
        """Mask head forward function used in both training and testing"""
        mask_roi_extractor = self.mask_roi_extractor
        mask_head = self.mask_head
        if enlarge:
            enlarged_rois = self.enlarge_rois(rois, img_shape, ratio=self.enlarge_ratio[stage])
        else:
            enlarged_rois = rois
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                enlarged_rois)
        if self.with_semantic:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat

        #base_rois = mpn_results['base_rois']
        #res_feat = mpn_results['res_feat']
        #res_feat = self.relative_roi_extractor(res_feat, enlarged_rois, base_rois)

        res_feat = None
        for i in range(stage):
            tmp_res_feat = self.mpn[i](mask_feats, res_feat, return_logits=False)
            #if res_feat is not None:
            #    res_feat = res_feat + tmp_res_feat
            #else:

            res_feat = tmp_res_feat

        mask_pred = mask_head(mask_feats, res_feat, return_feat=False)
        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            semantic_feat,
                            img_metas):
        """Run forward function and calculate loss for mask head in training"""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        if len(pos_rois) == 0:
            # If there are no predicted and/or truth boxes, then we cannot
            # compute head / mask losses
            return dict(loss_mask=None)
        img_shape = img_metas[0]['img_shape']
        mask_results = self._mask_forward(stage, x, pos_rois,semantic_feat, img_shape, enlarge=False)

        mask_targets = self.mask_head.get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def _mpn_forward(self, stage, x, rois, semantic_feat, img_shape, enlarge=True):
        """Mask head forward function used in both training and testing"""
        mpn_roi_extractor = self.mpn_roi_extractor[stage]
        mpn = self.mpn[stage]
        if enlarge:
            enlarged_rois = self.enlarge_rois(rois, img_shape, ratio=self.enlarge_ratio[stage])
        else:
            enlarged_rois = rois
        mpn_feats = mpn_roi_extractor(x[:mpn_roi_extractor.num_inputs],
                                enlarged_rois)

        if self.with_semantic:
            mpn_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if mpn_semantic_feat.shape[-2:] != mpn_feats.shape[-2:]:
                mpn_semantic_feat = F.adaptive_avg_pool2d(
                    mpn_semantic_feat, mpn_feats.shape[-2:])
            mpn_feats += mpn_semantic_feat

#        res_feat = None
#        if mpn_results is not None:
#            base_rois = mpn_results['base_rois']
#            res_feat = mpn_results['res_feat']
#            res_feat = self.relative_roi_extractor(res_feat, enlarged_rois, base_rois)
        res_feat = None
        for i in range(stage):
            tmp_res_feat = self.mpn[i](mpn_feats, res_feat, return_logits=False)
            #if res_feat is not None:
            #    res_feat = res_feat + tmp_res_feat
            #else:
            res_feat = tmp_res_feat

        mpn_pred, res_feat = mpn(mpn_feats, res_feat)
        mpn_results = dict(mpn_pred=mpn_pred, res_feat=res_feat, base_rois=enlarged_rois)
        return mpn_results

    def _mpn_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            semantic_feat,
                            img_metas):
        """Run forward function and calculate loss for mask head in training"""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        if len(pos_rois) == 0:
            # If there are no predicted and/or truth boxes, then we cannot
            # compute head / mask losses
            return dict(loss_mpn=None)
        img_shape = img_metas[0]['img_shape']
        mpn_results = self._mpn_forward(stage, x, pos_rois,semantic_feat, img_shape)

        mpn_targets = self.mpn[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mpn = self.mpn[stage].loss(mpn_results['mpn_pred'],
                                               mpn_targets, pos_labels)

        mpn_results.update(loss_mpn=loss_mpn)
        return mpn_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
                                   
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask or self.with_mpn:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels=gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            if self.with_mpn:
                mpn_results = self._mpn_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,semantic_feat,img_metas)
                # TODO: Support empty tensor input. #2280
                if mpn_results['loss_mpn'] is not None:
                    losses[f's{i}.loss_mpn'] = mpn_results['loss_mpn']['loss_mask'] * lw['loss_mpn']

            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg,semantic_feat, img_metas)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw[name] if 'loss' in name else value)

            # mask head forward and loss
            

            # refine bboxes
            if i < self.num_stages-1:
                #pos_is_gts = [res.pos_is_gt for res in sampling_results]
                pos_is_gts = [torch.zeros_like(res.pos_is_gt) for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    #if i > 0 and ((bbox_results['rois'] - prev_rois).abs() > 1e-1).sum() > 0:
                    #if i > 0:
                        #print(prev_rois.shape, bbox_results['rois'].shape, bbox_results['bbox_pred'].shape)
                        #print(((bbox_results['rois'] - prev_rois).abs() > 1e-1).sum())
                        #print(bbox_results['rois'][bbox_results['rois'] != prev_rois][:10], prev_rois[bbox_results['rois'] != prev_rois][:10])
                    prev_rois = bbox_results['rois']
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)
                        #torch.zeros_like(bbox_results['bbox_pred']), pos_is_gts, img_metas)
                    #total = 0
                    #for p in proposal_list:
                    #    print("len(p)",len(p))

        if self.with_mask:
            rcnn_train_cfg = self.train_cfg[self.num_stages]
            lw = self.stage_loss_weights[self.num_stages]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask or self.with_mpn:
                bbox_assigner = self.bbox_assigner[self.num_stages]
                bbox_sampler = self.bbox_sampler[self.num_stages]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels=gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            mask_results = self._mask_forward_train(
                    self.num_stages, x, sampling_results, gt_masks, rcnn_train_cfg,semantic_feat,img_metas)
            if mask_results['loss_mask'] is not None:
                losses['loss_mask'] = mask_results['loss_mask']['loss_mask'] * lw['loss_mask']

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        img_shape = img_metas[0]['img_shape']
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois, semantic_feat, img_shape, enlarge=(i != (self.num_stages-1)))
            ms_scores.append(bbox_results['cls_score'])

            if i < self.num_stages - 1:
                bbox_label = bbox_results['cls_score'].argmax(dim=1)
                rois = self.bbox_head[i].regress_by_class(
                    rois, bbox_label, bbox_results['bbox_pred'], img_metas[0])

        #cls_score = sum(ms_scores) / self.num_stages
        cls_score = (ms_scores[1]+ms_scores[2]) / 2
        #cls_score = ms_scores[2]
        det_bboxes, det_labels = self.bbox_head[-1].get_bboxes(
            rois,
            cls_score,
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head.num_classes
                segm_result = [[] for _ in range(mask_classes)]
            else:
                _bboxes = (
                    det_bboxes[:, :4] * det_bboxes.new_tensor(scale_factor)
                    if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                mask_results = self._mask_forward(self.num_stages, x, mask_rois, semantic_feat, img_shape)
                segm_result = self.mask_head.get_seg_masks(
                    mask_results['mask_pred'].sigmoid().cpu().numpy(), _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
                
            ms_segm_result['ensemble'] = segm_result

        if self.with_mask:
            results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        #TODO Semantic not added
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(
                            mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return bbox_result, segm_result
        else:
            return bbox_result
