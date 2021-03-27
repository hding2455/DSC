import torch

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


@BBOX_SAMPLERS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, add_gt_as_proposals=True, **kwargs):
        self.add_gt_as_proposals = add_gt_as_proposals

    def _sample_pos(self, **kwargs):
        """Sample positive samples"""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples"""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, gt_labels,**kwargs):
        """Directly returns the positive and negative indices  of samples

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results

        """
        
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        #print("before:",bboxes.shape[0],pos_inds.shape[0], neg_inds.shape[0], gt_bboxes.shape[0])
        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        #print("in sample",gt_flags.sum())
        #print( torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1))
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        #print("sample",pos_inds)
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        #print("after:",bboxes.shape[0],pos_inds.shape[0], neg_inds.shape[0], gt_bboxes.shape[0])
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
