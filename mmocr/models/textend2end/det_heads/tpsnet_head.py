import torch.nn as nn
import torch
from mmcv.cnn import normal_init, ConvModule
from mmcv.runner import BaseModule
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmocr.models.textend2end.postprocess import tps_decode
from mmocr.models.textdet.dense_heads.head_mixin import HeadMixin
from mmocr.utils.tps_util import TPS
from ..postprocess.tps_decoder import  poly_nms
import numpy as np


@HEADS.register_module()
class TPSHead(HeadMixin, BaseModule):

    def __init__(self,
                 in_channels,
                 scales,
                 num_fiducial=8,
                 fiducial_dist='cross',
                 num_fiducial_gt=14,
                 num_sample=20,
                 num_reconstr_points=20,
                 sample_size=(8,32),
                 # decoding_type='tps',
                 loss=dict(type='TPSLoss'),
                 score_thr=0.3,
                 nms_thr=0.1,
                 alpha=1.0,
                 beta=1.0,
                 num_convs=0,
                 dcn=False,
                 use_sigmod=False,
                 train_cfg=None,
                 test_cfg=None):

        super().__init__()
        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.scales = scales
        self.num_fiducial = num_fiducial
        self.sample_num = num_sample
        self.num_reconstr_points = num_reconstr_points
        loss['num_fiducial'] = num_fiducial
        loss['fiducial_dist'] = fiducial_dist
        loss['num_fiducial_gt'] = num_fiducial_gt
        loss['num_sample'] = num_sample
        loss['steps'] = scales
        self.use_sigmod = use_sigmod
        if self.use_sigmod:
            loss['use_sigmod'] = use_sigmod
        self.loss_module = build_loss(loss)
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.alpha = alpha
        self.beta = beta
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_convs = num_convs
        self.dcn = dcn
        self.sample_size = sample_size

        if self.use_sigmod:
            p = 1
        else:
            p=2
        self.out_channels_cls = 2 * p
        self.out_channels_reg = (self.num_fiducial + 3) * 2

        self.decoder = TPS(self.num_fiducial,num_points=num_reconstr_points,grid_size=sample_size, fiducial_dist=fiducial_dist)

        if self.num_convs > 0:
            cls_convs = []
            reg_convs = []
            if self.dcn:
                conv_cfg = dict(type='DCNv2', deform_groups=2)
            else:
                conv_cfg = None
            # norm = dict(type='GN', num_groups=32, requires_grad=True)
            norm = None
            for i in range(self.num_convs):

                cls_convs.append(ConvModule(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1,
                                            conv_cfg=conv_cfg if i < 3 else None, norm_cfg=norm, act_cfg=dict(type='ReLU')))
                reg_convs.append(ConvModule(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1,
                                            conv_cfg=conv_cfg if i < 3 else None, norm_cfg=norm, act_cfg=dict(type='ReLU')))
            self.cls_convs = nn.Sequential(*cls_convs)
            self.reg_convs = nn.Sequential(*reg_convs)

        self.out_conv_cls = nn.Conv2d(
            self.in_channels,
            self.out_channels_cls,
            kernel_size=3,
            stride=1,
            padding=1)
        self.out_conv_reg = nn.Conv2d(
            self.in_channels,
            self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1)

        self.init_weights()

    def init_weights(self):
        normal_init(self.out_conv_cls, mean=0, std=0.01)
        normal_init(self.out_conv_reg, mean=0, std=0.01)

    def forward(self, feats):
        cls_res, reg_res = multi_apply(self.forward_single, feats)
        level_num = len(cls_res)
        preds = [[cls_res[i], reg_res[i]] for i in range(level_num)]
        return preds

    def forward_single(self, x):
        if self.num_convs > 0:
            x_cls = self.cls_convs(x)
            x_reg = self.reg_convs(x)
        else:
            x_cls = x
            x_reg = x
        cls_predict = self.out_conv_cls(x_cls)
        reg_predict = self.out_conv_reg(x_reg)
        return cls_predict, reg_predict

    def resize_grid(self, grids, scale_factor):
        assert isinstance(scale_factor, np.ndarray)
        assert scale_factor.shape[0] == 4
        assert grids[0].shape[-1] == 2

        for g in grids:
            sz = g.shape[0]
            g[:] = g[:] * np.tile(scale_factor[:2], (sz,1))
        return grids



    def get_boundary(self, score_maps, img_metas, rescale):

        assert len(score_maps) == len(self.scales)

        boundaries = []
        grids = []
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundary, grid = self._get_boundary_single(
                score_map, scale, False)
            boundaries = boundaries + boundary
            if len(grid) > 0:
                grids = grids + [grid*scale]

        # nms
        boundaries, keep_index = poly_nms(boundaries, self.nms_thr, with_index=True)
        if len(grids) > 0:
            grids = torch.cat(grids, dim=0)[keep_index]

        if rescale:
            boundaries = self.resize_boundary(
                boundaries, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=boundaries, grids_results=grids,scales=self.scales)
        return results



    def _get_boundary_single(self, score_map, scale, gt_vis=False):
        assert len(score_map) == 2


        return tps_decode(
            # decoding_type=self.decoding_type,
            preds=score_map,
            decoder=self.decoder,
            scale=scale,
            alpha=self.alpha,
            beta=self.beta,
            text_repr_type='poly',
            score_thr=self.score_thr,
            nms_thr=self.nms_thr,
            # gt_val=gt_vis,
            # with_direction=self.with_direction,
            test_cfg=self.test_cfg
        )
