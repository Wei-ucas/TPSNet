import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from mmocr.utils.tps_util import TPS
from mmdet.core import multi_apply
from mmdet.models.builder import LOSSES

Pi = np.pi


@LOSSES.register_module()
class TPSLoss(nn.Module):

    def __init__(self, num_fiducial=8, num_sample=20,num_fiducial_gt=14, ohem_ratio=3.,gauss_center=False,
                point_loss=False,with_BA=False,border_relax_thr=1.0, with_weight=True,with_area_weight=True
                 , fiducial_dist="cross",steps = [8,16,32]
                 ):
        super().__init__()
        self.eps = 1e-6
        self.num_fiducial = num_fiducial
        self.num_fiducial_gt = num_fiducial_gt
        self.num_sample = num_sample
        self.ohem_ratio = ohem_ratio
        self.with_BA = with_BA
        self.with_point_loss = point_loss
        self.with_center_weight = with_weight
        self.with_area_weight = with_area_weight
        self.gauss_center = gauss_center
        self.fiducial_dist = fiducial_dist
        self.steps = steps
        self.border_relax_thr = border_relax_thr

        self.tps_decoder_gt = TPS(num_fiducial_gt, num_points=num_sample if not self.with_BA else 5*num_sample,grid_size=(2,2))
        self.tps_decoder = TPS(num_fiducial, num_points=num_sample,grid_size=(2,2), fiducial_dist=fiducial_dist)


    def forward(self, preds, _, p3_maps, p4_maps, p5_maps,polygons_area=None, **kwargs):

        assert isinstance(preds, list)
        assert p3_maps[0].shape[0] == 2 * (self.num_fiducial_gt + 3) + 3

        device = preds[0][0].device
        gts = [p3_maps, p4_maps, p5_maps]
        if self.with_area_weight:
            assert polygons_area is not None
            max_num_polygon = max([len(p) for p in polygons_area])
            pad_polygon_areas = torch.zeros(len(polygons_area), max_num_polygon, device=device)
            for bi, po in enumerate(polygons_area):
                if len(po) == 0:
                    continue
                pad_polygon_areas[bi, :len(po)] = torch.from_numpy(polygons_area[bi]).to(device)
        else:
            pad_polygon_areas = None
        gt_polygons_areas = [pad_polygon_areas] * 3
        down_sample_rates = self.steps

        for idx, maps in enumerate(gts):
            gts[idx] = maps.float()

        losses = multi_apply(self.forward_single, preds, gts, down_sample_rates, gt_polygons_areas)

        loss_tr = torch.tensor(0., device=device,requires_grad=True).float()
        loss_tcl = torch.tensor(0., device=device,requires_grad=True).float()
        loss_point = torch.tensor(0., device=device,requires_grad=True).float()
        loss_ba = torch.tensor(0.0, device=device,requires_grad=True).float()

        for idx, loss in enumerate(losses):
            if idx == 0:
                loss_tr = loss_tr + sum(loss)
            elif idx == 1:
                loss_tcl = loss_tcl + sum(loss)
            elif idx == 2:
                loss_point = loss_point + sum(loss)
            elif idx == 3:
                loss_ba = loss_ba + sum(loss)

        results = dict(
            loss_text=loss_tr,
            loss_center=loss_tcl,
        )

        if self.with_point_loss:
            results['loss_point'] = loss_point
        if self.with_BA:
            results['loss_ba'] = loss_ba

        return results

    def smooth_ce_loss(self, preds, targets):
        log_preds = -F.log_softmax(preds, dim=-1)
        return targets*log_preds[:,1] + (1-targets)*log_preds[:,0]

    def forward_single(self, pred, gt, downsample_rate=None,areas=None):
        cls_pred = pred[0].permute(0, 2, 3, 1).contiguous()
        reg_pred = pred[1].permute(0, 2, 3, 1).contiguous()
        gt = gt.permute(0, 2, 3, 1).contiguous()

        tr_pred = cls_pred[:, :, :, :2].view(-1, 2)
        tcl_pred = cls_pred[:, :, :, 2:4].view(-1, 2)
        tps_pred = reg_pred[:, :, :, :].view(-1, (self.num_fiducial + 3) * 2)

        if self.with_BA or self.with_area_weight:
            tr_mask_idx = gt[:, :, :, :1].long()
            tr_mask = (tr_mask_idx != 0).view(-1)
            batch_size, H, W, _ = tr_mask_idx.shape
            batch_idx = torch.arange(0, batch_size)[:, None, None].repeat(1, H, W).to(cls_pred.device)
            batch_idx = batch_idx.view(-1)
            tr_mask_idx = tr_mask_idx.view(-1) - 1
        else:
            tr_mask = gt[:, :, :, :1].view(-1)

        tcl_mask = gt[:, :, :, 1:2].view(-1)
        train_mask = gt[:, :, :, 2:3].view(-1)
        tps_map = gt[:, :, :, 3:].view(-1, (self.num_fiducial_gt + 3) * 2)


        tr_train_mask = ((train_mask * tr_mask) > 0).float()
        device = tps_pred.device
        # tr loss
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())

        # num_pos = tr_train_mask.sum().item()
        pos_idx = torch.where(tr_train_mask > 0)[0]
        # tcl loss
        loss_tcl = torch.tensor(0.).float().to(device)
        tr_neg_mask = 1 - tr_train_mask
        if tcl_mask.max() > 1:
            print(tcl_mask)
        assert tcl_mask.min() >= 0
        if tr_train_mask.sum().item() > 0:
            if self.gauss_center:
                loss_tcl_pos = self.smooth_ce_loss(
                    tcl_pred[pos_idx],
                    tcl_mask[pos_idx].float())
            else:
                loss_tcl_pos = F.cross_entropy(
                    tcl_pred[pos_idx],
                    tcl_mask[pos_idx].long(), reduction='none')

            loss_tcl_pos = torch.mean(loss_tcl_pos)
            loss_tcl_neg = F.cross_entropy(tcl_pred[tr_neg_mask.bool()],
                                           tcl_mask[tr_neg_mask.bool()].long())
            loss_tcl = loss_tcl_pos + 0.5 * loss_tcl_neg


        # regression loss
        loss_point = torch.tensor(0.,device=device,requires_grad=True).float()
        loss_ba = torch.tensor(0.,device=device,requires_grad=True).float()
        num_pos = tr_train_mask.sum().item()
        if num_pos> 0 and  (self.with_point_loss or self.with_BA) :
            tps_map = tps_map[pos_idx]
            tps_pred = tps_pred[pos_idx]
            if self.with_area_weight:
                batch_idx = batch_idx[pos_idx]
                tr_mask_idx = tr_mask_idx[pos_idx]
            if self.with_center_weight:
                weight = (tr_mask[pos_idx].float() +
                          tcl_mask[pos_idx].float()) / 2
                weight = weight.contiguous()
            else:
                weight = torch.ones(num_pos,dtype=torch.float32, device=tps_map.device)

            if self.with_area_weight:
                pos_area = areas[batch_idx, tr_mask_idx] / downsample_rate**2
                num_instance = torch.sum(areas > 0)
                if num_instance == 0:
                    return loss_tr, loss_tcl, loss_point, loss_ba
                if torch.any(pos_area<=1):
                    pos_area[pos_area <=1] = 100000000
                area_weight = 1.0/pos_area
                weight = weight * area_weight * (1.0/num_instance)
            else:
                weight = weight * 1.0/pos_idx.shape[0]

            p_gt = self.tps_decoder_gt.build_P_border(tps_map)  # N * (n_sample*4) * 2
            p_pre = self.tps_decoder.build_P_border(tps_pred)

            boder_gt = p_gt
            boder_pre = p_pre

            if self.with_BA:
                cor_gt = self.tps_decoder_gt.build_P_grid(tps_map)
                cor_pre = self.tps_decoder.build_P_grid(tps_pred)
                loss_cor = F.smooth_l1_loss(cor_pre, cor_gt,reduction='none').mean(dim=(-2)).sum(-1)
                loss_ba = self.ba_loss(boder_pre[:,:self.num_sample//2], boder_gt[:,:self.num_sample*5//2], scale=downsample_rate) + self.ba_loss(boder_pre[:,self.num_sample//2:], boder_gt[:,self.num_sample*5//2:], scale=downsample_rate)
                loss_ba = torch.sum(weight*(loss_cor+loss_ba))

            if self.with_point_loss:
                # weight = weight.view(-1,1)
                ft_x, ft_y =boder_gt[:,:,0], boder_gt[:,:,1]
                ft_x_pre, ft_y_pre = boder_pre[:,:,0], boder_pre[:,:,1]
                loss_reg_x = torch.sum( weight*F.smooth_l1_loss(
                    ft_x_pre[:, :],
                    ft_x[:, :],
                    reduction='none').mean(dim=-1))
                loss_reg_y = torch.sum( weight* F.smooth_l1_loss(
                    ft_y_pre[:, :],
                    ft_y[:, :],
                    reduction='none').mean(dim=-1))
                loss_point = loss_reg_x + loss_reg_y

        return loss_tr, loss_tcl, loss_point, loss_ba


    def ba_loss(self, pred_boundary, gt_boundary, scale):
        '''
        :param pred_boundary:  N x n x 2
        :param gt_boundary: N x n_gt x 2
        :return:
        '''
        dist2gt_point = torch.norm((pred_boundary[:,:,None,:] - gt_boundary[:,None, :, :])**2, dim=-1) # N x n x n_gt
        dist2gt_boundary, index = dist2gt_point.min(dim=-1)
        # print(dist2gt_boundary)
        loss= F.smooth_l1_loss(pred_boundary, torch.gather(gt_boundary, 1, index.unsqueeze(-1).expand(-1,-1,2)), reduction='none').sum(-1)
        loss[dist2gt_boundary < (1 / scale) *(1 - self.border_relax_thr)] = 0.0
        return loss.mean(-1)

    def ohem(self, predict, target, train_mask):
        pos = (target * train_mask).bool()
        neg = ((1 - target) * train_mask).bool()

        n_pos = pos.float().sum()

        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(
                predict[pos], target[pos], reduction='sum')
            loss_neg = F.cross_entropy(
                predict[neg], target[neg], reduction='none')
            n_neg = min(
                int(neg.float().sum().item()),
                int(self.ohem_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.)
            loss_neg = F.cross_entropy(
                predict[neg], target[neg], reduction='none')
            n_neg = 100
        if len(loss_neg) > n_neg:
            loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()