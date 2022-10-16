import torch
from torch import nn
import numpy as np
# from mmocr.models.textdet.postprocess.wrapper import fill_hole, poly_nms
import cv2
import torch.nn.functional as F
from mmocr.core.evaluation.utils import boundary_iou
import Polygon as plg
from mmocr.utils.tps_util import TPS

def check_polygon(polygon):
    polygon = np.array(polygon).reshape(-1,2)
    pp = plg.Polygon(polygon)
    if pp.area() < 30 or pp.orientation()[0] != 1:
        return False
    return True


def poly_nms(polygons, threshold, with_index=False):
    assert isinstance(polygons, list)
    keep_poly = []
    keep_index = []
    if len(polygons) != 0:
        polygons = np.array(polygons)
        scores = polygons[:, -1]
        sorted_index = np.argsort(scores)
        polygons = polygons[sorted_index]
        # polygons = np.array(sorted(polygons, key=lambda x: x[-1]))


        index = [i for i in range(polygons.shape[0])]
        vaild_index = np.ones(len(index))
        for i in range(len(index)):
            if not check_polygon(polygons[index[i]][:-1]):
                vaild_index[i] = 0
        invalid_index = np.where(vaild_index==0)
        index = np.delete(index, invalid_index)

        while len(index) > 0:
            keep_poly.append(polygons[index[-1]].tolist())
            keep_index.append(sorted_index[index[-1]])
            A = polygons[index[-1]][:-1]
            index = np.delete(index, -1)

            iou_list = np.zeros((len(index), ))
            for i in range(len(index)):
                B = polygons[index[i]][:-1]

                iou_list[i] = boundary_iou(A, B)
            remove_index = np.where(iou_list > threshold)
            index = np.delete(index, remove_index)

    if with_index:
        return keep_poly, keep_index
    else:
        return keep_poly

def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return ~canvas | input_mask


def tps_decode( preds,
                        decoder,
                        scale,
                        alpha=1.0,
                        beta=2.0,
                        text_repr_type='poly',
                        score_thr=0.3,
                        nms_thr=0.9,
                        test_cfg=None,
                        ):
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert text_repr_type == 'poly'

    cls_pred = preds[0][0]
    # if not gt_val:
    tr_pred = cls_pred[0:2].softmax(dim=0).data.cpu().numpy()
    tcl_pred = cls_pred[2:4].softmax(dim=0).data.cpu().numpy()
        # if with_direction:
        #     direction_pred = cls_pred[4:].softmax(dim=0)
    score_pred = (tr_pred[1] ** alpha) * (tcl_pred[1] ** beta)**(1.0/(alpha+beta))
    # else:
    #     score_pred = cls_pred[0]
    #     # direction_map =
    #     preds[1] = torch.tensor(preds[1], dtype=torch.float32)

    reg_pred = preds[1][0].permute(1, 2, 0)
    tps_pred = reg_pred[:, :, :].reshape((-1, (decoder.num_fiducial + 3) * 2))

    tr_pred_mask = (score_pred) > score_thr
    tr_mask = fill_hole(tr_pred_mask)

    tr_contours, _ = cv2.findContours(
        tr_mask.astype(np.uint8), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)  # opencv4

    mask = np.zeros_like(tr_mask)
    boundaries = []
    grids = []
    for cont in tr_contours:
        deal_map = mask.copy().astype(np.int8)
        cv2.drawContours(deal_map, [cont], -1, 1, -1)

        score_map = score_pred * deal_map
        score_mask = score_map > 0
        xy_text = np.argwhere(score_mask)
        tps_c = tps_pred[score_mask.reshape(-1)]
        polygons= decoder.build_P_border(tps_c)
        sample_grids = decoder.build_P_grid(tps_c)
        polygons = polygons.cpu().numpy()
        sample_grids = sample_grids
        score = score_map[score_mask].reshape(-1, 1)

        polygons[:, :, 0] += xy_text[:, 1, None]
        polygons[:, :, 1] += xy_text[:, 0, None]
        sample_grids[:,:,0] += torch.from_numpy(xy_text[:,1, None]).to(sample_grids.device)
        sample_grids[:,:,1] += torch.from_numpy(xy_text[:,0, None]).to(sample_grids.device)
        # if polygons.shape[0] == 0:
        #     continue
        polygons = polygons.reshape(polygons.shape[0], -1) * scale
        sample_grids = sample_grids
        polygons, keep_index = poly_nms(np.hstack((polygons, score)).tolist(), nms_thr, with_index=True)
        # polygons = polygons.tolist()
        if len(keep_index) > 0:
            grids = grids + [sample_grids[keep_index]]
            boundaries = boundaries + polygons

    boundaries,keep_index = poly_nms(boundaries, 1.0, with_index=True)
    if len(grids) > 0:
        grids = torch.cat(grids,dim=0)[keep_index]
        # grids = grids.cpu().numpy() * scale
    return boundaries, grids