#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xywh2xyxy",
    "xyxy2cxcywh",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    #customed = True
    customed = False
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if customed:
            single_output = customed_nms(detections, (detections[:, 4] * detections[:, 5]),
                                         nms_thre)
            single_output = torch.from_numpy(single_output)
            if output[i] is None:
                output[i] = single_output
            else:
                output[i] = torch.cat((output[i], single_output))
        else:
            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )

            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

    return output


def rect_union(rect_a, rect_b):
    out_rect = rect_a.copy()
    out_rect[0] = min(rect_a[0], rect_b[0])
    out_rect[1] = min(rect_a[1], rect_b[1])
    out_rect[2] = max(rect_a[2], rect_b[2])
    out_rect[3] = max(rect_a[3], rect_b[3])
    if out_rect[2] < out_rect[0]:
        out_rect = [0, 0, 0, 0]
    if out_rect[3] < out_rect[1]:
        out_rect = [0, 0, 0, 0]
    return out_rect


def rect_interact(rect_a, rect_b):
    out_rect = rect_a.copy()
    out_rect[0] = max(rect_a[0], rect_b[0])
    out_rect[1] = max(rect_a[1], rect_b[1])
    out_rect[2] = min(rect_a[2], rect_b[2])
    out_rect[3] = min(rect_a[3], rect_b[3])
    if out_rect[2] < out_rect[0]:
        out_rect = [0, 0, 0, 0]
    if out_rect[3] < out_rect[1]:
        out_rect = [0, 0, 0, 0]
    return out_rect


def customed_nms(bboxes, confs, thres):
    s_confs, s_rank = confs.sort(descending=True)
    s_confs = s_confs.cpu().numpy()
    s_rank = s_rank.cpu().numpy()
    bboxes = bboxes.cpu().numpy()
    s_bboxes = bboxes[s_rank]
    picked = []
    for i, box in enumerate(s_bboxes):
        keep = True
        for j in picked:
            rect_i = rect_interact(box, s_bboxes[j, :])
            #rect_u = rect_union(box, s_bboxes[j, :])
            rea_i = (rect_i[2] - rect_i[0])*(rect_i[3] - rect_i[1])
            rea_a = (box[2]-box[0])*(box[3]-box[1])
            rea_b = (s_bboxes[j, 2]-s_bboxes[j, 0]) * (s_bboxes[j, 3]-s_bboxes[j, 1])
            rea_u = rea_a + rea_b - rea_i
            if box[6] == s_bboxes[j, 6] and (rea_i/rea_u > thres or rea_a * 0.8 < rea_i or rea_b * 0.8 < rea_i):
                keep = False

        if keep:
            picked.append(i)

    bboxes[s_rank[picked], :] = s_bboxes[picked, :]
    return bboxes[s_rank[picked], :]


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xywh2xyxy(bboxes):
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
