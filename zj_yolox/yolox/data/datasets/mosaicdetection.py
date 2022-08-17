#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import random
import time
import cv2
import numpy as np

from yolox.utils import adjust_box_anns, get_local_rank, xyxy2xywh, xywh2xyxy

from ..data_augment import random_affine
from .datasets_wrapper import Dataset

import albumentations as album

random.seed(2)

#os.environ.setdefault('debug', 'images')


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h,
                          input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w,
                                     input_w * 2), min(input_h * 2,
                                                       yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


def augmentation_with_album(transforms: list, img, label, aug_filter=False):
    transform1 = transforms[0]
    transform2 = transforms[1]

    bboxes = label[:, :4].copy()
    category_ids = label[:, 4:].copy()

    if random.random() < 0.5:
        img = transform1(image=img)['image']

    else:
        if label.shape[0] != 0:
            height = img.shape[0]
            width = img.shape[1]
            np.clip(bboxes[:, 0], 0, width - 2, out=bboxes[:, 0])
            np.clip(bboxes[:, 1], 0, height - 2, out=bboxes[:, 1])
            np.clip(bboxes[:, 2], 0, width - 1, out=bboxes[:, 2])
            np.clip(bboxes[:, 3], 0, height - 1, out=bboxes[:, 3])
            bboxes = xyxy2xywh(bboxes)
            np.clip(bboxes[:, 2], 1, width - 1, out=bboxes[:, 2])
            np.clip(bboxes[:, 3], 1, height - 1, out=bboxes[:, 3])
            # ori_bboxes = bboxes.copy()
            transformed = transform2(image=img.copy(),
                                     bboxes=bboxes,
                                     category_ids=category_ids)
            bboxes = np.array(transformed['bboxes'])
            if len(bboxes) <= 0:
                label = np.zeros((0, 5), dtype=float)
            else:
                category_ids = np.array(transformed['category_ids'])
                if aug_filter:
                    get_invalid_obj = False
                    for bbox in bboxes:
                        if ((bbox[0] == 0 or bbox[0]+bbox[2]>width-1) and (bbox[2] < width*0.08)) or \
                        ((bbox[1] == 0 or bbox[1]+bbox[3]>height-1) and (bbox[3] < height*0.08)):
                            get_invalid_obj = True
                    if get_invalid_obj:
                        img = transform1(image=img)['image']
                        return img, label

                bboxes = xywh2xyxy(bboxes)
                label = np.hstack((bboxes, category_ids))
        else:
            transformed = transform2(image=img,
                                     bboxes=bboxes,
                                     category_ids=category_ids)
        img = transformed['image']
    return img, label


def save_aug_img(img, label, sample_idx, debug_dir):
    time_array = time.localtime(time.time())
    time_tag = time.strftime("%Y-%m-%d_%H-%M-%S", time_array)
    img_name = "{}_{}.jpg".format(time_tag, sample_idx)
    img_aug = img.transpose(1, 2, 0).copy()
    for bbox_idx in range(label.shape[0]):
        label_int = [int(i) for i in label[bbox_idx]]
        left_top = tuple(label_int[:2])
        right_bottom = tuple(label_int[2:4])
        # cls_idx = label_int[4]
        img_aug = cv2.rectangle(img_aug,
                                left_top,
                                right_bottom, (255, 0, 0),
                                thickness=1)
    cv2.imwrite(os.path.join(debug_dir, img_name), img_aug)


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(self,
                 dataset,
                 img_size,
                 mosaic=True,
                 is_album=True,
                 preproc=None,
                 degrees=10.0,
                 translate=0.1,
                 mosaic_scale=(0.5, 1.5),
                 mixup_scale=(0.5, 1.5),
                 shear=2.0,
                 enable_mixup=True,
                 mosaic_prob=1.0,
                 mixup_prob=1.0,
                 *args):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.album = is_album
        self.local_rank = get_local_rank()

        # bbox不变性增强
        transform1 = album.Compose([
            album.HueSaturationValue(hue_shift_limit=20,
                                     sat_shift_limit=30,
                                     val_shift_limit=(-50, 10),
                                     p=0.5),
            album.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2,
                                           p=0.5),
            album.Blur(blur_limit=3, p=0.3),
            album.MedianBlur(blur_limit=3, p=0.2),
            album.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, p=0.2),
            album.MotionBlur(blur_limit=7, p=0.1),
            album.GaussNoise(var_limit=(50, 500), mean=0, p=0.3),
            album.ISONoise(color_shift=(0.01, 0.10),
                           intensity=(0.1, 0.8),
                           p=0.2),
            album.GridDropout(ratio=0.1,
                              shift_x=0,
                              shift_y=0,
                              fill_value=(114, 114, 114),
                              p=0.1),
            album.CoarseDropout(max_holes=10,
                                max_height=8,
                                max_width=8,
                                fill_value=(114, 114, 114),
                                p=0.1)
        ])

        # bbox变换增强
        transform2 = album.Compose([
            album.HorizontalFlip(p=0.5),
            album.VerticalFlip(p=0.5),
            album.ShiftScaleRotate(shift_limit=0.01,
                                   scale_limit=0.1,
                                   rotate_limit=5,
                                   border_mode=cv2.BORDER_CONSTANT,
                                   shift_limit_x=None,
                                   value=(114, 114, 114),
                                   p=0.5)
        ],
                                   bbox_params=album.BboxParams(
                                       format='coco',
                                       label_fields=['category_ids']))
        self.transforms = [transform1, transform2]

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [
                random.randint(0,
                               len(self._dataset) - 1) for _ in range(3)
            ]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id = self._dataset.pull_item(index)  # _labels format: (x1, y1, x2, y2)
                if self.album and random.random() < 0.5:
                    img, _labels = augmentation_with_album(
                        self.transforms, img, _labels)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)),
                                 interpolation=cv2.INTER_LINEAR)
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c),
                                         114,
                                         dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2,
                 l_y2), (s_x1, s_y1, s_x2,
                         s_y2) = get_mosaic_coordinate(mosaic_img, i_mosaic,
                                                       xc, yc, w, h, input_h,
                                                       input_w)

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0],
                        0,
                        2 * input_w,
                        out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1],
                        0,
                        2 * input_h,
                        out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2],
                        0,
                        2 * input_w,
                        out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3],
                        0,
                        2 * input_h,
                        out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
            )

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (self.enable_mixup and not len(mosaic_labels) == 0
                    and random.random() < self.mixup_prob):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img,
                                                       mosaic_labels,
                                                       self.input_dim)

            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels,
                                                  self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])
            label = mosaic_labels
            padded_img = mix_img
            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            # return mix_img, padded_labels, img_info, img_id

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id = self._dataset.pull_item(idx)
            input_h, input_w = self.input_dim[0], self.input_dim[1]
            if self.album and random.random() < 0.5 :
                img, label = augmentation_with_album(self.transforms, img,
                                                     label)

            padded_img, padded_labels = self.preproc(img, label,
                                                     self.input_dim)


        return padded_img, padded_labels, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones(
                (input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0],
                             input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio),
             int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[:int(img.shape[0] *
                    cp_scale_ratio), :int(img.shape[1] *
                                          cp_scale_ratio)] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor),
             int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3),
            dtype=np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        cp_bboxes_origin_np = adjust_box_anns(cp_labels[:, :4].copy(),
                                              cp_scale_ratio, 0, 0, origin_w,
                                              origin_h)
        if FLIP:
            cp_bboxes_origin_np[:,
                                0::2] = (origin_w -
                                         cp_bboxes_origin_np[:, 0::2][:, ::-1])
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w)
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h)

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(
            np.float32)

        return origin_img.astype(np.uint8), origin_labels
