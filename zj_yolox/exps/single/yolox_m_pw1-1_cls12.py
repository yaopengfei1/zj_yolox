#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # Define yourself dataset path
        #self.data_dir = "/home/lsc/dataset/12/COCO"
        self.data_dir = "/data/ypf/yingyu_caiji/pw1-1/cls12"
        self.train_ann = "train_cls12_pw1-1.json"
        self.val_ann = "val_cls12_pw1-1.json"
        self.test_ann = "val_cls12_pw1-1.json"
        self.train_dir = 'images'
        self.test_dir = 'images'
        self.val_dir = 'images'
        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 5
        self.num_classes = 12
                
        # --------------- model  config ----------------- #
        self.obj_loss = 'bce'  # [bce,varifocal_loss,focal_loss]
        self.iou_loss = 'iou'  # [iou,giou,ciou,diou,eiou]
        
        
        # --------------- transform config ----------------- #
        # self.img_mode = 'gray'
        
        self.album = True
        self.mosaic_prob = 0.4
        self.mixup_prob = 0.0
        self.hsv_prob = 0.6
        self.degrees = 5.0
        self.translate = 0.1
        self.mosaic_scale = (0.8 , 1.2)

        # --------------  training config --------------------- #
        self.multiscale_range = 3
        self.basic_lr_per_img = 0.01 / 64.0

        # -----------------  testing config ------------------ #
        # image size
        self.img_size = 640
        self.input_size = (self.img_size, self.img_size)  # (height, width)
        self.test_size = (self.img_size, self.img_size)
        self.test_conf = 0.01
        self.nmsthre = 0.65
        
        self.color_channel = 1 if self.img_mode=='gray' else 3
