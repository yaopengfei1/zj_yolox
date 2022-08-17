#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import pdb
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.img_size = 320
        # ---------------- model config ---------------- #
        self.depth = 1.00
        self.width = 1.00
        self.act = 'relu'
        self.album = False  #True #

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 16
        self.input_size = (self.img_size, self.img_size)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.data_dir = None
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_test2017.json"
        self.test_ann = "instances_test2017.json"
        self.train_dir = 'train2017'
        self.test_dir = 'test2017'
        self.val_dir = 'val2017'

        # --------------- transform config ----------------- #
        self.img_mode = 'BGR'
        self.brightness_mean = None
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.5, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = False

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15  # 在最后的15个epoch关闭强数据增强 mixup和mosaic
        self.min_lr_ratio = 0.05
        self.ema = True
        

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = False
        self.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split(".")[0]

        
        # --------------  model  config --------------------- #
        self.backbone_name = "CSPDarknet"
        self.obj_loss = 'bce' #['bce','focal_loss','varifocal_loss']
        self.iou_loss = 'iou'
        self.cls_loss='bce'  #['bce','qfocal_loss','eqlv2']
        self.freeze = False   #
        self.image_net_pre_train=True
        self.pretrain_weights_path=''
        
        
        # -----------------  testing config ------------------ #
        self.test_size = (self.img_size, self.img_size)
        self.test_conf = 0.01
        self.nmsthre = 0.65
        self.aug_img_path='./YOLOX_outputs/aug_img'  #保存增强后的图片

        self.color_channel = 1 if self.img_mode=='gray' else 3

    def _get_model(self):
        from yolox.models import YOLOX, YOLOPAFPNWITHBB, YOLOXHead
        _in_features=("dark3", "dark4", "dark5") if self.backbone_name == "CSPDarknet" else  ("res3", "res4", "res5")
        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPNWITHBB(self.depth,
                                       self.width,
                                       in_channels=in_channels,
                                       act=self.act,
                                       in_features=_in_features,
                                       color_channel=self.color_channel,
                                       freeze=self.freeze,
                                       image_net_pre_train=self.image_net_pre_train,
                                       pretrain_path=self.pretrain_weights_path)
            head = YOLOXHead(self.num_classes,
                             self.width,
                             in_channels=in_channels,
                             act=self.act,
                             iou_loss_type=self.iou_loss,
                             obj_loss=self.obj_loss,cls_loss=self.cls_loss)
            self.model = YOLOX(backbone, head)

    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self._get_model()
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self,
                        batch_size,
                        is_distributed,
                        no_aug=False,
                        cache_img=False):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                name=self.train_dir,
                preproc=TrainTransform(max_labels=50,
                                       flip_prob=self.flip_prob,
                                       hsv_prob=self.hsv_prob),
                cache=cache_img,
                img_mode=self.img_mode,
                brightness_mean=self.brightness_mean
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            is_album=self.album,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=120,
                                   flip_prob=self.flip_prob,
                                   hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset),
                                  seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True
        }
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(inputs,
                                               size=tsize,
                                               mode="bilinear",
                                               align_corners=False)
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        def get_pg_module(M, pg):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d):
                    pg0.append(v.weight)  # no decay
                elif "bn" in k and isinstance(v, nn.ModuleList):
                    for m in v:
                        pg0.append(m.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(
                        v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(pg0,
                                        lr=lr,
                                        momentum=self.momentum,
                                        nesterov=True)
            optimizer.add_param_group({
                "params": pg1,
                "weight_decay": self.weight_decay
            })  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self,
                        batch_size,
                        is_distributed,
                        testdev=False,
                        legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name=self.val_dir if not testdev else self.test_dir,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            img_mode=self.img_mode,
            brightness_mean=self.brightness_mean
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset,
                                                 **dataloader_kwargs)

        return val_loader

    def get_evaluator(self,
                      batch_size,
                      is_distributed,
                      testdev=False,
                      legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev,
                                          legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
            color_channel=self.color_channel
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)
