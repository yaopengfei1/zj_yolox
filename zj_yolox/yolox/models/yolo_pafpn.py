#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .res2net_v1b import res2net50_v1b_26w_csp
import math
from loguru import logger

class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        color_channel=3,
        freeze=False,
        image_net_pre_train=True,
        pretrain_path=None
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act, in_channel=color_channel)
        if not image_net_pre_train:
            logger.info('CSPDarknet [classify ] weight')
            model_weights_path = pretrain_path
            pre_weights = torch.load(model_weights_path)['model']
            net_weights=self.backbone.state_dict()
            new_dict={}
            for k,v in pre_weights.items():
                if not net_weights.keys().__contains__(k):
                    continue
                if net_weights[k].numel() == v.numel():
                    new_dict[k]=v        
            logger.info(f'CSPDarknet load dict,  match state dict count: {len(new_dict)}')
            self.backbone.load_state_dict(new_dict,strict=False)#missing count:80  load_dict:286-80
        if freeze:
            logger.info('CSPDarknet [freeze]')
            for p in self.backbone.parameters():
                p.requires_grad = False

            
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width),
                                      int(in_channels[1] * width),
                                      1,
                                      1,
                                      act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width),
                                     int(in_channels[0] * width),
                                     1,
                                     1,
                                     act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(int(in_channels[0] * width),
                             int(in_channels[0] * width),
                             3,
                             2,
                             act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(int(in_channels[1] * width),
                             int(in_channels[1] * width),
                             3,
                             2,
                             act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


class YOLOPAFPNWITHBB(YOLOPAFPN):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(self,
                 depth=1.0,
                 width=1.0,
                 in_features=("dark3", "dark4", "dark5"),
                 in_channels=[256, 512, 1024],
                 depthwise=False,
                 act="silu",
                 backbone_name="CSPDarknet",
                 color_channel=3,
                 freeze=False,
                 image_net_pre_train=True,
                 pretrain_path=None):
        super(YOLOPAFPNWITHBB, self).__init__(depth=depth,
                                              width=width,
                                              in_features=in_features,
                                              in_channels=in_channels,
                                              depthwise=depthwise,
                                              act=act,
                                              color_channel=color_channel,freeze=freeze,image_net_pre_train=image_net_pre_train,pretrain_path=pretrain_path)
        if backbone_name == "res2net50_v1b_26w_csp":
            if math.isclose(width, 0.5):
                self.backbone = res2net50_v1b_26w_csp(True, depth, color_channel=color_channel,freeze=freeze,image_net_pre_train=image_net_pre_train,pretrain_path=pretrain_path)
            elif math.isclose(width, 1.0):
                self.backbone = res2net50_v1b_26w_csp(False,
                                                      depth,
                                                      expansion=2,
                                                      color_channel=color_channel,freeze=freeze,image_net_pre_train=image_net_pre_train,pretrain_path=pretrain_path)
            else:
                raise Exception("not support width or depth parameters")