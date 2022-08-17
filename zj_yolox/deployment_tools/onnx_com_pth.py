#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

#对齐onnx和pth

import argparse
import os
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="../reshape.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", type=str, default='YOLOX_outpus/yolox_m/best_ckpt.pth',help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default='exps/example/custom/yolox_m.py',
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default='yolox-m', help="model name")
    parser.add_argument("-c", "--ckpt", default='/home/lsc/workspace/YOLOX-main/YOLOX_outputs/yolox_m/best_ckpt.pth', type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False

    _input = torch.ones(1, 3, 640, 640)

    outputs = model(_input)
    import onnxruntime
    session = onnxruntime.InferenceSession('./bc5.onnx')
    ort_inputs = {session.get_inputs()[0].name: _input.numpy()}
    output = session.run(None, ort_inputs)[0]
    print('-------------------------------pth-----------------------------------------')
    print(outputs)
    print('-------------------------------onnx-----------------------------------------')
    print(output)
    torch.testing.assert_allclose(outputs, output, rtol=1e-07, atol=1e-04) #测试onnx结果和pth结果是否对齐






if __name__ == "__main__":
    main()
