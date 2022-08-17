#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tqdm import tqdm
import cv2
import pathlib
import os

import torch
from terminaltables import AsciiTable

from yolox.utils import (gather, is_main_process, postprocess, synchronize,
                         time_synchronized, xyxy2xywh)
import matplotlib.pyplot as plt
import numpy as np


def plot_pr_curve(coco_eval):
    # extract eval data
    precisions = coco_eval.eval["precision"]
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    pr_01 = precisions[0, :, 0, 0, 2]
    pr_array1 = precisions[0, :, 0, 0, 2]
    pr_array2 = precisions[0, :, 1, 0, 2]
    pr_array3 = precisions[0, :, 2, 0, 2]
    pr_array4 = precisions[0, :, 3, 0, 2]
    pr_array5 = pr_01
    pr_array6 = pr_01
    pr_array7 = pr_01
    pr_array8 = pr_01
    pr_array9 = pr_01
    pr_array10 = pr_01

    #print('==========', pr_array1)

    x = np.arange(0.0, 1.01, 0.01)
    # plot PR curve
    plt.plot(x, pr_array1, label="iou=0.10")
    plt.plot(x, pr_array2, label="iou=0.15")
    plt.plot(x, pr_array3, label="iou=0.20")
    plt.plot(x, pr_array4, label="iou=0.25")
    plt.plot(x, pr_array5, label="iou=0.30")
    plt.plot(x, pr_array6, label="iou=0.35")
    plt.plot(x, pr_array7, label="iou=0.40")
    plt.plot(x, pr_array8, label="iou=0.45")
    plt.plot(x, pr_array9, label="iou=0.50")
    plt.plot(x, pr_array10, label="iou=0.55")

    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.show()


class COCOEvaluator:
    """
    COCO_ AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO_ API.
    """

    def __init__(self,
                 dataloader,
                 img_size,
                 confthre,
                 nmsthre,
                 num_classes,
                 testdev=False,
                 color_channel=3):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.color_channel = color_channel

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        experiment_name='',
    ):
        """
        COCO_ average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO_ API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO_ AP of IoU=50:95
            ap50 (float) : COCO_ AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, infos, info_imgs,
                       ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                imgs = imgs[:, :self.color_channel]
                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(outputs, self.num_classes, self.confthre,
                                      self.nmsthre)

                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(
                self.convert_to_coco_format(outputs, info_imgs, ids))
            #self.save_image_result('result', imgs, outputs, ids)

        statistics = torch.cuda.FloatTensor(
            [inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics, experiment_name)
        synchronize()
        return eval_results

    def save_image_result(self, dir, imgs, outputs, ids):
        dir = pathlib.Path(dir)
        if dir.exists() is False:
            dir.mkdir()
        imgs_info = self.dataloader.dataset.coco.imgs

        data_dir = self.dataloader.dataset.data_dir
        for (output, id) in zip(outputs, ids):
            id = int(id)
            name = imgs_info[id]['file_name']
            name_dst = os.path.join(str(dir), name)
            name_ori = os.path.join(data_dir, self.dataloader.dataset.name,
                                    name)
            img = cv2.imread(name_ori)
            anns_ids = self.dataloader.dataset.coco.imgToAnns[id]
            gt_color = (0, 255, 0)
            for anns_id in anns_ids:
                anns_info = self.dataloader.dataset.coco.anns[anns_id['id']]
                bbox = anns_info['bbox']
                category_id = anns_info['category_id']
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                              gt_color, 2)
                label_id = 'gt_{}'.format(category_id)
                cv2.putText(img, label_id, (int(bbox[0]), int(bbox[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, gt_color, 2)

            cv2.imwrite(name_dst, img)

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(outputs, info_imgs[0],
                                                  info_imgs[1], ids):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(self.img_size[0] / float(img_h),
                        self.img_size[1] / float(img_w))
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                #print("-------------------", int(cls[ind]))
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO_ json format
                data_list.append(pred_data)
        return data_list

    def class_wise_caculate(self, cocoEval):
        # Compute per-category AP
        # from https://github.com/facebookresearch/detectron2/
        precisions = cocoEval.eval['precision']
        # precision: (iou, recall, cls, area range, max dets)
        cat_ids = cocoEval.cocoGt.getCatIds()
        assert len(cat_ids) == precisions.shape[2]

        results_per_category = []
        mAP = 0.0
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per im
            nm = cocoEval.cocoGt.cats[idx + 1]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = float(np.mean(precision))
            else:
                ap = float('nan')
            mAP += ap
            results_per_category.append(
                            (f'{nm["name"]}', f'{ap:0.3f}'))
        mAP /= (idx+1)
        results_per_category.append(('平均(mAP)', f'{mAP:0.3f}'))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        #logger.info('\n' + table.table)
        return "\n*********\nclass wise average precision is:\n {}".format(
            table.table)

    def evaluate_prediction(self, data_dict, statistics, experiment_name=''):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples *
                                                self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join([
            "Average {} time: {:.2f} ms".format(k, v) for k, v in zip(
                ["forward", "NMS", "inference"],
                [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
            )
        ])

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                res_path = f"YOLOX_outputs/{experiment_name}/yolox_testdev_2017_new.json"
                json.dump(data_dict, open(res_path, "w"))
                cocoDt = cocoGt.loadRes(res_path)
            else:
                # _, tmp = tempfile.mkstemp()
                # json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(data_dict)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])

            #x = np.arange(0.05, 0.55, 0.05)
            cocoEval.params.iouThrs = [
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
            ]
            #cocoEval.params.iouThrs = [0.5]
            # cocoEval.params.recThrs =
            # print(cocoEval.params.iouThrs)
            # exit()

            # TODO 更改IOU为0.1
            cocoEval.evaluate()
            cocoEval.accumulate()
            table = self.class_wise_caculate(cocoEval)
            plot_pr_curve(cocoEval)

            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            info += table
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
