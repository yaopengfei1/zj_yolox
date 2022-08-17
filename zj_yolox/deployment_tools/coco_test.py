import contextlib
import io
import json
import os.path
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def vis_npu():
    coco_org = json.load(open('datasets/COCO_12/annotations/instances_test2017.json'))
    img_kv = {}
    for item in coco_org['images']:
        img_kv[item['id']] =item['file_name']
    base_dir= 'datasets/COCO_12/test2017'
    npu_res = json.load(open('yolox_testdev_npu.json'))
    co_dets={}
    for item in npu_res:
        if co_dets.__contains__(item['image_id']):
            co_dets[item['image_id']].append(item['bbox'])
        else:
            co_dets[item['image_id']]=[item['bbox']]


    for item in npu_res:
        img_path=os.path.join(base_dir,img_kv[item['image_id']])
        bboxs=co_dets[item['image_id']]
        img=cv2.imread(img_path)
        for bb in bboxs:
            img=cv2.rectangle(img,(int(bb[0]),int(bb[1])),(int(bb[0]+bb[2]),int(bb[1]+bb[3])),(255,255,0),thickness=1)
        cv2.imshow('1',img)
        cv2.waitKey(0)
        cv2.imwrite(img_path.replace('test2017','npu2017'),img)


def convert_local(json_path):
    coco_org = json.load(open('datasets/COCO_12/annotations/instances_test2017.json'))
    img_kv = {}
    for item in coco_org['images']:
        img_kv[item['file_name']] = item['id']

    coco_dt = json.load(open(json_path))
    for item in coco_dt:
        img_id = img_kv[item['image_id'].split('/')[-1]]
        item['image_id'] = img_id
        item['category_id'] = item['category_id']+1
    json.dump(coco_dt, open(json_path, "w"))


def evaluate(results_json, targets_json):
    cocoGt = COCO(targets_json)
    coco_dets = cocoGt.loadRes(results_json)
    cocoEval = COCOeval(cocoGt, coco_dets, 'bbox')
    cocoEval.params.iouThrs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    info=''
    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        cocoEval.summarize()
    info += redirect_string.getvalue()
    print(info)
    return cocoEval.stats[0], cocoEval.stats[1], info




if __name__ == '__main__':
    #vis_npu()
    convert_local('yolox_testdev_onnx.json')
    evaluate('yolox_testdev_onnx.json', 'datasets/COCO/annotations/instances_test2017.json')