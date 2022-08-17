import argparse
import contextlib
import io
import json
import pdb
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def convert_local(dt_json_path, gt_json):
    print(dt_json_path)
    print(gt_json)
    # pdb.set_trace()
    coco_org = json.load(open(gt_json))
    img_kv = {}
    for item in coco_org['images']:
        img_kv[item['file_name']] = item['id']
    # pdb.set_trace()
    coco_dt = json.load(open(dt_json_path))
    for item in coco_dt:
        if '/' in str(item['image_id']):
            name = item['image_id'].split('/')[-1]
        else :
            name=item['image_id']
        if not img_kv.keys().__contains__(name):
            coco_dt.remove(item)
            continue
        if '/' in str(item['image_id']):
            img_id = img_kv[item['image_id'].split('/')[-1]]
        else :
            img_id=img_kv[item['image_id']]
        item['image_id'] = img_id
        item['category_id'] = item['category_id'] + 1
    json.dump(coco_dt, open(dt_json_path, "w"))


def evaluate(results_json, targets_json):
    cocoGt = COCO(targets_json)
    coco_dets = cocoGt.loadRes(results_json)
    cocoEval = COCOeval(cocoGt, coco_dets, 'bbox')
    cocoEval.params.iouThrs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    info = ''
    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        cocoEval.summarize()
    info += redirect_string.getvalue()
    return cocoEval.stats[0], cocoEval.stats[1], info


if __name__ == '__main__':
    parser = argparse.ArgumentParser("convert and eval")
    parser.add_argument("-gt_path", type=str, default='./test.json')
    parser.add_argument("-dt_path", type=str, default='./u_model_result.json')
    parser.add_argument("-conf", type=int, default=0.1)
    args = parser.parse_args()
    convert_local(args.dt_path, args.gt_path)
    _, _, info = evaluate(args.dt_path, args.gt_path)
    print(info)

