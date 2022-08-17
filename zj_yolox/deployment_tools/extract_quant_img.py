import argparse
import json
import os
import shutil
import pdb
import pandas as pd 

def extract_quant_img(json_path, img_path, save_path, single_class_img_quantity):
    json_dict = json.load(open(json_path, 'r'))
    image_map = {}
    ex_img_count = {}
    ex_img_id = set([])
    for image in json_dict['images'][::-1]:
        image_map[int(image['id'])] = image['file_name']

    for anno in json_dict['annotations'][::-1]:
        category_id = int(anno['category_id'])
        img_id = int(anno['image_id'])
        if img_id in ex_img_id:
            continue
        if category_id in ex_img_count.keys():
            if ex_img_count[category_id] < single_class_img_quantity:
                ex_img_count[category_id] += 1
                ex_img_id.add(img_id)
        else:
            ex_img_count[category_id] = 1
            ex_img_id.add(img_id)
    #####抽取背景图片
    img_df=pd.DataFrame(json_dict['images'])
    ann_df=pd.DataFrame(json_dict['annotations'])
    ann_df_list=ann_df['image_id'].tolist()
    img_df=img_df[~img_df['id'].isin(ann_df_list)]
    bg_image_list=img_df['id'].tolist()
    bg_id=[bg_image_list[id] for id in range(0,single_class_img_quantity)]
    ex_img_count['bg']=single_class_img_quantity
    # ex_img_id.add(id for id in bg_id)
    for id in  bg_id:
        ex_img_id.add(id)
    # pdb.set_trace()

    save_img_path = os.path.join(img_path, save_path)
    if os.path.exists(save_img_path):
        shutil.rmtree(save_img_path)
    os.makedirs(os.path.join(img_path, save_path), exist_ok=True)
    for img_id in ex_img_id:
        image_name = image_map[img_id]
        # pdb.set_trace()
        shutil.copy(os.path.join(img_path, image_name), os.path.join(img_path, save_path))
    print('extract quant image success,  counts:', ex_img_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("extract quant img")
    parser.add_argument("-json_path", type=str, default='/ypf/yingyu_caiji/mix7_v2/cls12/annotations/train_cls12_mix7_v2_gl_erwema_yiwu_beijing_525.json')
    parser.add_argument("-img_path", type=str, default='/data/ypf/image_place/images')
    parser.add_argument("-save_path", type=str, default='/data/ypf/mix7_train')
    parser.add_argument("-single_class_img_quantity", type=int, default=20)
    args = parser.parse_args()
    extract_quant_img(args.json_path, args.img_path, args.save_path, args.single_class_img_quantity)
