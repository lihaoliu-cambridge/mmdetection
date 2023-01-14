#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import imgviz
import numpy as np
import pandas as pd

import labelme

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


def main(stage="train"):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()

    output_dir = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco/"

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    if not osp.exists(osp.join(output_dir, "annotations")):
        os.makedirs(osp.join(output_dir, "annotations"))
    print("Creating dataset:", output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[
            dict(
                url=None,
                id=0,
                name=None,
            )
        ],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
                {'id': 1, 'name': 'Motor_Bike'},
                {'id': 2, 'name': 'Bus'},
                {'id': 3, 'name': 'LMV'},
                {'id': 4, 'name': 'Auto'},
                {'id': 5, 'name': 'Bike'},
                {'id': 6, 'name': 'Pedestrian'},
                {'id': 7, 'name': 'LCV'},
                {'id': 8, 'name': 'E-rickshaw'},
                {'id': 9, 'name': 'Tractor'},
                {'id': 10,'name': 'Truck'}
        ],
    )

    # class_name_to_id = {}
    # for i, line in enumerate(open(args.labels).readlines()):
    #     class_id = i - 1  # starts with -1
    #     class_name = line.strip()
    #     if class_id == -1:
    #         assert class_name == "__ignore__"
    #         continue
    #     class_name_to_id[class_name] = class_id
    #     data["categories"].append(
    #         dict(
    #             supercategory=None,
    #             id=class_id,
    #             name=class_name,
    #         )
    #     )
    class_name_to_id = {'_background_': 0,
                'Motor Bike': 1,
                'Bus': 2,
                'LMV': 3,
                'Auto': 4,
                'Bike': 5,
                'Pedestrian': 6,
                'Pesestrian': 6,
                'LCV': 7,
                'E-rickshaw': 8,
                'Tractor': 9,
                'Truck': 10,
                # some annotation noise
                'Moter Bike': 1,
                'Moterbike': 1,
                'MotoeBike': 1,
                'm': 1,
                'e-rikshwa': 8,
                'e-rickshaw': 8,
                'MotorBike': 1,
                'cycle': 5,
                'BUS': 2,
                'truck': 10,
                'MortorBike': 1,
                'Motor Bike': 1,
                'e_rickshaw': 8,
                'Pedestrain': 6,
                'e-rickshow': 8,
                'e-Rickshaw': 8,
                'e-ricshaw': 8,
                'MoterBike': 1,
                'Rickshow': 8,
                'eRickshaw': 8,
                'Truk': 10,
                'Trauck': 10,
                'Motor bike': 1,
                'MototBike': 1,
                'bus': 1,
                'MotorBIke': 1,
                'Motir Bike': 1
                }

    out_ann_file = osp.join(output_dir, "annotations", "instances_{}2017.1_4-unlabeled.json".format(stage))
    
    
    ## todo here
    images_txt_path = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/splits/semi-supervised/option3/1_4/train_unlabel.txt"
    
    df = pd.read_csv("/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/tools/preprocess/3_semi_supervised/all/train_all_image_size_info.csv")
    with open(images_txt_path) as file:
        lines = [line.rstrip() for line in file]

        for tmp_idx, file_path in enumerate(lines):
            query = df.loc[df['image_path'] == file_path]
            image_id = int(query["image_id"].values[0][:-4])
            filename = str(query["image_id"].values[0].replace(".jpg", ".json"))
            img_width = int(query["width"].values[0])
            img_height = int(query["height"].values[0])
            # print(tmp_idx, image_id, file_path)
            print(image_id)
            
            # if image_id <= 1050:
            #     continue

            # error file
            if '02129' in filename:
                continue

            # print("Generating dataset from:", filename)

            # json_file = os.path.join("/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/FirstFrame_annotate", file_path+".json")
            # label_file = labelme.LabelFile(filename=json_file)

            # base = osp.splitext(osp.basename(filename))[0]
            # out_img_file = osp.join(output_dir, "{}2017".format(stage), base + ".jpg")

            # img = labelme.utils.img_data_to_arr(label_file.imageData)
            # imgviz.io.imsave(out_img_file, img)
            data["images"].append(
                dict(
                    license=0,
                    url=None,
                    file_name= filename.split("/")[-1].split(".")[0]+".jpg",
                    height=img_height,
                    width=img_width,
                    date_captured=None,
                    id=image_id,
                )
            )

            masks = {}  # for area
            segmentations = collections.defaultdict(list)  # for segmentation
            # print(type(img.shape[:2]), img.shape[:2], (img_height, img_width))
            segmentations = dict(segmentations)

            # if image_id >= 2400:
            #     break
            
            # if not args.noviz:
            #     viz = img
            #     if masks:
            #         labels, captions, masks = zip(
            #             *[
            #                 (class_name_to_id[cnm], cnm, msk)
            #                 for (cnm, gid), msk in masks.items()
            #                 if cnm in class_name_to_id
            #             ]
            #         )
            #         viz = imgviz.instances2rgb(
            #             image=img,
            #             labels=labels,
            #             masks=masks,
            #             captions=captions,
            #             font_size=15,
            #             line_width=2,
            #         )
            #     out_viz_file = osp.join(
            #         output_dir, "{}2017_vis".format(stage), base + ".jpg"
            #     )
            #     imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main(stage="train")