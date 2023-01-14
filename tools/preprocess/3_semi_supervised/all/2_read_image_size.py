import os
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# train image -> FirstFrame_annotate/ .jpg
# train label -> FirstFrame_annotatation/ .png

# test image  -> Fully_annotate/ .jpg
# test label  -> Fully__annotatation/ .png

image_suffix = ".jpg"
label_suffix = ".png"
json_suffix = ".json"


def main(splits, image_folder):
    with open(splits) as file:
        lines = [line.rstrip() for line in file]

        images_info_list = []
        
        for idx, file_path in enumerate(lines):
            print(idx)

            image_path = os.path.join(image_folder, file_path+image_suffix) 
            img_pil = Image.open(image_path)
            width, height = img_pil.size

            images_info = ["{:05d}{}".format(idx, image_suffix), file_path, width, height]
            images_info_list.append(images_info)

            # if idx >= 300:
            #     break
            
        df = pd.DataFrame(images_info_list, columns = ['image_id', 'image_path', 'width', 'height'])
        df.to_csv("/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/tools/preprocess/3_semi_supervised/all/train_all_image_size_info.csv", index=False)

            

if __name__ == "__main__":
    train_splits = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/tools/preprocess/3_semi_supervised/all/train_label_all.txt"
    train_image_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/FirstFrame_annotate"
    
    main(train_splits, train_image_folder)