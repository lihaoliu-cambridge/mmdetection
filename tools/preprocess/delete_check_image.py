import os
import json
import numpy as np
import pandas as pd
import cv2
import PIL 
from get_instance_label import instance_label


# train image -> FirstFrame_annotate/ .jpg
# train label -> FirstFrame_annotatation/ .png

# test image  -> Fully_annotate/ .jpg
# test label  -> Fully__annotatation/ .png

image_suffix = ".jpg"
label_suffix = ".png"
json_suffix = ".json"


def main(splits, image_folder, label_folder, image_output_folder, semantic_mask_output_folder, instance_mask_output_folder):
    with open(splits) as file:
        lines = [line.rstrip() for line in file]

        for idx, file_path in enumerate(lines):
            if idx not in [2105]:
                continue
            print(idx)

            image_path = os.path.join(image_folder, file_path+image_suffix)
            label_path = os.path.join(label_folder, file_path+label_suffix)
            json_path = os.path.join(image_folder, file_path+json_suffix)
            assert(os.path.exists(image_path) and os.path.exists(label_path))

            # generate annotations: json -> ins mask -> json
            if not os.path.exists(instance_mask_output_folder):
                os.makedirs(instance_mask_output_folder) 
            
            img_pil = PIL.Image.open(image_path)
            width, height = img_pil.size
            print(1, width, height)

            img_np = np.asarray(img_pil)
            # print(1, np.asarray(img_np).shape)
            assert len(np.unique(img_np)) > 10

            type_map, instance_map = instance_label(json_path)
            print(2, type_map.shape)
            type_map_255 = (255. * (1.0 * type_map - type_map.min()) / type_map.max()).astype(int)
            type_map_255 = np.concatenate([type_map_255[:, :, np.newaxis]]*3, axis=-1)
            # print(2, type_map_255.shape)

            blended = 0.5 * img_np + 0.5 * type_map_255

            pil_img = PIL.Image.fromarray(np.uint8(blended))
            pil_img.save(os.path.join(instance_mask_output_folder, "{}.png".format(idx)))

            # if idx >= 300:
            #     break


if __name__ == "__main__":
    train_splits = "../../data/traffic_cambridge/splits/supervised/train.txt"
    
    train_image_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/FirstFrame_annotate"
    train_label_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/FirstFrame_annotatation"

    train_image_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco_test/train2017"
    train_semantic_mask_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco_test/train2017_semantic_masks"
    train_instance_mask_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco_test/annotations"

    main(train_splits, train_image_folder, train_label_folder, train_image_output_folder, train_semantic_mask_output_folder, train_instance_mask_output_folder)
    


    test_splits = "../../data/traffic_cambridge/splits/supervised/test.txt"

    test_image_folder =  "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/Fully_annotate"
    test_label_folder =  "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/Fully_annotatation"

    test_image_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco_test/test2017"
    test_semantic_mask_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco_test/test2017_semantic_masks"
    test_instance_mask_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco_test/annotations"

    main(test_splits, test_image_folder, test_label_folder, test_image_output_folder, test_semantic_mask_output_folder, test_instance_mask_output_folder)
