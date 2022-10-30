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


def mask2box(mask):
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    y1 = int(np.min(rows))  # y
    x1 = int(np.min(clos))  # x
    y2 = int(np.max(rows))
    x2 = int(np.max(clos))
    print(y2-y1, x2-x1)
    if y2-y1 < 2 or x2-x1 <2:
        with open('/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/tools/preprocess/small.txt', 'a') as f:
            f.write(x1, y1, x2, y2, y2-y1, x2-x1)

    return (x1, y1, x2, y2)


def main(splits, image_folder, label_folder, image_output_folder, semantic_mask_output_folder, instance_mask_output_folder):
    with open(splits) as file:
        lines = [line.rstrip() for line in file]

        # for stats
        stats_cls_ins = []

        # for datasets
        result = {
            "info": {"description": "Cambridge Traffic Dataset."},
            "categories": [
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
            ]
        }

        images_info = []
        labels_info = []
        obj_count = 0
        
        for idx, file_path in enumerate(lines):
            print(idx, file_path)

            image_path = os.path.join(image_folder, file_path+image_suffix)
            label_path = os.path.join(label_folder, file_path+label_suffix)
            json_path = os.path.join(image_folder, file_path+json_suffix)
            assert(os.path.exists(image_path) and os.path.exists(label_path))

            # move and rename image
            if not os.path.exists(image_output_folder):
                os.makedirs(image_output_folder) 

            os.system("cp {} {}".format(image_path, os.path.join(image_output_folder, "{:05d}{}".format(idx, image_suffix))))
            # print("cp {} {}".format(image_path, os.path.join(image_output_folder, "{:05d}.jpg".format(idx))))
            
            # move and rename semantic label
            if not os.path.exists(semantic_mask_output_folder):
                os.makedirs(semantic_mask_output_folder) 

            os.system("cp {} {}".format(label_path, os.path.join(semantic_mask_output_folder, "{:05d}{}".format(idx, label_suffix))))
            # print("cp {} {}".format(label_path, os.path.join(semantic_mask_output_folder, "{:05d}.jpg".format(idx))))
            
            # generate annotations: json -> ins mask -> json
            if not os.path.exists(instance_mask_output_folder):
                os.makedirs(instance_mask_output_folder) 
            os.system("cp {} {}".format(json_path, os.path.join(instance_mask_output_folder, "{:05d}{}".format(idx, json_suffix))))
            
            img_pil = PIL.Image.open(image_path)
            width, height = img_pil.size

            _, instance_map = instance_label(json_path)
            instance_ids = np.unique(instance_map)

            type_map = np.asarray(PIL.Image.open(label_path).convert("L"))
            
            image_name = f"{idx:05d}{image_suffix}"
            images_info.append(
                {
                    "file_name": image_name,
                    "width": width,
                    "height": height,
                    "id": idx
                }
            )

            cls_ins_number = [0] * 10

            for instance_id in instance_ids:
                if instance_id == 0:
                    continue
            
                # category_id
                instance_part = (instance_map == instance_id)
                if np.sum(instance_part) < 4:
                    continue

                category_ids_in_instance = np.unique(type_map[instance_part])
                print(1)
                assert len(category_ids_in_instance) == 1
                category_id = int(category_ids_in_instance[0])
                if category_id > 10 or category_id == 0:
                    raise Exception("Only 10 types")

                # area
                area = int(instance_part.sum())
                if area <= 4:
                    continue

                # bbox
                x1, y1, x2, y2 = mask2box(instance_part)
                w = x2 - x1 + 1
                h = y2 - y1 + 1

                # segmentation (polygon, which means contour)
                segmentation = []
                contours, _ = cv2.findContours((instance_part * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                assert len(category_ids_in_instance) == 1
                contour = contours[0].flatten().tolist()
                segmentation.append(contour)
                if len(segmentation) == 0:
                    raise Exception("Error: no segmentations.")

                # add all label information for one instance
                labels_info.append(
                    {
                        "segmentation": segmentation,  # poly
                        "area": area,  # segmentation area
                        "iscrowd": 0,
                        "image_id": idx,
                        "bbox": [x1, y1, w, h],
                        "category_id": category_id,
                        "id": obj_count
                    },
                )
                obj_count += 1
                cls_ins = cls_ins_number[category_id-1]
                cls_ins_number[category_id-1] = (cls_ins + 1)
            stats_cls_ins.append(cls_ins_number)
            
        # print(stats_cls_ins)
        stats_cls_ins_np = np.asarray(stats_cls_ins).astype(int)
        df = pd.DataFrame(stats_cls_ins_np)
        phase = "train" if "train2017" in image_output_folder else "test"
        json_file_path = instance_mask_output_folder+"/stats_{}_cls_ins.csv".format(str(phase))
        df.to_csv(json_file_path, header=None, index=False) # C is a list of string corresponding to the title of each column of A

        result["images"] = images_info
        result["annotations"] = labels_info
        with open(instance_mask_output_folder + '/instances_{}2017.json'.format(phase), 'w') as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    # train_splits = "../../data/traffic_cambridge/splits/supervised/train.txt"
    
    # train_image_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/FirstFrame_annotate"
    # train_label_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/FirstFrame_annotatation"

    # train_image_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco/train2017"
    # train_semantic_mask_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco/train2017_semantic_masks"
    # train_instance_mask_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco/annotations"

    # main(train_splits, train_image_folder, train_label_folder, train_image_output_folder, train_semantic_mask_output_folder, train_instance_mask_output_folder)
    


    # test_splits = "../../data/traffic_cambridge/splits/supervised/test.txt"

    # test_image_folder =  "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/Fully_annotate"
    # test_label_folder =  "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/Fully_annotatation"

    # test_image_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco/test2017"
    # test_semantic_mask_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco/test2017_semantic_masks"
    # test_instance_mask_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco/annotations"

    # main(test_splits, test_image_folder, test_label_folder, test_image_output_folder, test_semantic_mask_output_folder, test_instance_mask_output_folder)



    train_splits = "../../data/traffic_cambridge/splits/supervised/train.txt"
    
    train_image_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/FirstFrame_annotate"
    train_label_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/FirstFrame_annotatation"

    train_image_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco_test/train2017"
    train_semantic_mask_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco_test/train2017_semantic_masks"
    train_instance_mask_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco_test/labelme_train_dir"

    main(train_splits, train_image_folder, train_label_folder, train_image_output_folder, train_semantic_mask_output_folder, train_instance_mask_output_folder)
    

    test_splits = "../../data/traffic_cambridge/splits/supervised/test.txt"

    test_image_folder =  "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/Fully_annotate"
    test_label_folder =  "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/Fully_annotatation"

    test_image_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco_test/test2017"
    test_semantic_mask_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco_test/test2017_semantic_masks"
    test_instance_mask_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco_test/labelme_test_dir"

    main(test_splits, test_image_folder, test_label_folder, test_image_output_folder, test_semantic_mask_output_folder, test_instance_mask_output_folder)
