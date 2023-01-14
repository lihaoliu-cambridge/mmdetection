import os

# train image -> FirstFrame_annotate/ .jpg
# train label -> FirstFrame_annotatation/ .png

# test image  -> Fully_annotate/ .jpg
# test label  -> Fully__annotatation/ .png

image_suffix = ".jpg"
label_suffix = ".png"
json_suffix = ".json"





def main(splits, image_folder, image_output_folder):
    with open(splits) as file:
        lines = [line.rstrip() for line in file]
        
        for idx, file_path in enumerate(lines):
            if idx <= 2307:
                continue
            print(idx)

            image_path = os.path.join(image_folder, file_path+image_suffix)
            assert(os.path.exists(image_path))

            # move and rename image
            if not os.path.exists(image_output_folder):
                os.makedirs(image_output_folder) 
            
            # print("cp {} {}".format(image_path, os.path.join(image_output_folder, "{:05d}.jpg".format(idx))))
            os.system("cp {} {}".format(image_path, os.path.join(image_output_folder, "{:05d}{}".format(idx, image_suffix))))

            # if idx >= 2310:
            #     break


if __name__ == "__main__":
    train_splits = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/tools/preprocess/3_semi_supervised/all/train_label_all.txt"
    train_image_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/traffic_cambridge/FirstFrame_annotate"
    train_image_output_folder = "/home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/data/coco/train2017"

    main(train_splits, train_image_folder, train_image_output_folder)
    
