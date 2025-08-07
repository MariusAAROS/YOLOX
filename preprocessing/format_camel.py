import os
import shutil
import json
import random

train_test_ratio = 0.8
human_index = [1, 2, 3, 4, 5, 6, 7, 8, 13, 20,
                21,24, 25, 26, 27, 28, 29, 30]
train_index = random.sample(human_index, int(len(human_index) * train_test_ratio))
val_index = list(set(human_index) - set(train_index))
exclude = ["train2017", "val2017", "annotations"]
img_typ = "IR" # IR or Vis

dataset_name = "Camel"
data_path = f"datasets/{dataset_name}"
target_path = f"datasets/{dataset_name}_{img_typ}_formatted"

os.makedirs(target_path, exist_ok=True)
os.makedirs(os.path.join(target_path, "train2017"), exist_ok=True)
os.makedirs(os.path.join(target_path, "val2017"), exist_ok=True)
os.makedirs(os.path.join(target_path, "annotations"), exist_ok=True)

annotations = {
    "info": {
        "description": f"{dataset_name} {img_typ} dataset"
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "human",
            "supercategory": "person"
        }
    ],
    "type": "instances"
}

dirs_seqs = os.listdir(data_path)

id_counter = 1
for subset_name, subset_index in zip(["train2017", "val2017"], [train_index, val_index]):
    for cdir in dirs_seqs:
        if cdir not in exclude:
            try:
                index = int(cdir.split("_")[1])
            except:
                index = -1
            if index in subset_index:
                print(f"Processing directory: {cdir}")
                cfiles = os.listdir(os.path.join(data_path, cdir))
                cimgs_dir = os.path.join(data_path, cdir, f"{"Visual" if img_typ == "Vis" else "IR"}")
                cimgs_files = os.listdir(cimgs_dir)
                
                # metadata
                with open(os.path.join(data_path, cdir, f"Seq{index}-info.txt"), "r") as f:
                    metadata = f.readlines()
                for line in metadata:
                    if f'{"Visual" if img_typ == "Vis" else "IR"} Resolution' in line:
                        width, height = line.split(" - ")[1].strip().split("x")
                        width, height = int(width), int(height)

                for cimg in cimgs_files:
                    #copy image to target directory
                    new_name = f"{index}_{cimg}"
                    src = os.path.join(cimgs_dir, cimg)
                    dst = os.path.join(target_path, subset_name, new_name)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy(src, dst)
                    #add image to annotations
                    annotations["images"].append({
                        "id": id_counter,
                        "file_name": new_name,
                        "width": width,
                        "height": height,
                    })

                    #TO DO - add the annotations 
                    annotations["annotations"].append({
                        'image_id': id_counter,
                        'bbox': [214.14976, 41.29011799999999, 348.25984, 243.78],
                        'category_id': 1,
                        'id': id_counter
                    })

                    id_counter += 1
                with open(os.path.join(target_path, "annotations", f"instances_{subset_name}.json"), "w") as f:
                    json.dump(annotations, f, indent=4)
                break