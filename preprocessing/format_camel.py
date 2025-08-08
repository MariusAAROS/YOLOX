import os
import shutil
import json
import random
from copy import deepcopy

NEW_SPLIT = False
REMOVE_SOURCE = False
split_path = "datasets/Camel/split.txt"
exclude = ["train2017", "val2017", "annotations"]
img_typ = "Vis" # IR or Vis
dataset_name = "Camel"

print(os.getcwd())

if NEW_SPLIT or not os.path.exists(split_path):
    train_test_ratio = 0.8
    human_index = [1, 2, 3, 4, 5, 6, 7, 8, 13, 20,
                    21, 25, 26, 27, 28, 30]
    train_index = random.sample(human_index, int(len(human_index) * train_test_ratio))
    val_index = list(set(human_index) - set(train_index))
    with open("datasets/Camel/split.txt", "w") as f:
        f.write("Train: " + ", ".join(map(str, train_index)) + "\n")
        f.write("Val: " + ", ".join(map(str, val_index)) + "\n")
elif not NEW_SPLIT and os.path.exists(split_path):
    with open(split_path, "r") as f:
        lines = f.readlines()
        train_index = list(map(int, lines[0].strip().split(": ")[1].split(", ")))
        val_index = list(map(int, lines[1].strip().split(": ")[1].split(", ")))

data_path = f"datasets/{dataset_name}"
target_path = f"datasets/{dataset_name}_{img_typ}_formatted"

os.makedirs(target_path, exist_ok=True)
os.makedirs(os.path.join(target_path, "train2017"), exist_ok=True)
os.makedirs(os.path.join(target_path, "val2017"), exist_ok=True)
os.makedirs(os.path.join(target_path, "annotations"), exist_ok=True)

DEFAULT_ANNOTATIONS = {
    "info": {
        "description": f"{dataset_name} {img_typ} dataset"
    },
    "licenses": [
        {
            "id": 1,
            "name": "Unknown",
            "url": "Unknown"}
    ],
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

for subset_name, subset_index in zip(["train2017", "val2017"], [train_index, val_index]):
    annotations = deepcopy(DEFAULT_ANNOTATIONS)
    id_counter = 1
    for cdir in dirs_seqs:
        if cdir not in exclude:
            try:
                index = int(cdir.split("_")[1])
            except:
                index = -1
            if index in subset_index:
                print(f"Processing directory: {cdir}")
                cfiles = os.listdir(os.path.join(data_path, cdir))
                cimgs_dir = os.path.join(data_path, cdir, f"{'Visual' if img_typ == 'Vis' else 'IR'}")
                cimgs_files = os.listdir(cimgs_dir)
                
                # metadata
                with open(os.path.join(data_path, cdir, f"Seq{index}-info.txt"), "r") as f:
                    metadata = f.readlines()
                for line in metadata:
                    if f"{'Visual' if img_typ == 'Vis' else 'IR'} Resolution" in line:
                        width, height = line.split(" - ")[1].strip().split("x")
                        width, height = int(width), int(height)
                        break
                with open(os.path.join(data_path, cdir, f"Seq{index}-{img_typ}.txt"), "r") as f:
                    annots = f.readlines()
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
                    id_counter += 1

                for line in annots:
                    if line.strip():
                        parts = line.strip().split("\t")
                        frame_id, object_id, object_class = map(int, parts[:3])
                        x1, y1, w, h = map(float, parts[3:])
                        annotations["annotations"].append({
                            'iscrowd': 0,
                            'image_id': frame_id,
                            'bbox': [x1, y1, w, h],
                            'area': w * h,
                            'category_id': object_class,
                            'id': object_id
                        })
                with open(os.path.join(target_path, "annotations", f"instances_{subset_name}.json"), "w") as f:
                    json.dump(annotations, f, indent=4)

if REMOVE_SOURCE:
    shutil.rmtree(data_path)