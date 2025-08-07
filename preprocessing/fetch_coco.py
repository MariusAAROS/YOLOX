import fiftyone.zoo as foz
import fiftyone as fo


dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["person", "car", "truck", "traffic light"],
    max_samples=100,
)

print("ok")
export_dir = "datasets/new_coco"
dataset_type = fo.types.COCODetectionDataset
label_field = "ground_truth"

dataset.export(
    export_dir=export_dir,
    dataset_type=dataset_type,
    label_field=label_field,
)