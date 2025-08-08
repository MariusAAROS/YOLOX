import sys
sys.path.append('..')

from yolox.models.yolox import YOLOX
from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.models.yolo_head import YOLOXHead
import torch

act = "silu"
size = "s" # s or m
if size == "s":
    depth = 0.33
    width = 0.50
elif size == "m":
    depth = 0.67
    width = 0.75
in_channels = [256, 512, 1024]
num_classes = 1  # assuming original

# Reconstruct original backbone and head to load weights
backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
checkpoint = torch.load(f"yolox/weights/yolox_{size}.pth", map_location="cpu")

print(checkpoint)
# backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
# head = YOLOXHead(80, width, in_channels=in_channels)
# model = YOLOX(backbone, head)

# # Load pretrained weights
# model.load_state_dict(torch.load(f"yolox/weights/yolox_{size}.pth", map_location="cpu"))
# backbone = model.backbone
# new_head = YOLOXHead(new_num_classes, width, in_channels=in_channels)
# new_model = YOLOX(backbone, new_head)

# save_path = f"yolox/weights/yolo_{size}_custom.pth"
# torch.save(model.state_dict(), save_path)
# pprint("model_saved to: "+save_path)