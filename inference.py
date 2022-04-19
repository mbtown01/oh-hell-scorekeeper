import cv2
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='../yolov5/runs/train/exp8/weights/best.pt')

# # Images
# # OpenCV image (BGR to RGB)
# im1 = cv2.imread('data/pytorch/v2/images/val/scene_0566.jpg')[..., ::-1]
# # OpenCV image (BGR to RGB)
# im2 = cv2.imread('data/pytorch/v2/images/val/scene_0567.jpg')[..., ::-1]
# imgs = [im1, im2]  # batch of images

imgs = list(cv2.imread(f"data/pytorch/v2/images/val/scene_0{a}.jpg")[..., ::-1]
            for a in range(521, 572))

# Inference
results = model(imgs[0:1], size=416)  # includes NMS
results.print()
results = model(imgs[1:2], size=416)  # includes NMS
results.print()
results = model(imgs[2:3], size=416)  # includes NMS
results.print()


# Results
# results.print()
# results.save()  # or .show()
# results.show()

results.xyxy[0]  # im1 predictions (tensor)
results.pandas().xyxy[0]  # im1 predictions (pandas)
