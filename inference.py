import cv2
import torch


# define a video capture object
vid = cv2.VideoCapture(0)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

imageCount = 9
while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    height, width, _ = frame.shape
    cy, cx, size = height // 2, width // 2, min(height, width)
    frame = frame[cy-size//2:cy+size//2, cx-size//2:cx+size//2, :]

    cv2.imshow('frame', frame)
    displayFrame = frame[..., ::-1]
    # results = model(displayFrame, size=416)  # includes NMS
    # results.print()

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    if cv2.waitKey(1) & 0xFF == ord('p'):
        fileName = f"capture_{imageCount:03}.png"
        cv2.imwrite(fileName, frame)
        print(f"Captured {fileName}")
        imageCount += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# # After the loop release the cap object
# vid.release()
# exit()

# Model

# # Images
# # OpenCV image (BGR to RGB)
# im1 = cv2.imread('data/pytorch/v2/images/val/scene_0566.jpg')[..., ::-1]
# # OpenCV image (BGR to RGB)
# im2 = cv2.imread('data/pytorch/v2/images/val/scene_0567.jpg')[..., ::-1]
# imgs = [im1, im2]  # batch of images

# imgs = list(cv2.imread(f"data/pytorch/v2/images/val/scene_0{a}.jpg")[..., ::-1]
#             for a in range(521, 572))

# # Inference
# results = model(imgs[0:1], size=416)  # includes NMS
# results.print()
# results = model(imgs[1:2], size=416)  # includes NMS
# results.print()
# results = model(imgs[2:3], size=416)  # includes NMS
# results.print()


# Results
# results.print()
# results.save()  # or .show()
# results.show()

# results.xyxy[0]  # im1 predictions (tensor)
# results.pandas().xyxy[0]  # im1 predictions (pandas)
