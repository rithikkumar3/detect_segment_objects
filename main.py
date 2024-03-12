# from ultralytics import YOLO

# model_seg = YOLO("models/yolov8x-seg.pt")
# results_seg =  model_seg(source = "april_6_1_rect_left_0.png", show=True, conf =0.2, save=True)

from ultralytics import YOLO
import random
import cv2
import numpy as np

model = YOLO("models/yolov8x-seg.pt")
img = cv2.imread("collection_rect_left_0.png")

# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

conf = 0.5

results = model.predict(img, conf=conf)
colors = [random.choices(range(256), k=3) for _ in classes_ids]
print(results)
for result in results:
    for mask, box in zip(result.masks.xy, result.boxes):
        points = np.int32([mask])
        cv2.polylines(img, points, True, (255, 0, 0), 1)
        color_number = classes_ids.index(int(box.cls[0]))
        cv2.fillPoly(img, points, colors[color_number])

cv2.imshow("Image", img)
cv2.imwrite('processed_image.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()