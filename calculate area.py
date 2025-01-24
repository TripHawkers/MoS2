import cv2
from ultralytics import YOLO

model = YOLO('weight.pt')

input_image_path = 'xxx.png'
output_image_path = 'xxx.jpg'
img = cv2.imread(input_image_path)
results = model(img)
img_height, img_width, _ = img.shape
total_area = img_height * img_width
area_by_class = {}

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])

        area = (x2 - x1) * (y2 - y1)
        if cls not in area_by_class:
            area_by_class[cls] = 0
        area_by_class[cls] += area

        label = f"{model.names[cls]} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

area_percentages = {cls: (area / total_area) * 100 for cls, area in area_by_class.items()}

for cls, percentage in area_percentages.items():
    print(f"class {model.names[cls]} Proportion of total area: {percentage:.2f}%")

cv2.imwrite(output_image_path, img)