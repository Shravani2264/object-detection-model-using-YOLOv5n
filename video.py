import cv2
import torch

# Load your custom YOLOv5 model
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="C:/Users/Admin/Desktop/object detection/yolov5/person_yolo/custom_model2/weights/best.pt",
)

# Start webcam
cap = cv2.VideoCapture(0)

# Detection thresholds and filters
CONF_THRESHOLD = 0.4  # Raised to reduce false detections
ASPECT_RATIO_RANGE = (0.4, 1.3)  # width / height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]
    people_count = 0

    for *box, conf, cls in detections:
        if int(cls) == 0 and conf > CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            area = w * h
            aspect_ratio = w / (h + 1e-6)

            # Smart filters

            if not (ASPECT_RATIO_RANGE[0] <= aspect_ratio <= ASPECT_RATIO_RANGE[1]):
                continue
            if aspect_ratio > 1.2 and h < 250:
                continue
            if x2 < 200 and y2 < 200:  # Filter top-left false positive zone
                continue

            # Valid person detected
            people_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display count
    cv2.putText(frame, f"People: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show output
    cv2.imshow("YOLOv5 Person Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
