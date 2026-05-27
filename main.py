import cv2
from ultralytics import YOLO
model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = box.conf[0]
            if conf > 0.7:
                label_text = f"{label} {conf:.2f}"
                if label in ['Sleeping', 'Usingmobile']:
                    cv2.imwrite(f'PROOF_{label}.jpg', frame)
                    print(f"НАРУШЕНИЕ! Зафиксировано: {label}")
    cv2.imshow('Student Monitor', results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()