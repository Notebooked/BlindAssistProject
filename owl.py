import torch
import cv2
from PIL import Image
import numpy as np

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text_queries = ["person", "bicycle", "dog", "cell phone", "airplane", "zebra", "remote", "coffee mug", "fire hydrant"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

frame_count = 0
results = None 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 5 == 0:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=text_queries, images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.Tensor([img.size[::-1]]).to(device)
        results = processor.post_process(outputs=outputs, target_sizes=target_sizes)[0]

    if results is not None:
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        for box, score, label in zip(boxes, scores, labels):
            if score < 0.3:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_text = f"{text_queries[label]} {score:.2f}"

            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)  # filled rectangle
            cv2.putText(frame, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow("OWL-ViT Open-Vocabulary Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
