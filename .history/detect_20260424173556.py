"""import cv2
import torch
import numpy as np
from collections import deque
from src.model import AccidentTransformer

# ================= LOAD MODEL =================
model = AccidentTransformer()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ================= PARAMETERS =================
frame_buffer = deque(maxlen=16)
prediction_buffer = deque(maxlen=20)

# ================= VIDEO =================
cap = cv2.VideoCapture("acc1.mp4")   # or 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (224,224))
    frame_normalized = frame_resized / 255.0

    frame_buffer.append(frame_normalized)

    show_label = "LOADING..."
    color = (255,255,0)

    # ================= WHEN BUFFER FULL =================
    if len(frame_buffer) == 16:
        clip = np.array(frame_buffer)
        clip = torch.from_numpy(clip).permute(0,3,1,2).float()
        clip = clip.unsqueeze(0)  # (1,16,3,224,224)

        with torch.no_grad():
            output = model(clip)
            prob = torch.softmax(output, dim=1)
            confidence, pred = torch.max(prob, 1)

        # ================= CONFIDENCE FILTER =================
        if confidence.item() < 0.85:
            pred = torch.tensor([1])  # NORMAL

        prediction_buffer.append(pred.item())

        acc_count = prediction_buffer.count(0)

        # ================= FINAL DECISION =================
        if acc_count > 10:
            show_label = "🚨 ACCIDENT DETECTED"
            color = (0,0,255)
        else:
            show_label = "✅ NORMAL"
            color = (0,255,0)

    # ================= DISPLAY =================
    cv2.putText(frame, show_label, (40,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 3)

    cv2.imshow("Accident Detection (Transformer)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

import cv2
import torch
import numpy as np
from collections import deque
from src.model import AccidentTransformer

# ================= LOAD MODEL =================
model = AccidentTransformer()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ================= PARAMETERS =================
frame_buffer = deque(maxlen=16)
prediction_buffer = deque(maxlen=15)

frame_count = 0

# ================= VIDEO =================
cap = cv2.VideoCapture("traffic2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ================= SPEED =================
    if frame_count % 2 != 0:
        cv2.imshow("Accident Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ================= PREPROCESS =================
    resized = cv2.resize(frame, (224,224))
    normalized = resized / 255.0
    frame_buffer.append(normalized)

    show_label = "LOADING..."
    color = (255,255,0)

    # ================= MODEL =================
    if len(frame_buffer) == 16:
        clip = np.array(frame_buffer)
        clip = torch.from_numpy(clip).permute(0,3,1,2).float()
        clip = clip.unsqueeze(0)

        with torch.no_grad():
            output = model(clip)
            prob = torch.softmax(output, dim=1)
            confidence, pred = torch.max(prob, 1)

        prediction_buffer.append(pred.item())
        acc_count = prediction_buffer.count(0)

        # ================= SIMPLE + RELIABLE LOGIC =================
        if acc_count > 6 and confidence.item() > 0.75:
            show_label = "🚨 ACCIDENT DETECTED"
            color = (0,0,255)

        elif acc_count > 3:
            show_label = "⚠️ COLLISION RISK"
            color = (0,165,255)

        else:
            show_label = "✅ NORMAL"
            color = (0,255,0)

    # ================= DISPLAY =================
    cv2.putText(frame, show_label, (40,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 3)

    cv2.imshow("Accident Detection (Final)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()