import cv2
import os

input_root = "dataset/data"
output_root = "dataset_clips"

def process_split(split):
    for cls in ["accident", "normal"]:
        input_path = os.path.join(input_root, split, cls)
        output_path = os.path.join(output_root, split, cls)

        os.makedirs(output_path, exist_ok=True)

        clip_id = 0

        for file in os.listdir(input_path):
            path = os.path.join(input_path, file)

            # If image dataset
            if file.endswith((".jpg", ".png")):
                img = cv2.imread(path)
                img = cv2.resize(img, (224,224))

                clip_folder = os.path.join(output_path, f"clip_{clip_id}")
                os.makedirs(clip_folder, exist_ok=True)

                for i in range(16):
                    cv2.imwrite(f"{clip_folder}/frame_{i}.jpg", img)

                clip_id += 1

            # If video dataset
            else:
                cap = cv2.VideoCapture(path)
                frame_buffer = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (224,224))
                    frame_buffer.append(frame)

                    if len(frame_buffer) == 16:
                        clip_folder = os.path.join(output_path, f"clip_{clip_id}")
                        os.makedirs(clip_folder, exist_ok=True)

                        for i, f in enumerate(frame_buffer):
                            cv2.imwrite(f"{clip_folder}/frame_{i}.jpg", f)

                        frame_buffer = []
                        clip_id += 1

                cap.release()

for split in ["train", "val", "test"]:
    process_split(split)

print("Clips created successfully!")