import cv2
import mediapipe as mp
import os
import pandas as pd

# === Mediapipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,  # allow both hands
    min_detection_confidence=0.5
)

# === Dataset paths ===
dataset_dir = "E:/sign2.O/asl_alphabet_train"   # <-- put all your images here (class subfolders)
output_csv = "asl_landmark_dataset.csv"

data = []

# === Loop over folders (each folder = label) ===
for label in os.listdir(dataset_dir):
    label_path = os.path.join(dataset_dir, label)
    if not os.path.isdir(label_path):
        continue

    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to load {img_path}")
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Default: 42 coords per hand (x,y,z * 21)
        coords = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

        # If only one hand detected, pad with zeros
        if len(results.multi_hand_landmarks or []) == 1:
            coords.extend([0.0] * 63)

        # Skip if no hand detected
        if len(coords) != 126:
            continue

        data.append(coords + [label])

# === Save to CSV ===
columns = []
for h in ["H1", "H2"]:
    for i in range(21):
        for c in ["x", "y", "z"]:
            columns.append(f"{h}_{i}_{c}")
columns.append("label")

df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)
print(f"[✅] Dataset saved at {output_csv} with {len(df)} samples")
