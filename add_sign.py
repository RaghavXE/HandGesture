import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from tqdm import tqdm

custom_file = "custom_signs.pkl"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Load existing custom signs if available
if os.path.exists(custom_file):
    custom_signs = joblib.load(custom_file)
else:
    custom_signs = {}

def extract_landmarks(image):
    """Extract hand landmarks (126 features) using Mediapipe."""
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmark_list = [0.0] * 126
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            offset = idx * 63
            for i, lm in enumerate(hand_landmarks.landmark):
                landmark_list[offset + i*3 + 0] = lm.x
                landmark_list[offset + i*3 + 1] = lm.y
                landmark_list[offset + i*3 + 2] = lm.z
    return np.array(landmark_list, dtype=np.float32)

def augment(landmark_vector):
    """Generate augmented versions of a landmark vector."""
    augmented = []
    for _ in range(50):  # generate 50 variations
        noise = np.random.normal(0, 0.01, landmark_vector.shape)
        aug = landmark_vector + noise
        augmented.append(aug)
    return augmented

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam not accessible.")
    exit()

print("[INFO] Press SPACE to capture a snapshot of your sign. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Press SPACE to capture sign, Q to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Add Custom Sign", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # SPACE key pressed
        landmarks = extract_landmarks(frame)
        if np.all(landmarks == 0):
            print("[WARN] No hands detected, try again.")
            continue

        label = input("Enter label for this sign (e.g., Hello, Ankit): ").strip()
        if not label:
            print("[WARN] Empty label, skipping.")
            continue

        dataset = [landmarks]
        for _ in tqdm(range(20), desc="Augmenting"):
            dataset.extend(augment(landmarks))

        if label in custom_signs:
            custom_signs[label].extend(dataset)
        else:
            custom_signs[label] = dataset

        joblib.dump(custom_signs, custom_file)
        print(f"[✅ SAVED] Added {len(dataset)} samples for sign '{label}'")

cap.release()
cv2.destroyAllWindows()
hands.close()


# import joblib
# import os

# custom_file = "custom_signs.pkl"

# if os.path.exists(custom_file):
#     # Reset with an empty dict
#     joblib.dump({}, custom_file)
#     print(f"[✅ DONE] Cleared all custom signs from {custom_file}")
# else:
#     print(f"[INFO] {custom_file} does not exist, nothing to clear.")
