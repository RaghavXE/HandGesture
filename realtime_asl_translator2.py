

import torch
import numpy as np
import mediapipe as mp
import torch.nn as nn
from flask import Flask, request, jsonify
from PIL import Image
import io
from flask_cors import CORS
import os
import numpy as np

class ASL_MLP(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=29):
        super(ASL_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
# model = ASL_MLP()
# # model.load_state_dict(torch.load('asl_mlp_model.pth', map_location=device))

# model_path = os.path.join(os.path.dirname(__file__), "asl_mlp_model.pth")
# model.load_state_dict(torch.load(model_path))
# model.to(device)
# model.eval()

model = ASL_MLP()
model_path = os.path.join(os.path.dirname(__file__), "asl_mlp_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))  # ✅ force map
model.to(device)   # put model on chosen device
model.eval()

# === Labels ===
labels = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
    'O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space'
]


# === Mediapipe hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# === Flask app ===
app = Flask(__name__)
CORS(app)

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "frame" not in request.files:
#         return jsonify({"error": "No frame uploaded"}), 400
    
#     file = request.files["frame"]
#     image = Image.open(io.BytesIO(file.read())).convert("RGB")
#     frame = np.array(image)

#     results = hands.process(frame)
#     label = None
    

#     # landmark_list = [0.0] * 126
#     # if results.multi_hand_landmarks:
#     #     for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
#     #         offset = idx * 63
#     #         for i, lm in enumerate(hand_landmarks.landmark):
#     #             landmark_list[offset + i*3 + 0] = lm.x
#     #             landmark_list[offset + i*3 + 1] = lm.y
#     #             landmark_list[offset + i*3 + 2] = lm.z
#     landmark_list = []
#     if results.multi_hand_landmarks:
#         print("[INFO] Hand detected")
#         hand_landmarks = results.multi_hand_landmarks[0]  # only first detected hand
#         for lm in hand_landmarks.landmark:
#             landmark_list.extend([lm.x, lm.y, lm.z])

#         if len(landmark_list) == 63:
#             input_tensor = torch.tensor([landmark_list], dtype=torch.float32).to(device)
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 pred_idx = torch.argmax(output, dim=1).item()
#                 label = labels[pred_idx]


#             if any(val != 0.0 for val in landmark_list):
#                 input_tensor = torch.tensor([landmark_list], dtype=torch.float32).to(device)
#                 with torch.no_grad():
#                     output = model(input_tensor)
#                     pred_idx = torch.argmax(output, dim=1).item()
#                     label = labels[pred_idx]
#     else:
#         print("[INFO] No hand detected")
#     return jsonify({"prediction": label if label else "None"})



@app.route("/predict", methods=["POST"])
def predict():
    if "frame" not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400
    
    file = request.files["frame"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    frame = np.array(image)

    # Ensure correct format for Mediapipe (RGB, uint8)
    if frame is None or frame.size == 0:
        print("[ERROR] Empty frame received")
        return jsonify({"prediction": "None"})

    try:
        results = hands.process(frame)  # static_image_mode=True is critical
    except Exception as e:
        print("[ERROR] Mediapipe failed:", e)
        return jsonify({"prediction": "None"})
    
    label = None
    if results.multi_hand_landmarks:
        print("[INFO] Hand detected")
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_list = []
        for lm in hand_landmarks.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])

        if len(landmark_list) == 63:
            input_tensor = torch.tensor([landmark_list], dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                label = labels[pred_idx]
    else:
        print("[INFO] No hand detected")

    return jsonify({"prediction": label if label else "None"})



if __name__ == "__main__":
    app.run(port=5000, debug=True)
