import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn as nn
import pyttsx3
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
from google import generativeai as genai  
from packaging import version
import hashlib

# Securely load API key from environment variable
API_KEY = os.environ.get("GOOGLE_GENAI_API_KEY")
if not API_KEY:
if not API_KEY:
    raise RuntimeError("Google Generative AI API key not found in environment variable 'GOOGLE_GENAI_API_KEY'.")
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

def fix_grammar(raw_sentence):
    prompt = (
        f"Fix the grammar of this ASL sentence: \"{raw_sentence}\". "
        f"If it's valid, return only the word. If incomplete, suggest the corrected version in next line. "
        f"Check if it is related to ['Ankit','Anmol','Nishant','Sai','Raghav'] if it not don't mention this.check it resemble some thing like hru means how are you..etc"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return raw_sentence  

# Load label encoder and model
le = joblib.load("label_encoder.pkl")
labels = list(le.classes_)

class ASL_MLP(nn.Module):
    def __init__(self, input_size=126, hidden_size=256, num_classes=len(labels)):
        super(ASL_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

def verify_file_hash(filepath, expected_hash):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASL_MLP(input_size=126, hidden_size=256, num_classes=len(labels)).to(device)

# Secure torch.load by restricting to only loading tensors, with version check and file hash verification
MODEL_PATH = "asl_mlp_model.pth"
MODEL_HASH = os.environ.get("ASL_MLP_MODEL_HASH")  # Set this env var to the expected SHA256 hash
if not MODEL_HASH:
    raise RuntimeError("Expected model hash not set in environment variable 'ASL_MLP_MODEL_HASH'.")
if not verify_file_hash(MODEL_PATH, MODEL_HASH):
    raise RuntimeError(f"Model file hash mismatch for {MODEL_PATH}. Possible tampering detected.")
if version.parse(torch.__version__) < version.parse("2.2.0"):
    raise RuntimeError("PyTorch >= 2.2.0 is required for safe model loading with weights_only=True.")
with open(MODEL_PATH, "rb") as f:
    state_dict = torch.load(f, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Load custom signs if available
custom_file = "custom_signs.pkl"
def verify_file_integrity(file_path, expected_hash_path):
    """Verify the SHA256 hash of a file matches the expected hash."""
    if not os.path.exists(expected_hash_path):
        print(f"[SECURITY WARNING] Hash file {expected_hash_path} not found. Skipping custom sign loading.")
        return False
    with open(expected_hash_path, 'r') as f:
        expected_hash = f.read().strip()
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    file_hash = sha256.hexdigest()
    if file_hash != expected_hash:
        print(f"[SECURITY WARNING] Integrity check failed for {file_path}. File hash does not match expected hash.")
        return False
    return True

custom_file = "custom_signs.pkl"
custom_hash_file = "custom_signs.pkl.sha256"
if os.path.exists(custom_file):
    if verify_file_integrity(custom_file, custom_hash_file):
        custom_signs = joblib.load(custom_file)
        print(f"[INFO] Loaded {len(custom_signs)} custom signs from {custom_file}")
    else:
        custom_signs = {}
        print(f"[INFO] Custom signs file failed integrity check. No custom signs loaded.")
else:
    custom_signs = {}
    print("[INFO] No custom signs found.")
# --- End Secure Custom Signs Loading ---

engine = pyttsx3.init()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

reference_img = cv2.imread(r"e:\sign2.O\src\bothHand\refrence.jpg") 
if reference_img is None:
    print("[ERROR] Could not load reference image.")
    exit()

sentence = ""
prev_label = None
stable_label = None
label_counter = 0
stable_threshold = 8
custom_threshold = 0.88  # cosine similarity threshold for custom signs

def compare_with_custom(embedding):
    """Check similarity of live embedding with stored custom signs"""
    best_match, best_score = None, -1
    for label, vectors in custom_signs.items():
        sims = cosine_similarity([embedding], vectors)
        max_sim = np.max(sims)
        if max_sim > best_score:
            best_match, best_score = label, max_sim
    if best_score >= custom_threshold:
        return best_match, best_score
    return None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    label = None
    landmark_list = [0.0] * 126

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            offset = idx * 63
            for i, lm in enumerate(hand_landmarks.landmark):
                landmark_list[offset + i*3 + 0] = lm.x
                landmark_list[offset + i*3 + 1] = lm.y
                landmark_list[offset + i*3 + 2] = lm.z

        if any(val != 0.0 for val in landmark_list):
            # Default model prediction
            input_tensor = torch.tensor([landmark_list], dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                label = labels[pred_idx]

            # Check for custom sign override
            custom_label, score = compare_with_custom(landmark_list)
            if custom_label:
                print(f"[CUSTOM MATCH] {custom_label} ({score:.2f})")
                label = custom_label

    if label:
        if label == prev_label:
            label_counter += 1
        else:
            label_counter = 0
        if label_counter >= stable_threshold and label != stable_label:
            stable_label = label
            print(f"[PREDICTED] {label}")
            if label == "space":
                sentence += " "
                engine.say("space")
            elif label == "del":
                sentence = sentence[:-1]
                engine.say("delete")
            elif label == "nothing":
                pass
            else:
                sentence += label
                engine.say(label)
            engine.runAndWait()
            label_counter = 0
        prev_label = label

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Sign: {label if label else 'None'}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    ref_resized = cv2.resize(reference_img, (frame.shape[1], frame.shape[0]))
    combined = np.hstack((frame, ref_resized))

    cv2.imshow('Real-Time ASL Translator (MLP + Custom Signs)', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  
        print(f"\n[RAW SENTENCE] {sentence.strip()}")
        corrected_sentence = fix_grammar(sentence.strip())
        print(f"[CORRECTED SENTENCE] {corrected_sentence}")
        engine.say("Final sentence is")
        engine.say(corrected_sentence)
        engine.runAndWait()
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
