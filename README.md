# ✋ HandGesture — Real-time Sign Language Translator  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)  
![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange?logo=pytorch)  
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)  
![License](https://img.shields.io/badge/License-Educational-lightgrey)  

---



## 📖 Overview  
HandGesture is a **real-time American Sign Language (ASL) translator** built using **Computer Vision (OpenCV + Mediapipe)** and a **Deep Learning model (PyTorch)**.  
It allows users to **translate ASL gestures into text instantly**, bridging the gap between the deaf community and non-signers.  

The project also includes a **simple web interface** for demonstrations, making it accessible and easy to use.  

---



## ✨ Features  
- **Real-time detection** of ASL hand gestures using a webcam  
- **Pre-trained MLP model** (`asl_mlp_model.pth`) for gesture classification  
- **Web-based interface** (`index.html`, `run.html`) for interactive usage  
- **Text-to-speech support** for vocalizing translated signs  
- Includes **dataset folder (`asl_alphabet_train/`)** for retraining  

---



## 🛠 Tech Stack  
- **Programming Language**: Python 3.10+  
- **Libraries & Frameworks**:  
  - [PyTorch](https://pytorch.org/) — Deep Learning  
  - [OpenCV](https://opencv.org/) — Computer Vision  
  - [Mediapipe](https://developers.google.com/mediapipe) — Hand Landmark Detection  
  - [NumPy](https://numpy.org/) — Matrix Computations  
- **Frontend**: HTML, CSS, JavaScript  

---



## ⚙️ Installation  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/RaghavXE/HandGesture.git
   cd HandGesture

2. **Install dependencies**
   ```bash
   pip install torch opencv-python mediapipe numpy


3. **Ensure you have a webcam connected.**

  ▶️ Usage
  🔹 Run Real-time Translator
      
      python realtime_asl_translator2.py
      
  🔹 Launch Web Interface
      Simply open the files in your browser:
        • index.html → Main interface
        • run.html → Gesture translation demo
  
  🔹 (Optional) Train Model from Scratch
      
      python New.py
      Dataset used: asl_alphabet_train/
  

## 📂 Project Structure

      HandGesture/
      │── New.py                     # Training script
      │── realtime_asl_translator2.py # Real-time ASL translator
      │── asl_mlp_model.pth           # Pre-trained PyTorch model
      │── asl_alphabet_train/         # Training dataset
      │── index.html                  # Web UI (main)
      │── run.html                    # Gesture demo page
      │── script.js                   # Frontend logic
      │── style.css                   # Frontend styling
      │── logo.png                    # Project logo
      │── ankit.jpg                   # Team member photo
      │── anmol.jpg                   # Team member photo
      │── raghav.jpg                  # Team member photo
      │── srisai.jpg                  # Team member photo
      │── README.md                   # Documentation

---
## 📸 Demo

  🎥 Real-time Translation Example:
    
  
---

## 👥 Contributors
    
  🔹Raghav — Backend Developer 
  
  🔹Ankit — Model Training & Testing
  
  🔹Anmol — Dataset Development
  
  🔹Sri Sai — Frontend Developer
  
  🔹Nishant — UI/UX & Integration
  

## 📜 License

  
  This project is released for educational purposes.
  Feel free to fork, modify, and learn from it with proper attribution.
