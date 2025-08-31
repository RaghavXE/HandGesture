import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === Config ===
csv_path = "asl_landmark_dataset.csv"   # <-- CSV created from images
model_path = "asl_mlp_model.pth"
encoder_path = "label_encoder.pkl"

# === Load dataset ===
df = pd.read_csv(csv_path)

print(f"[INFO] Loaded dataset with {len(df)} samples and {df.shape[1]-1} features")

# Separate features and labels
X = df.drop(columns=["label"]).values.astype(np.float32)   # 126 features
y = df["label"].values

# Encode labels (letters + words)
le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(le.classes_)

print(f"[INFO] Classes: {list(le.classes_)}")
joblib.dump(le, encoder_path)
print(f"[INFO] Label encoder saved to {encoder_path}")

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# === Dataset & DataLoader ===
train_ds = torch.utils.data.TensorDataset(X_train, y_train)
test_ds = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)

# === Model ===
class ASL_MLP(nn.Module):
    def __init__(self, input_size=126, hidden_size=256, num_classes=num_classes):
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
print(f"[INFO] Using device: {device}")

model = ASL_MLP(input_size=126, hidden_size=256, num_classes=num_classes).to(device)

# === Training setup ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Training loop ===
epochs = 30
print("[INFO] Starting training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# === Evaluation ===
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"[RESULT] Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# === Save model ===
torch.save(model.state_dict(), model_path)
print(f"[✅ DONE] Model saved as {model_path}")
