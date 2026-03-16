

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "best-model.h5"   # path to your trained .h5 model
TEST_DIR = "test"            # folder with emotion subfolders
IMG_SIZE = 48

# 7 emotion classes (FER-2013 compatible)
LABEL_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

LABEL_NAMES = list(LABEL_MAP.keys())

# -----------------------------------------
# 2. LOAD MODEL
# -----------------------------------------
print("[INFO] Loading trained CNN model...")
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully")
print("[INFO] Model output shape:", model.output_shape)

# -----------------------------------------
# 3. LOAD TEST DATA
# -----------------------------------------
print("[INFO] Loading test dataset...")

X_test = []
y_test = []

for label in LABEL_MAP:
    folder_path = os.path.join(TEST_DIR, label)
    if not os.path.exists(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X_test.append(img)
        y_test.append(LABEL_MAP[label])

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)

print(f"[INFO] Total test samples loaded: {len(X_test)}")
print("[INFO] Unique labels in test set:", np.unique(y_test))

# -----------------------------------------
# 4. MODEL PREDICTION
# -----------------------------------------
print("[INFO] Performing model inference...")
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# -----------------------------------------
# 5. EVALUATION METRICS
# -----------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n========== EVALUATION METRICS ==========")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        y_test,
        y_pred,
        labels=list(LABEL_MAP.values()),
        target_names=LABEL_NAMES
    )
)

# -----------------------------------------
# 6. CONFUSION MATRIX + HEATMAP
# -----------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=list(LABEL_MAP.values()))

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABEL_NAMES,
    yticklabels=LABEL_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Facial Emotion Recognition")
plt.tight_layout()

# Save image for report
plt.savefig("confusion_matrix_heatmap.png", dpi=300)
plt.show()

print("[INFO] Confusion matrix saved as confusion_matrix_heatmap.png")

# -----------------------------------------
# 7. OPTIONAL: EMOTION PROBABILITY HEATMAP
# -----------------------------------------
sample_probs = y_pred_prob[:10]

plt.figure(figsize=(10,4))
sns.heatmap(
    sample_probs,
    annot=True,
    cmap="YlGnBu",
    xticklabels=LABEL_NAMES,
    yticklabels=[f"Sample {i}" for i in range(10)]
)

plt.xlabel("Emotion Class")
plt.ylabel("Test Samples")
plt.title("Emotion Probability Heatmap")
plt.tight_layout()
plt.show()

print("[INFO] Evaluation completed successfully")
