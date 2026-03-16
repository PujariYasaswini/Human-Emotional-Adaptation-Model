import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential

# 1. SETTINGS
DATASET_PATH = r"C:\Users\My Dell\Downloads\HEAM\dataset\train"
MODEL_SAVE_PATH = r"C:\Users\My Dell\Downloads\HEAM\emotion_models\best-model-1.h5"
IMG_SIZE = (224, 224) # MobileNetV2 standard size
BATCH_SIZE = 32

# 2. DYNAMIC CLASS DETECTION
# This prevents the "logits and labels must be broadcastable" error
class_names = sorted(os.listdir(DATASET_PATH))
num_classes = len(class_names)
print(f"✅ Found {num_classes} classes: {class_names}")

# 3. DATA PREPARATION (Augmentation)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% for testing
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 4. MODEL ARCHITECTURE (Transfer Learning)
# Using 'imagenet' weights gives us a pre-trained "brain"
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the heavy lifting layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),  # Prevents the model from "over-memorizing"
    Dense(num_classes, activation='softmax') # Dynamic output layer
])

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# 5. TRAINING
print("🚀 Starting Training...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10, 
    verbose=1
)

# 6. SAVE
# Create directory if it doesn't exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)

print(f"✨ Success! New high-accuracy model saved to: {MODEL_SAVE_PATH}")
print(f"Class Mapping for HEAM: {train_data.class_indices}")