import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# SETTINGS
# ==========================
IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 15

# ==========================
# DATASET PATH
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "dataset")
)

print("Dataset Path =", DATASET_PATH)
print("Exists =", os.path.exists(DATASET_PATH))

print("\nFolders found:")
for item in os.listdir(DATASET_PATH):
    full_path = os.path.join(DATASET_PATH, item)
    if os.path.isdir(full_path):
        print(item)

# ==========================
# DATA GENERATOR
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# ==========================
# CLASS INFO
# ==========================
print("\nClass Indices:")
print(train_generator.class_indices)

num_classes = len(train_generator.class_indices)

print("Number of Classes:", num_classes)

# ==========================
# MODEL
# ==========================
model = Sequential([
    Input(shape=(256, 256, 3)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

# ==========================
# COMPILE
# ==========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

print("\nOutput Shape:", model.output_shape)

# ==========================
# TRAIN
# ==========================
history = model.fit(
    train_generator,
    epochs=EPOCHS
)

# ==========================
# SAVE MODEL
# ==========================
MODEL_PATH = os.path.join(BASE_DIR, "tomato_model.h5")

model.save(MODEL_PATH)

print("\nModel saved successfully!")
print("Saved to:", MODEL_PATH)
print("Final Output Shape:", model.output_shape)