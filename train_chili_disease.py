import os
import glob
import json
import numpy as np # NEW: Import numpy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
# NEW: Import additional callbacks and utility
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# ==== Paths ====
train_dir = r"data/train/chili"
valid_dir = r"data/valid/chili"
# NEW: Let's use the .keras format, which is the modern standard
model_save_path = "chili_disease_model.keras" 

# ==== Hyperparameters ====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# NEW: Split epochs for feature extraction and fine-tuning
INITIAL_EPOCHS = 10 
FINE_TUNE_EPOCHS = 10 # Train for 10 more epochs after unfreezing
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
LEARNING_RATE = 0.0001
# NEW: A very low learning rate for the fine-tuning phase
FINE_TUNE_LR = 0.00001 

# ==== Data Generators ====
# Your data generators are great, no changes needed here
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ==== Save class indices ====
with open("chili_class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

# ==== Build the Model ====
# We are starting fresh, so no need for checkpoint logic for this first run
print("Building model from scratch.")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Start with the base frozen

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x) # A dropout rate of 0.3 is a good starting point
predictions = Dense(len(train_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==== NEW: Calculate Class Weights to handle imbalance ====
class_labels = list(train_gen.class_indices.keys())
class_indices_values = list(train_gen.class_indices.values())
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(zip(np.unique(train_gen.classes), class_weights))
print(f"Calculated Class Weights: {class_weight_dict}")


# ==== Callbacks ====
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5, # Slightly increased patience
    restore_best_weights=True
)
# NEW: Add a learning rate scheduler
lr_scheduler_cb = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, # Reduce LR by a factor of 5
    patience=2, 
    verbose=1
)

# ==== PHASE 1: Feature Extraction Training ====
print("\n--- Starting Phase 1: Feature Extraction ---")
history = model.fit(
    train_gen,
    epochs=INITIAL_EPOCHS,
    validation_data=val_gen,
    callbacks=[earlystop_cb, lr_scheduler_cb],
    class_weight=class_weight_dict # NEW: Use the class weights
)

# ==== PHASE 2: Fine-Tuning ====
print("\n--- Starting Phase 2: Fine-Tuning ---")
# Unfreeze the top layers of the model
base_model.trainable = True

# Let's unfreeze from this layer onwards
fine_tune_at = 100 # Unfreeze the top ~1/3 of MobileNetV2 layers

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile the model with a very low learning rate
model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training
history_fine_tune = model.fit(
    train_gen,
    epochs=TOTAL_EPOCHS, # Train until the total number of epochs
    initial_epoch=history.epoch[-1] + 1, # Start from where the last phase left off
    validation_data=val_gen,
    callbacks=[earlystop_cb, lr_scheduler_cb],
    class_weight=class_weight_dict
)

# ==== Save final model ====
model.save(model_save_path)
print(f"Final model saved as {model_save_path}")