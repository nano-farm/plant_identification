import os
import glob
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# ==== Paths ====
train_dir = r"data/train/tomato"
valid_dir = r"data/valid/tomato"
checkpoint_dir = "tomato_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# ==== Hyperparameters ====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

# ==== Data Generators ====
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
with open("tomato_class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

# ==== Load latest checkpoint if available ====
latest_checkpoint = None
checkpoints = glob.glob(os.path.join(checkpoint_dir, "epoch_*.h5"))
if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)

if latest_checkpoint:
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    model = tf.keras.models.load_model(latest_checkpoint)
    initial_epoch = int(os.path.basename(latest_checkpoint).split("_")[1].split(".")[0])
else:
    print("No checkpoint found. Starting from scratch.")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(len(train_gen.class_indices), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    initial_epoch = 0

# ==== Callbacks ====
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "epoch_{epoch:02d}.h5"),
    save_best_only=False
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ==== Training ====
model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint_cb, earlystop_cb],
    initial_epoch=initial_epoch
)

# ==== Save final model ====
model.save("tomato_disease_model.h5")
print("Final model saved as tomato_disease_model.h5")
