<<<<<<< HEAD
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# ✅ Path to your dataset
DATA_DIR = r"C:\Users\NIBRAS\Desktop\FarmGuard_AI\farmguard_dataset\cassava"

# ✅ Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ✅ Load training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ✅ Get class names
class_names = train_ds.class_names
print("Detected Classes:", class_names)

# ✅ Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ✅ Load pre-trained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

# ✅ Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

# ✅ Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Summary
model.summary()

# ✅ Train model
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ✅ Save model to folder
EXPORT_DIR = "cassava_model"
if os.path.exists(EXPORT_DIR):
    import shutil
    shutil.rmtree(EXPORT_DIR)
model.save(EXPORT_DIR)

# ✅ Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(EXPORT_DIR)
tflite_model = converter.convert()

# ✅ Save the TFLite model
with open("cassava_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ cassava_model.tflite saved successfully.")
=======
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# ✅ Path to your dataset
DATA_DIR = r"C:\Users\NIBRAS\Desktop\FarmGuard_AI\farmguard_dataset\cassava"

# ✅ Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ✅ Load training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ✅ Get class names
class_names = train_ds.class_names
print("Detected Classes:", class_names)

# ✅ Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ✅ Load pre-trained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

# ✅ Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

# ✅ Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Summary
model.summary()

# ✅ Train model
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ✅ Save model to folder
EXPORT_DIR = "cassava_model"
if os.path.exists(EXPORT_DIR):
    import shutil
    shutil.rmtree(EXPORT_DIR)
model.save(EXPORT_DIR)

# ✅ Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(EXPORT_DIR)
tflite_model = converter.convert()

# ✅ Save the TFLite model
with open("cassava_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ cassava_model.tflite saved successfully.")
>>>>>>> 191e879d08ad402996ad27385b5329ba3fdaf72f
