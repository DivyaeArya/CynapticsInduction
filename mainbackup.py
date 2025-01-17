from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm

# Helper function to create data
def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, target_size=(224, 224))  # Resize to MobileNetV2's input size
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(features.shape[0], 224, 224, 3)  # Reshape all images in one go
    return features

# Set the path to your dataset
TRAIN_DIR = "/kaggle/input/aivsreal/Data/Train"

# Create dataframe
train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

# Extract features and normalize
train_features = extract_features(train['image'])
x_data = train_features / 255.0  # Normalize pixel values to [0, 1]

# Encode labels
le = LabelEncoder()
le.fit(train['label'])
y_data = le.transform(train['label'])
y_data = to_categorical(y_data, num_classes=2)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Load the MobileNetV2 model
base_model = MobileNetV2(weights='/kaggle/input/weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=32,
    epochs=10,
    verbose=1
)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(x_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
