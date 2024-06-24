import os
import shutil
import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib
import torch
import clip

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Image paths
original_data_dir = '/Users/michaelrodden/Desktop/original_grape_data'
train_dir = '/Users/michaelrodden/Desktop/original_grape_data/binary_train'
test_dir = '/Users/michaelrodden/Desktop/original_grape_data/binary_test'

# Clear and create directories
def clear_and_create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

clear_and_create_dir(train_dir)
clear_and_create_dir(test_dir)

for category in ['healthy', 'esca']:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

def move_files(src_dir, dst_dir, category):
    for folder in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if category == 'healthy' and folder != 'ESCA':
                    shutil.copy(file_path, os.path.join(dst_dir, 'healthy'))
                elif category == 'esca' and folder == 'ESCA':
                    shutil.copy(file_path, os.path.join(dst_dir, 'esca'))

# Combine images to create healthy and esca paths
move_files(os.path.join(original_data_dir, 'train'), train_dir, 'healthy')
move_files(os.path.join(original_data_dir, 'train'), train_dir, 'esca')
move_files(os.path.join(original_data_dir, 'test'), test_dir, 'healthy')
move_files(os.path.join(original_data_dir, 'test'), test_dir, 'esca')

# Define the function to extract features using CLIP
def extract_clip_features(image, model, preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy()

# Data augmentation generator for synthesizing esca-like images
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=60,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

def extract_augmented_features(directory, model, preprocess, datagen):
    features = []
    labels = []
    for label in ['healthy', 'esca']:
        label_dir = os.path.join(directory, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = Image.open(img_path)
            img = np.array(img)
            img = img.reshape((1,) + img.shape)  # Reshape for augmentation

            # Apply augmentation and extract features
            for batch in datagen.flow(img, batch_size=1):
                img_aug = Image.fromarray((batch[0] * 255).astype(np.uint8))
                feature = extract_clip_features(img_aug, model, preprocess)
                features.append(feature)
                labels.append(0 if label == 'healthy' else 1)
                break  # Only one augmented image per original image
    return np.array(features).reshape(len(features), -1), np.array(labels)

train_features, train_labels = extract_augmented_features(train_dir, model, preprocess, datagen)
test_features, test_labels = extract_augmented_features(test_dir, model, preprocess, datagen)

# Normalize features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Train an Isolation Forest model for anomaly detection
iso_forest = IsolationForest(contamination=0.1)
iso_forest.fit(train_features[train_labels == 0])  # Train only on healthy images

# Predict anomalies in the test set
test_pred = iso_forest.predict(test_features)
test_pred = np.where(test_pred == 1, 0, 1)  # Convert 1 (inlier) to 0 and -1 (outlier) to 1

# Evaluate the model
print(classification_report(test_labels, test_pred, target_names=['healthy', 'esca']))

# Save the Isolation Forest model and scaler
joblib.dump(iso_forest, '/Users/michaelrodden/Desktop/iso_forest_model.joblib')
joblib.dump(scaler, '/Users/michaelrodden/Desktop/scaler.joblib')
