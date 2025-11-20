import os
import tensorflow as tf
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import cv2
import numpy as np

# Check available devices
for device in tf.config.list_physical_devices():
    print(f"{device.name}")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Handle truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def plot_loss(history):
    plt.figure(figsize=(20, 10))
    sns.set_style('whitegrid')
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.show()

def plot_acc(history):
    plt.figure(figsize=(20, 10))
    sns.set_style('whitegrid')
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()

# Paths and settings
train_path = "Dataset/train"
valid_path = "Dataset/valid"
test_path = "Dataset/test"

im_size = 224
image_resize = (im_size, im_size, 3)
batch_size = 100
num_classes = 2

cam_model = load_model('saved_model/custom_model.h5')
print(cam_model.summary())
modela = load_model('saved_model/custom_best_model.h5')
print(modela.summary())

model = load_model('saved_model/custom_model.h5')
print(model.summary())




def show_cam(image_value, features, results):
    features_for_img = features[0]
    prediction = results[0]

    class_activation_weights = model.layers[-1].get_weights()[0][:, 1]
    class_activation_features = sp.ndimage.zoom(features_for_img, (im_size / features_for_img.shape[0], im_size / features_for_img.shape[1], 1), order=2)
    cam_output = np.dot(class_activation_features, class_activation_weights)

    # Visualize the results
    plt.figure(figsize=(12, 12))
    plt.imshow(cam_output, cmap='jet', alpha=0.5)
    plt.imshow(tf.squeeze(image_value), alpha=0.5)
    plt.title('Class Activation Map')
    plt.figtext(.5, .05, f"No Wildfire Probability: {results[0][0] * 100:.2f}%\nWildfire Probability: {results[0][1] * 100:.2f}%", ha="center", fontsize=12, bbox={"facecolor": "green", "alpha": 0.5, "pad": 3})
    plt.colorbar()
    plt.show()


def convert_and_classify(image_path):
    # Load the model
    cam_model = load_model('saved_model/custom_model.h5')

    # Create a model for extracting features (intermediate layer output)
    intermediate_layer_model = Model(inputs=cam_model.input, outputs=cam_model.layers[-3].output)

    # Read and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (im_size, im_size)) / 255.0
    tensor_image = np.expand_dims(img, axis=0)

    # Get features and predictions
    features = intermediate_layer_model.predict(tensor_image)
    results = cam_model.predict(tensor_image)

    # Generate and display the CAM
    show_cam(tensor_image, features, results)


convert_and_classify('Dataset/test/nowildfire/-73.463612,45.570149.jpg')

convert_and_classify('Dataset/test/wildfire/-66.1249,48.1247.jpg')
 
path = 'Dataset/test/wildfire/-75.91405,46.57195.jpg'
img = cv2.imread(path)
print('CAM Model Prediction:\n')
convert_and_classify(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (im_size, im_size))
x = np.expand_dims(img, axis=0)
classes = model.predict(x)
print('Custom Model Prediction:\n')
print("No Wildfire Probability: %", round(classes[0][0] * 100, 2))
print("Wildfire Probability: %", round(classes[0][1] * 100, 2))
