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

# Data generators
data_generator = ImageDataGenerator(rescale=1./255.)

train_generator = data_generator.flow_from_directory(
    train_path,
    batch_size=batch_size,
    target_size=(im_size, im_size),
    class_mode='categorical'
)

valid_generator = data_generator.flow_from_directory(
    valid_path,
    batch_size=batch_size,
    target_size=(im_size, im_size),
    class_mode='categorical'
)

# Class mapping
class_mapping = train_generator.class_indices
class_names = list(class_mapping.keys())
print("Class names:", class_names)

# Class distribution
def plot_class_distribution(generator, title):
    labels = generator.classes
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    data = pd.DataFrame({'Class': [class_names[i] for i in unique_labels], 'Count': label_counts})
    custom_palette = {'nowildfire': 'skyblue', 'wildfire': 'orange'}

    plt.figure(figsize=(20, 7))
    sns.barplot(x='Class', y='Count', data=data, palette=custom_palette)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

plot_class_distribution(train_generator, 'Class Distribution in Training Set')
plot_class_distribution(valid_generator, 'Class Distribution in Validation Set')

# Model architecture
def base_model(input_shape, repetitions):
    input_ = Input(shape=input_shape, name='input')
    x = input_

    for i in range(repetitions):
        n_filters = 2**(4 + i)
        x = Conv2D(n_filters, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(2)(x)

    return x, input_

def final_model(input_shape, repetitions):
    x, input_ = base_model(input_shape, repetitions)
    x = Conv2D(64, 3, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    class_out = Dense(num_classes, activation='softmax', name='class_out')(x)

    model = Model(inputs=input_, outputs=class_out)
    print(model.summary())
    return model

model = final_model(image_resize, 4)

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=logdir)
checkpoint = ModelCheckpoint(
    'saved_model/custom_best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
)
callbacks_list = [checkpoint, tensorboard_callback]

# Train model
num_epochs = 2
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    callbacks=callbacks_list
)

# Plot results
plot_acc(history)
plot_loss(history)

# Save the model
model.save('saved_model/custom_model.h5')
print("Model saved!")

cam_model = load_model('saved_model/custom_model.h5')
def show_cam(image_value, features, results):
    features_for_img = features[0]
    prediction = results[0]

    class_activation_weights = model.layers[-1].get_weights()[0][:, 1]
    class_activation_features = sp.ndimage.zoom(features_for_img, (im_size/10, im_size/10, 1), order=2)
    cam_output = np.dot(class_activation_features, class_activation_weights)

    # Visualize the results
    plt.figure(figsize=(12, 12))
    plt.imshow(cam_output, cmap='jet', alpha=0.5)
    plt.imshow(tf.squeeze(image_value), alpha=0.5)
    plt.title('Class Activation Map')
    plt.figtext(.5, .05, f"No Wildfire Probability: {results[0][0] * 100}%\nWildfire Probability: {results[0][1] * 100}%", ha="center", fontsize=12, bbox={"facecolor":"green", "alpha":0.5, "pad":3})
    plt.colorbar()
    plt.show()

def convert_and_classify(image_path):
    cam_model = load_model('saved_model/custom_model.h5')

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (im_size, im_size)) / 255.0
    tensor_image = np.expand_dims(img, axis=0)

    features, results = cam_model.predict(tensor_image)
    show_cam(tensor_image, features, results)

convert_and_classify('Dataset/test/nowildfire/-73.7294,45.597491.jpg')
convert_and_classify('Dataset/test/wildfire/-61.87285,47.36931.jpg')

path = 'Dataset/test/wildfire/-62.81924,50.34413.jpg'
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
