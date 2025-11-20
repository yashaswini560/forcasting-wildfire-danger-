import os
import tensorflow as tf
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import cv2
import numpy as np

for device in tf.config.list_physical_devices():
    print(f"{device.name}")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

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

train_path = "Dataset/train"
valid_path = "Dataset/valid"
test_path = "Dataset/test"

im_size = 224
image_resize = (im_size, im_size, 3)
batch_size = 100
num_classes = 2

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

class_mapping = train_generator.class_indices
class_names = list(class_mapping.keys())
print("Class names:", class_names)

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


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=logdir)
checkpoint = ModelCheckpoint(
    'saved_model/custom_best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
)
callbacks_list = [checkpoint, tensorboard_callback]


num_epochs = 2
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    callbacks=callbacks_list
)


plot_acc(history)
plot_loss(history)


model.save('saved_model/custom_model.h5')
print("Model saved!")
