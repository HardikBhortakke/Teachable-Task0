import models
import tensorflow as tf
from tensorflow import keras
from keras.layers import Flatten, Input, Activation, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, GlobalAveragePooling2D, MaxPooling2D, Add, concatenate,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, save_model
from keras.utils import get_file
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
import os
import cv2
import numpy as np

mode = 0

# Function to load images and labels from a folder
def load_images_from_folder(folder_path):
    images = []
    labels = []
    num_max = max(enumerate(os.listdir(folder_path)))
    num_classes = num_max[0]
    print(num_classes)
    for label, class_folder in enumerate(os.listdir(folder_path)):
        class_folder_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_folder_path):
            for filename in os.listdir(class_folder_path):
                img_path = os.path.join(class_folder_path, filename)
                # Load image
                #print(img_path)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    row = [0] * (num_classes + 1)  # Initialize a row with zeros
                    row[label] = 1               # Set the element at the index 'value' to 1
                    labels.append(row)
    return images, labels, num_classes

# Function to preprocess images
def preprocess_images(images, size=(64, 64)):
    if(mode):
        size = (299,299)
    processed_images = []
    for img in images:
        resized_img = cv2.resize(img, size)  # Resize images to a fixed size
        processed_images.append(resized_img)
    return processed_images

# Function to save dataset
def save_dataset(images, labels, output_file):
    np.savez(output_file, images=images, labels=labels)

# Main function to iterate over folders, load, preprocess, and save dataset
def generate_dataset(folder_path):
    dataset_images = []
    dataset_images1 = []
    dataset_labels = []
    output_file = os.path.join("dataset.npz")
    output_file1 = os.path.join("dataset1.npz")
    
    print(f"Processing images from folders")
    images, labels, num_classes = load_images_from_folder(folder_path)
    if len(images) == 0:
        print(f"No images found in folders")
    else:
        processed_images = preprocess_images(images)
        processed_images1 = preprocess_images(images, size = (299,299))
        dataset_images.extend(processed_images)
        dataset_images1.extend(processed_images1)
        dataset_labels.extend(labels)

    # Convert lists to numpy arrays
    dataset_images = np.array(dataset_images)
    dataset_images1 = np.array(dataset_images1)
    dataset_labels = np.array(dataset_labels)

    # Shuffle dataset
    indices = np.arange(len(dataset_images))
    np.random.shuffle(indices)
    dataset_images = dataset_images[indices]
    dataset_images1 = dataset_images1[indices]
    dataset_labels = dataset_labels[indices]

    # Save dataset
    save_dataset(dataset_images, dataset_labels, output_file)
    save_dataset(dataset_images1, dataset_labels, output_file1)
    num_classes += 1
    if(mode):
        return output_file1,num_classes
    else:
        return output_file, num_classes

def load_dataset(dataset_file):
    data = np.load(dataset_file)
    print(data)
    images = data['images']
    labels = data['labels']
    return images, labels

def train_set(images, labels, num_classes = 2):
    X = np.array(images)
    #print(X.shape)
    X_flatten = X.reshape(X.shape[0], -1).T
    if (mode):
        # train_x = (np.transpose(X, (1, 2, 3, 0))/127.5) - 1
        train_x = (X/127.5) - 1
    else:
        train_x = (X_flatten/127.5) - 1
    #print(train_x.shape,train_x.dtype)
    train_y = np.array(labels)
    train_y = train_y.reshape(train_y.shape[0],num_classes)
    train_y = train_y.astype(np.float32)
    #print(train_y.shape,train_y.dtype)
    #print(train_y)
    return train_x, train_y

def test_set(images):
    X = np.array(images)
    #print(X.shape)
    if(mode):
        train_x = X/127.5
    else:
        X_flatten = X.reshape(X.shape[0], -1).T
        train_x = X_flatten/127.5
    train_x -= 1
    return train_x

def train_model(train_x, train_y, model_path, num_classes):
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        tf.random.set_seed(1234)  # applied to achieve consistent results
        model = models.custom_mod0(num_classes = num_classes)
    model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    )
    if (mode):
        model.fit(
            train_x,train_y,
            epochs=10,
        )
    else:
        model.fit(
            train_x.T,train_y,
            epochs=5,
        )
    model.save('model.keras')  

def predict(model_path, test_x):
    model = load_model(model_path)
    model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    )
    if (mode == 1):
        predictions = model.predict(test_x)
    else:
        predictions = model.predict(test_x.T)
    print(predictions)

    return predictions
