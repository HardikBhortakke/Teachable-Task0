import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, save_model
import os
import cv2
import numpy as np

# Function to load images and labels from a folder
def load_images_from_folder(folder_path):
    images = []
    labels = []
    for label, class_folder in enumerate(os.listdir(folder_path)):
        class_folder_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_folder_path):
            for filename in os.listdir(class_folder_path):
                img_path = os.path.join(class_folder_path, filename)
                # Load image
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)
    return images, labels

# Function to preprocess images
def preprocess_images(images, size=(64, 64)):
    processed_images = []
    for img in images:
        resized_img = cv2.resize(img, size)  # Resize images to a fixed size
        processed_images.append(resized_img)
    return processed_images

# Function to save dataset
def save_dataset(images, labels, output_file):
    np.savez(output_file, images=images, labels=labels)

# Main function to iterate over folders, load, preprocess, and save dataset
def generate_dataset(folder_path, output_file):
    dataset_images = []
    dataset_labels = []
    for class_folder in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_folder_path):
            print(f"Processing images from folder: {class_folder}")
            images, labels = load_images_from_folder(class_folder_path)
            if len(images) == 0:
                print(f"No images found in folder: {class_folder}")
            else:
                processed_images = preprocess_images(images)
                dataset_images.extend(processed_images)
                dataset_labels.extend(labels)

    # Convert lists to numpy arrays
    dataset_images = np.array(dataset_images)
    dataset_labels = np.array(dataset_labels)

    # Shuffle dataset
    indices = np.arange(len(dataset_images))
    np.random.shuffle(indices)
    dataset_images = dataset_images[indices]
    dataset_labels = dataset_labels[indices]

    # Save dataset
    save_dataset(dataset_images, dataset_labels, output_file)

def load_dataset(dataset_file):
    data = np.load(dataset_file)
    images = data['data']
    labels = data['labels']
    return images, labels

def train_set(images, labels):
    X = np.array(images)
    #print(X.shape)
    X_flatten = X.reshape(X.shape[0], -1).T
    train_x = X_flatten/255
    #print(train_x.shape,train_x.dtype)
    train_y = np.array(labels)
    train_y = train_y.reshape(train_y.shape[0],1)
    train_y = train_y.astype(np.float32)
    #print(train_y.shape,train_y.dtype)
    #print(train_y)
    return train_x, train_y

def train_model(train_x, train_y):
    tf.random.set_seed(1234)  # applied to achieve consistent results
    model = Sequential(
        [
            tf.keras.Input(shape=(12288,)),
            Dense(64, activation='LeakyReLU', name = 'layer1'),
            Dense(3, activation='LeakyReLU', name = 'layer2'),
            Dense(1, activation='sigmoid', name = 'layer3')
        ]
    )
    model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    )

    model.fit(
        train_x.T,train_y,
        epochs=50,
    )
    model.save('model.h5')  # Save the entire model to a single .h5 file

def predict(model_path, test_x, threshold):
    model = load_model(model_path)
    predictions = model.predict(test_x.T)
    for i in range(predictions.shape[0]):
        predictions[i] = 1 if predictions[i] > threshold else 0
    #print(predictions)

    return predictions

def cam_ip(model_path):
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the path to a video file
    model = load_model(model_path)
    while True:
        # Read a frame from the video feed
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        processed_frame = preprocess_images([frame])

        # Make predictions
        predictions = model.predict(processed_frame)

        # Optionally, display predictions on the frame
        # For example, you can draw bounding boxes or labels based on predictions

        # Display the frame
        cv2.imshow('Live Feed', frame)

        # Check for user interrupt
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture object
    cap.release()
    cv2.destroyAllWindows()