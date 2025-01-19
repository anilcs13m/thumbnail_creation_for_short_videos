import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
import glob
import time

def load_nima_model(weights_path):
    """
    Load the NIMA (Neural Image Assessment) model with MobileNet as base.

    Args:
        weights_path (str): Path to the model weights.

    Returns:
        Model: A compiled Keras model.
    """
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    layer = Dropout(0.75)(base_model.output)
    output_layer = Dense(10, activation='softmax')(layer)
    model = Model(base_model.input, output_layer)
    model.load_weights(weights_path)
    return model

def calculate_mean_score(scores):
    """
    Calculate the mean score for an image based on its scores.

    Args:
        scores (np.array): Array of scores.

    Returns:
        float: Mean score.
    """
    si = np.arange(1, 11, 1)
    return np.sum(scores * si)

def get_image_score(image_path, model):
    """
    Calculate the NIMA score for a single image.

    Args:
        image_path (str): Path to the image.
        model (Model): Pre-trained NIMA model.

    Returns:
        float: Calculated NIMA score.
    """
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0)
    image_array = preprocess_input(image_array)
    scores = model.predict(image_array, batch_size=1, verbose=0)[0]
    return calculate_mean_score(scores)

def get_image_list(directory):
    """
    Retrieve the list of image files in a directory.

    Args:
        directory (str): Path to the image directory.

    Returns:
        list: List of image filenames.
    """
    return [f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

def process_images_in_batches(image_list, image_dir, model, batch_size=45):
    """
    Process images in batches to calculate NIMA scores.

    Args:
        image_list (list): List of image filenames.
        image_dir (str): Path to the image directory.
        model (Model): Pre-trained NIMA model.
        batch_size (int): Number of images to process in a batch.

    Returns:
        dict: Dictionary with image filenames as keys and NIMA scores as values.
    """
    nima_scores = {}
    total_images = len(image_list)

    for start in range(0, total_images, batch_size):
        end = min(start + batch_size, total_images)
        batch_images = image_list[start:end]

        batch_data = []
        for img_name in batch_images:
            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (224, 224))
            batch_data.append(img_resized)

        batch_data = np.array(batch_data)
        batch_data = preprocess_input(batch_data)

        scores_batch = model.predict(batch_data, batch_size=batch_size, verbose=0)
        for i, score in enumerate(scores_batch):
            nima_scores[batch_images[i]] = calculate_mean_score(score)

    return nima_scores

def select_top_images(image_scores, top_n=2):
    """
    Select top N images based on their NIMA scores.

    Args:
        image_scores (dict): Dictionary of image scores.
        top_n (int): Number of top images to select.

    Returns:
        list: List of tuples with image filename and its score.
    """
    return sorted(image_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

if __name__ == "__main__":
    WEIGHTS_PATH = "mobilenet_weights.h5"
    IMAGE_DIR = "/path/to/images"

    model = load_nima_model(WEIGHTS_PATH)
    images = get_image_list(IMAGE_DIR)

    # Process images and get NIMA scores
    scores = process_images_in_batches(images, IMAGE_DIR, model, batch_size=45)

    # Get top images based on scores
    top_images = select_top_images(scores, top_n=5)
    print("Top Images:", top_images)

def test_calculate_mean_score():
    scores = np.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0])
    expected_mean = 4.25
    assert abs(calculate_mean_score(scores) - expected_mean) < 1e-6

def test_get_image_list():
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    test_files = ["test1.jpg", "test2.png", "ignore.txt"]
    for file in test_files:
        open(os.path.join(test_dir, file), 'a').close()
    
    image_list = get_image_list(test_dir)
    assert len(image_list) == 2
    assert "test1.jpg" in image_list
    assert "test2.png" in image_list

    shutil.rmtree(test_dir)

