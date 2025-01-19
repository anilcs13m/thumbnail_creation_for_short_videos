import os
import glob
import numpy as np
import tensorflow as tf
from keras.applications.xception import preprocess_input
from keras.models import load_model
import keras.utils as image

# Set visible GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Load the pre-trained model
model = load_model('updated_face_validation_model.h5')
label_map = {'none_face': 1, 'face': 0}

def filter_non_faces(image_directory, image_list, start_index=0):
    """
    Removes images classified as 'none_face' from the given directory.

    Args:
        image_directory (str): Path to the image directory.
        image_list (list): List of image filenames.
        start_index (int): Index to start processing from.
    """
    for count, image_name in enumerate(image_list[start_index:], start=start_index):
        print(f"Processing image {count + 1}...")
        try:
            image_path = os.path.join(image_directory, image_name)
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            predictions = model.predict(img_array)[0]
            predicted_class = sorted(label_map)[np.argmax(predictions)]

            if predicted_class == 'none_face':
                os.remove(image_path)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

def batch_remove_non_faces(image_directory, image_list, batch_size=45):
    """
    Processes images in batches and removes 'none_face' classified images.

    Args:
        image_directory (str): Path to the image directory.
        image_list (list): List of image filenames.
        batch_size (int): Number of images to process in a batch.
    """
    total_images = len(image_list)
    
    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        batch_images = image_list[batch_start:batch_end]

        try:
            batch_data = []
            for image_name in batch_images:
                image_path = os.path.join(image_directory, image_name)
                img = image.load_img(image_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                batch_data.append(img_array)

            batch_data = np.array(batch_data)
            batch_data = preprocess_input(batch_data)

            predictions = model.predict(batch_data)

            for idx, prediction in enumerate(predictions):
                predicted_class = sorted(label_map)[np.argmax(prediction)]
                if predicted_class == 'none_face':
                    os.remove(os.path.join(image_directory, batch_images[idx]))

        except Exception as e:
            print(f"Error processing batch {batch_start // batch_size + 1}: {e}")

if __name__ == "__main__":
    image_directory = "/nfs/training/anil/thumnail_detection_crop/acting"
    image_list = glob.glob(os.path.join(image_directory, "*"))

    # Uncomment the method you want to use:

    # Single image processing
    # filter_non_faces(image_directory, image_list, start_index=2050646)

    # Batch processing
    batch_remove_non_faces(image_directory, image_list, batch_size=45)

