import os
import subprocess
import math
import cv2
import glob
import shutil
import numpy as np
from remove_none_face import filter_non_faces, batch_remove_non_faces
from nima_score import process_images_in_batches, select_top_images, load_nima_model
from deepface import DeepFace
import pdb

def calculate_face_distance_and_area(image_path, image_name, bbox, video_id, face_crop_directory):
    """
    Calculate the distance of the face from the image center and crop the face area.

    Args:
        image_path (str): Path to the image.
        image_name (str): Name of the image file.
        bbox (dict): Bounding box of the face.
        video_id (str): Video ID for directory organization.
        face_crop_directory (str): Directory to save cropped faces.

    Returns:
        tuple: Distance of face from image center, area of the face.
    """
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    image_center_x, image_center_y = image_width // 2, image_height // 2

    left_x, top_y = bbox['x'], bbox['y']
    face_width, face_height = bbox['w'], bbox['h']

    face_center_x = left_x + face_width // 2
    face_center_y = top_y + face_height // 2

    distance = math.sqrt((image_center_x - face_center_x)**2 + (image_center_y - face_center_y)**2)
    face_area = face_width * face_height

    right_x = left_x + face_width
    bottom_y = top_y + face_height

    scale_x = int(0.35 * face_width)
    scale_y = int(0.35 * face_height)

    crop_left_x = max(0, left_x - scale_x)
    crop_top_y = max(0, top_y - scale_y)
    crop_right_x = min(image_width, right_x + scale_x)
    crop_bottom_y = min(image_height, bottom_y + scale_y)
    try:
        cropped_face = image[crop_top_y:crop_bottom_y, crop_left_x:crop_right_x]
        video_face_dir = os.path.join(face_crop_directory, video_id)
        os.makedirs(video_face_dir, exist_ok=True)
        cropped_face_path = os.path.join(video_face_dir, image_name)
        cv2.imwrite(cropped_face_path, cropped_face)
    except Exception as e:
        print(f"Error cropping face: {e}")

    return distance, face_area

def list_images_in_directory(directory):
    """
    Retrieve a list of image files from the specified directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        list: List of image filenames.
    """
    return [file for file in os.listdir(directory) if file.lower().endswith(('png', 'jpg', 'jpeg'))]

def compute_nima_scores(image_directory, image_files):
    """
    Compute NIMA scores for a list of images.

    Args:
        image_directory (str): Directory containing images.
        image_files (list): List of image filenames.

    Returns:
        dict: Mapping of image filenames to NIMA scores.
    """
    nima_scores = {}
    for image_name in image_files:
        image_path = os.path.join(image_directory, image_name)
        score = get_nima_score(image_path)
        nima_scores[image_name] = score
    return nima_scores

def select_top_emotion_images(image_directory, candidate_images, video_id, face_crop_directory):
    weight_dict = {
        'angry': 0.22, 'disgust': 0.1, 'fear': 0.15, 'happy': 0.66,
        'sad': 0.1, 'surprise': 0.44, 'neutral': 0.33
    }

    emotion_scores = {}
    center_distances = {}
    face_areas = {}
    for image_name in candidate_images:
        try:
            image_path = os.path.join(image_directory, image_name)
            analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'])
            dominant_emotion = analysis['dominant_emotion']
            emotion_score = analysis['emotion'][dominant_emotion] * weight_dict[dominant_emotion]

            emotion_scores[image_name] = emotion_score

            if 'region' in analysis and analysis['region']:
                distance, area = calculate_face_distance_and_area(
                    image_path, image_name, analysis['region'], video_id, face_crop_directory
                )
                center_distances[image_name] = distance
                face_areas[image_name] = area
        except Exception as e:
            print(f"Error analyzing image {image_name}: {e}")

    # Ensure the cropped faces directory exists
    cropped_faces_directory = os.path.join(face_crop_directory, video_id)
    if not os.path.exists(cropped_faces_directory):
        print(f"Directory not found: {cropped_faces_directory}")
        return None

    valid_cropped_images = list_images_in_directory(cropped_faces_directory)
    batch_remove_non_faces(cropped_faces_directory, valid_cropped_images)

    valid_images_after_removal = list_images_in_directory(cropped_faces_directory)

    if not valid_images_after_removal:
        return None

    sorted_by_distance = sorted(valid_images_after_removal, key=lambda img: center_distances.get(img, float('inf')))
    top_near_faces = sorted_by_distance[:50]

    sorted_by_area = sorted(top_near_faces, key=lambda img: face_areas.get(img, 0), reverse=True)
    top_area_images = sorted_by_area[:15]

    sorted_by_emotion = sorted(top_area_images, key=lambda img: emotion_scores.get(img, 0), reverse=True)

    return sorted_by_emotion[:5]


def main(frame_output_path, video_id, face_crop_directory,WEIGHTS_PATH):
    """
    Generate a thumbnail for a video based on specific criteria.

    Args:
        frame_output_path (str): Path to extracted video frames.
        video_id (str): Video ID.
        face_crop_directory (str): Directory for cropped face images.

    Returns:
        str: Path to the selected thumbnail.
    """
    video_frames_path = os.path.join(frame_output_path, video_id)
    image_files = sorted(list_images_in_directory(video_frames_path))
    model = load_nima_model(WEIGHTS_PATH)
    nima_scores = process_images_in_batches(image_files, video_frames_path, model, batch_size=45)
    sorted_nima_scores = dict(sorted(nima_scores.items(), key=lambda item: item[1], reverse=True))
    selected_images = list(sorted_nima_scores.keys())[:150]
    
    top_emotion_images = select_top_emotion_images(video_frames_path, selected_images, video_id, face_crop_directory)
    pdb.set_trace()
    if not top_emotion_images:
        return os.path.join(video_frames_path, next(iter(sorted_nima_scores)))

    top_image = max(top_emotion_images, key=lambda img: sorted_nima_scores.get(img, 0))
    return os.path.join(video_frames_path, top_image)

class VideoFrameExtractor:
    def __init__(self, output_directory):
        self.output_directory = output_directory

    def extract_frames(self, video_path, fps):
        video_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(self.output_directory, video_name)
        os.makedirs(output_path, exist_ok=True)

        command = f"ffmpeg -i {video_path} -threads 4 -qscale:v 2 -vf fps={fps} {output_path}/frame_%06d.png"
        subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    WEIGHTS_PATH = "mobilenet_weights.h5"
    video_directory = "test_video/"
    frame_output_path = "frame_path"
    face_crop_directory = "cropped_faces"

    video_files = glob.glob(os.path.join(video_directory, "*.mp4"))

    for video_path in video_files:
        video_id = os.path.basename(video_path).split('.')[0]

        extractor = VideoFrameExtractor(frame_output_path)
        extractor.extract_frames(video_path, fps=12)

        thumbnail_path = main(frame_output_path, video_id, face_crop_directory,WEIGHTS_PATH)
        print(f"Thumbnail for video {video_id}: {thumbnail_path}")

