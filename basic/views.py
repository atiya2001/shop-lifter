import os
import numpy as np                                   # type: ignore
import cv2                                           # type: ignore
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from tensorflow.keras.models import load_model       # type: ignore
import logging

logger = logging.getLogger(__name__)

# Load the pre-trained model
try:
    model = load_model(settings.MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to load the model: {e}")
    model = None

def index(request):
    return render(request, 'video_processing.html')

def extract_frames(video_file):
    frames = []
    video = cv2.VideoCapture(video_file.temporary_file_path())
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frame_resized = cv2.resize(frame, (224, 224))
        frames.append(frame_resized)
    
    video.release()
    return np.array(frames) / 255.0

def video_processing_view(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('video'):
        try:
            video = request.FILES['video']
            
            # Extract frames
            frames = extract_frames(video)
            
            if model is None:
                raise ValueError("Model not loaded")
            
            # Predict using the model
            predictions = model.predict(frames)
            mean_prediction = np.mean(predictions)
            predicted_label = 'Positive' if mean_prediction > 0.5 else 'Negative'
            
            context['predicted_label'] = predicted_label
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            context['error_message'] = f"An error occurred while processing the video: {str(e)}"
    
    return render(request, 'video_processing.html', context)