from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .face_recognition import FaceRecognizer
from django.core.files.uploadedfile import InMemoryUploadedFile
from PIL import Image
import numpy as np
import cv2
import io
import tempfile
import os
import logging
import time
from django.conf import settings

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CameraTestView(APIView):
    def __init__(self):
        super().__init__()
        self.recognizer = FaceRecognizer()

    def post(self, request):
        try:
            # Create debug directory
            debug_dir = os.path.join(settings.BASE_DIR, 'debug')
            os.makedirs(debug_dir, exist_ok=True)

            # Capture image from webcam
            image = self.recognizer.capture_webcam_image()

            # Save image for debugging
            debug_path = os.path.join(debug_dir, f"webcam_{int(time.time())}.jpg")
            cv2.imwrite(debug_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved webcam image to {debug_path}")

            # Save image temporarily for DeepFace
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                cv2.imwrite(temp_file.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                temp_file_path = temp_file.name

            # Get face embedding
            embedding = self.recognizer.get_face_embedding(temp_file_path)
            os.unlink(temp_file_path)  # Clean up

            if embedding is None:
                return Response(
                    {"error": "No human face detected"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Compare with known faces
            name, distance = self.recognizer.compare_faces(temp_file_path)
            if name:
                return Response(
                    {"name": name, "distance": distance},
                    status=status.HTTP_200_OK
                )
            return Response(
                {"error": "No matching face found"},
                status=status.HTTP_404_NOT_FOUND
            )

        except Exception as e:
            logger.error(f"Error in CameraTestView: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CompareImageView(APIView):
    def __init__(self):
        super().__init__()
        self.recognizer = FaceRecognizer()

    def post(self, request):
        try:
            # Create debug directory
            debug_dir = os.path.join(settings.BASE_DIR, 'debug')
            os.makedirs(debug_dir, exist_ok=True)

            # Check if image is provided
            if 'image' not in request.FILES:
                return Response(
                    {"error": "No image provided"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Load uploaded image
            image_file = request.FILES['image']
            image = Image.open(image_file).convert('RGB')
            image_array = np.array(image)

            # Save image for debugging
            debug_path = os.path.join(debug_dir, f"uploaded_{int(time.time())}.jpg")
            cv2.imwrite(debug_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved uploaded image to {debug_path}")

            # Save image temporarily for DeepFace
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                cv2.imwrite(temp_file.name, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
                temp_file_path = temp_file.name

            # Get face embedding
            embedding = self.recognizer.get_face_embedding(temp_file_path)
            os.unlink(temp_file_path)  # Clean up

            if embedding is None:
                return Response(
                    {"error": "No human face detected"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Compare with known faces
            name, distance = self.recognizer.compare_faces(temp_file_path)
            if name:
                return Response(
                    {"name": name, "distance": distance},
                    status=status.HTTP_200_OK
                )
            return Response(
                {"error": "No matching face found"},
                status=status.HTTP_404_NOT_FOUND
            )

        except Exception as e:
            logger.error(f"Error in CompareImageView: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )