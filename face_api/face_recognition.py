import cv2
import numpy as np
from deepface import DeepFace
import os
import logging
from .models import Person
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self):
        self.known_faces = []
        self.load_known_faces()

    def preprocess_image(self, image):
        """Preprocess image to improve face detection with better low-light handling."""
        # Resize to ensure sufficient size
        image = cv2.resize(image, (300, 300))
        # Adjust brightness and contrast with milder settings
        image = cv2.convertScaleAbs(image, alpha=1.1, beta=5)  # Reduced alpha and beta
        return image

    def load_known_faces(self):
        """Load face embeddings from database."""
        people = Person.objects.all()
        if not people.exists():
            logger.warning("No persons found in the database")
            return

        for person in people:
            image_path = person.image.path
            logger.info(f"Loading image for {person.name} from {image_path}")
            try:
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.preprocess_image(image)

                debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'debug', 'preprocessed')
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f"{person.name}_image.jpg")
                cv2.imwrite(debug_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved preprocessed image to {debug_path}")

                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=True
                )[0]["embedding"]
                self.known_faces.append({
                    'name': person.name,
                    'embedding': np.array(embedding),
                    'image_path': image_path
                })
                logger.info(f"Loaded face for {person.name} from {image_path}")
            except Exception as e:
                logger.error(f"Error processing {image_path} for {person.name}: {str(e)}")

    def capture_webcam_image(self):
        """Capture an image from the webcam with improved initialization."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam - check permissions or camera connection")
            raise Exception("Could not open webcam")

        # Give the camera time to adjust (e.g., auto-exposure)
        time.sleep(1)  # Wait 1 second for stabilization
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.error("Could not capture image from webcam - frame is empty")
            raise Exception("Could not capture image")

        logger.info("Webcam frame captured successfully")
        # Convert BGR to RGB and preprocess
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = self.preprocess_image(rgb_frame)

        # Save for debugging
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"webcam_{int(time.time())}.jpg")
        cv2.imwrite(debug_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved webcam image to {debug_path}")

        return rgb_frame

    def get_face_embedding(self, image_path):
        """Get face embedding from an image."""
        try:
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=True
            )[0]["embedding"]
            logger.info(f"Face detected in {image_path}")
            return np.array(embedding)
        except Exception as e:
            logger.error(f"No face detected in {image_path}: {str(e)}")
            return None

    def compare_faces(self, unknown_image_path, threshold=1.0):
        """Compare unknown face embedding against known faces."""
        if not self.known_faces:
            logger.warning("No known faces loaded")
            return None, float('inf')

        distances = []
        for face in self.known_faces:
            try:
                result = DeepFace.verify(
                    img1_path=face['image_path'],
                    img2_path=unknown_image_path,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False,
                    distance_metric="cosine"
                )
                distance = result["distance"]
                distances.append((face['name'], distance))
                logger.info(f"Compared with {face['name']}: distance={distance}")
            except Exception as e:
                logger.error(f"Error comparing with {face['name']}: {str(e)}")

        if not distances:
            return None, float('inf')

        best_match = min(distances, key=lambda x: x[1])
        logger.info(f"Best match: {best_match[0]} with distance={best_match[1]}")
        if best_match[1] <= threshold:
            return best_match[0], best_match[1]
        return None, best_match[1]