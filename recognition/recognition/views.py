from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque
import json
import os 


# Initialize the gesture recognizer
class SignLanguageRecognizer:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, '..', 'sign_language_model.h5')  # Adjust path here
        print("path of the .h5 model: ", model_path)
        print("path of the os .h5 model: ", os.path.exists(model_path))  # Should return True if the file exists
        try:    
            self.model = load_model(model_path)
        except Exception as e:    
            print(f"Error loading the model: {str(e)}")
        self.max_length = 90
        self.sequence = deque(maxlen=self.max_length)
        self.gesture_labels = ['Alive', 'Bad', 'Beautiful', 'Big large', ...]
        self.predictions = []

        base_dir = os.path.dirname(os.path.abspath(__file__))

        def extract_landmarks(self, frame):
            # Extract landmarks from the image frame
            results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lh = np.zeros(21*3)
            rh = np.zeros(21*3)
            pose = np.zeros(33*4)
            face = np.zeros(468*3)
        
            if results.left_hand_landmarks:
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks:
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
            if results.pose_landmarks:
                pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
            if results.face_landmarks:
                face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
        
            return np.concatenate([lh, rh, pose, face])

        def recognize_gesture(self, landmarks):
            # Process sequence of landmarks and predict gesture
            if len(self.sequence) == self.max_length:
                res = self.model.predict(np.expand_dims(np.array(self.sequence), axis=0))[0]
                self.predictions.append(np.argmax(res))
                if len(self.predictions) > 5:
                    self.predictions = self.predictions[-5:]
                if len(self.predictions) >= 3 and len(set(self.predictions[-3:])) == 1:
                    return self.gesture_labels[self.predictions[-1]]
            return ""

        def process_frame(self, frame):
            landmarks = self.extract_landmarks(frame)
            self.sequence.append(landmarks)
            return self.recognize_gesture(landmarks)

recognizer = SignLanguageRecognizer()

def index(request):
    # Render the HTML template for the index page
    labels = recognizer.gesture_labels
    return render(request, 'recognition/index.html', {'labels': labels})

@csrf_exempt
def process_frame(request):
    if request.method == 'POST':
        # Check if 'frame' is included in the request
        if 'frame' not in request.FILES:
            return JsonResponse({'error': 'No frame part'}, status=400)

        frame_file = request.FILES.get('frame')
        if not frame_file:
            return JsonResponse({'error': 'No selected file'}, status=400)

        try:
            image_bytes = frame_file.read()
            image = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

            if frame is None:
                return JsonResponse({'error': 'Frame decoding failed'}, status=500)

            gesture = recognizer.process_frame(frame)
            if gesture:
                return JsonResponse({'gesture': gesture})
            else:
                return JsonResponse({'gesture': 'No gesture recognized'})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)
