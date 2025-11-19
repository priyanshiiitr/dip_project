from fer import FER
import cv2

_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = FER(mtcnn=True)   # Better accuracy
    return _detector

def predict_emotions_on_image(img):
    detector = get_detector()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return detector.detect_emotions(rgb)

def top_emotion_from_result(r):
    emotions = r["emotions"]
    top = max(emotions, key=emotions.get)
    return top, emotions[top]
