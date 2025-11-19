import cv2
from preprocessing import enhance_image
from classifier import predict_emotions_on_image, top_emotion_from_result

def process_image_file(path):
    img = cv2.imread(path)
    if img is None:
        print("Could not read image")
        return

    enhanced, steps = enhance_image(img)
    print("Enhancement Steps:", steps)

    results = predict_emotions_on_image(enhanced)

    if not results:
        print("No face detected")
    else:
        for i, r in enumerate(results):
            top, score = top_emotion_from_result(r)
            print(f"Face {i+1}: {top} ({score:.2f})")

    cv2.imshow("Enhanced Image", enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def webcam():
    cap = cv2.VideoCapture(1)
    print("Press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        enhanced, steps = enhance_image(frame)
        results = predict_emotions_on_image(enhanced)

        if results:
            for r in results:
                (x, y, w, h) = r["box"]
                top, score = top_emotion_from_result(r)

                cv2.rectangle(enhanced, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(enhanced, f"{top} {score:.2f}",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,0), 2)

        cv2.imshow("Emotion Webcam", enhanced)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--webcam", action="store_true")

    args = parser.parse_args()

    if args.image:
        process_image_file(args.image)
    elif args.webcam:
        webcam()
    else:
        print("Use --image or --webcam")
