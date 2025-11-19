import streamlit as st
import numpy as np
from PIL import Image
import cv2

from preprocessing import enhance_image
from classifier import predict_emotions_on_image, top_emotion_from_result

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av


st.title("Facial Emotion Recognition (with DIP preprocessing)")

mode = st.sidebar.radio("Select Mode", ["Upload Image", "Real-Time Camera"])



# ================================
# 1️⃣ UPLOAD IMAGE MODE (Keep DIP)
# ================================
if mode == "Upload Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if file is not None:
        pil = Image.open(file).convert("RGB")
        img = np.array(pil)[:, :, ::-1]

        st.image(pil, caption="Original Image")

        # Apply DIP pipeline
        enhanced, steps = enhance_image(img)
        st.write("DIP Steps:", steps)

        # Emotion Prediction
        results = predict_emotions_on_image(enhanced)

        if not results:
            st.warning("No face detected")
        else:
            for r in results:
                top, score = top_emotion_from_result(r)
                st.success(f"{top} — {score:.2f}")

        st.image(enhanced[:, :, ::-1], caption="Enhanced Image")



# ================================
# 2️⃣ REAL-TIME WEBCAM MODE (NO DIP)
# ================================
else:
    st.subheader("Real-Time Webcam Emotion Detection (No DIP – Faster FPS)")

    class FastEmotionProcessor(VideoTransformerBase):
        def transform(self, frame):

            # Convert frame to BGR
            img = frame.to_ndarray(format="bgr24")

            # Resize to speed up processing
            img = cv2.resize(img, (640, 480))

            # NO DIP HERE → Fast processing
            results = predict_emotions_on_image(img)

            # Draw predictions
            if results:
                for r in results:
                    (x, y, w, h) = r["box"]
                    emotion, score = top_emotion_from_result(r)

                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"{emotion.upper()} {score:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

            return img


    webrtc_streamer(
        key="fast-emotion-detection",
        video_transformer_factory=FastEmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
