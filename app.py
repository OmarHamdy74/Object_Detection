
import cv2
import mediapipe as mp
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Page Config
st.set_page_config(page_title="MediaPipe Object Detection", layout="centered")

# UI Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📦 Object Detection with MediaPipe</h1>", unsafe_allow_html=True)
st.write("Upload an image to detect objects using MediaPipe's pre-trained model.")

# Upload image
upload_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

# Constants
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # Red

def visualize(image, detection_result) -> np.ndarray:
    """Draw bounding boxes and labels."""
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        category = detection.categories[0]
        label = category.category_name
        score = round(category.score, 2)
        text = f"{label} ({score})"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image

# Load model (once)
@st.cache_resource
def load_detector():
    base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    return vision.ObjectDetector.create_from_options(options)

detector = load_detector()

if upload_file is not None:
    file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Detecting objects..."):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detection_result = detector.detect(mp_img)

    # Draw boxes
    annotated_image = visualize(img.copy(), detection_result)
    st.image(annotated_image, caption="✅ Detected Objects", use_column_width=True)

    # Show detected objects as list
    if detection_result.detections:
        st.subheader("📋 Detected Objects")
        for i, detection in enumerate(detection_result.detections):
            category = detection.categories[0]
            st.markdown(f"**{i+1}. {category.category_name}** — Score: `{round(category.score, 2)}`")
    else:
        st.warning("No objects detected.")
