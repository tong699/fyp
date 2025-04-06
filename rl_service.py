import streamlit as st
import numpy as np
import cv2
import os
from collections import Counter

st.set_page_config(page_title="SIFT Banknote Matcher", layout="centered")
st.title("💵 Multi-Banknote Classifier using SIFT")

# Match ROI with SIFT against all templates
def match_sift_roi(roi, templates):
    gray1 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)

    best_match = None
    best_score = 0

    for label, template in templates.items():
        gray2 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            continue

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) > best_score:
            best_score = len(good)
            best_match = label

    return best_match, best_score

# Load templates
TEMPLATE_PATH = "templates"
templates = {}
for file in os.listdir(TEMPLATE_PATH):
    if file.endswith((".jpg", ".jpeg", ".png")):
        label = os.path.splitext(file)[0]
        img = cv2.imread(os.path.join(TEMPLATE_PATH, file))
        if img is not None:
            templates[label] = img

# Upload an image
uploaded_file = st.file_uploader("Upload an image containing banknotes", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    # Preprocess image for contour detection
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_img = input_img.copy()
    detected_notes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = input_img[y:y+h, x:x+w]

        label, score = match_sift_roi(roi, templates)

        if label:
            detected_notes.append(label)
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output_img, f"${label} ({score})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)

    st.subheader("🧾 Results")
    st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), caption="Detected Banknotes", use_column_width=True)

    if detected_notes:
        note_count = Counter(detected_notes)
        st.success("Detected Notes:")
        for note, count in note_count.items():
            st.markdown(f"- **${note}** × {count}")
    else:
        st.warning("No recognizable banknotes found.")
else:
    st.info("Please upload an image to start.")
