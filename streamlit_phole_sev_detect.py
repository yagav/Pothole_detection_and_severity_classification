
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('best.pt')

def detect_severity(y,w,h,Ow,Oh):
    farther_score = y/Oh
    normalised_score = (h*w)/(Ow*Oh)
    severity_score = 0.6*farther_score + 0.4*normalised_score

    if severity_score<0.15:
        return "Low"
    elif severity_score<0.3:
        return "Mid"
    else:
        return "High"

def draw_box(img,x,y,w,h,severity):
    cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,0),thickness=2)
    cv2.putText(img,severity,(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=2)

    return img



import streamlit as st
from PIL import Image
st.set_page_config(page_title="Pothole Severity Detection", layout="centered")

st.title("Pothole Severity Detection System")
st.markdown("""
Welcome to the **Pothole Severity Detection System**!

This webpage will take a road image as an input and spot and classify the potholes by its severity.

---

## Explanation:
The YOLOv12 model has been used for detection of the potholes. The model was fine-tuned using an open-source dataset from roboflow.
The fine-tuning was done in google-colab and the updated weights are stored in the best.pt file. This file is then loaded for
getting inferences of where the pothole is. The inference from the model is used to calculate the severity. To calculate the
severity we take into account the normalised area - area of pothole relative to the whole image, and how far the pothole is in 
the image using the normalised y-axis distance.
            

""")
st.markdown("#### You may have a demo by uploading a image of a road")



uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.subheader("Input Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    for detects in model([img])[0]:
        detect = detects.boxes.cpu().numpy()[0]
        original_h, original_w =  detect.orig_shape
        x,y = detect.xyxy[0][:2]
        w,h = detect.xywh[0][2:]
        severity = detect_severity(y,w,h,original_h,original_h)
        img = draw_box(img,int(x),int(y),int(w),int(h),severity)


    st.subheader("Processed Output Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
else:
    st.info("Please upload a road image to begin.")
