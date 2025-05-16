import streamlit as st
from PIL import Image
from model_helper import predict

st.title("Vehicle Damage Detection based on Real-life Dataset")
st.info("The model is trained on third quarter front and rear view; the picture should capture this angle of the car.")

input_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if input_image is not None:
    image = Image.open(input_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    prediction = predict(image)
    st.info(f"Prediction: {prediction}")
