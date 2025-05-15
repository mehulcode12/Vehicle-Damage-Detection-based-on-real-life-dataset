import streamlit as st

from model_helper import predict

st.title("Vehicle Damage Detection based on Real-life Dataset")

input_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if input_image:

    # getting image in binary format and in temp file
    image_path = "\temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(input_image.getbuffer())


    # Display the uploaded image
    st.image(input_image, caption="Uploaded Image", use_container_width=True)
    prediction = predict(image_path)
    st.info(f"Prediction: {prediction}")


    


    

   
