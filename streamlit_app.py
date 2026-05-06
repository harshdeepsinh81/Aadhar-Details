import streamlit as st
import requests, base64, io
from PIL import Image

st.title("Aadhar OCR")

uploaded_file = st.file_uploader("Upload an Aadhar card image", type=["jpg", "jpeg", "png"]) 
#  .name = "ex1.jpg"
#  .type = "image/jpeg"
#  .getvalue() = raw bytes of the image

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Aadhar Card",width=400)

    if st.button("Extract Info"):
        files = {"file": uploaded_file.getvalue()}

        response = requests.post("http://localhost:5000/", files={"file": uploaded_file.getvalue()})

        if response.status_code == 200:
            data = response.json()
            st.success("Successfull")
            st.json(data["ocr"])

            img_bytes = base64.b64decode(data["masked_image"])
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, width=400)


        else:
            st.error("Error: " + response.text)