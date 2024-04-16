#pip install keras
#pip install tensorflow
#pip install scikit-image


import streamlit as st
import numpy as np
from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize

model = load_model("fruits_classification_model.h5")

categories = ['Guava', 'Papaya', 'Strawberry', 'Mango', 'Apple', 'Orange', 'Banana', 'Kiwi', 'Watermelon']


# Function to preprocess the image
def preprocess_image(image):
    img_array = imread(image)
    img_resized = resize(img_array, (150, 150, 3))
    return img_resized[np.newaxis, :]


# GitHub and Colab links
def render_links():
    st.sidebar.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue?logo=GitHub)](https://github.com/Mr-Nikhil-M/Fruits_classification_CNN)",
        unsafe_allow_html=True)
    st.sidebar.markdown(
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n34c5uzgQhmGVyEu7dfh-WBRkv0VTO-9#scrollTo=aq5cNx2EMIfk)",
        unsafe_allow_html=True)


# Streamlit app
def main():
    st.title("Fruit Classification")

    # Add image after the title
    st.image("image.jpg", caption="Fruit Image", use_column_width=True)

    # Render GitHub and Colab links in the sidebar
    render_links()

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
        st.write("")

        # Predict button
        if st.button("Predict", key="predict-button", help="Click to predict"):
            st.write("Classifying...")

            # Preprocess the image
            processed_image = preprocess_image(uploaded_image)

            # Make prediction
            prediction = model.predict(processed_image)
            predicted_class = categories[np.argmax(prediction)]

            # Display the result
            st.write("Prediction:", predicted_class)


if __name__ == "__main__":
    main()

