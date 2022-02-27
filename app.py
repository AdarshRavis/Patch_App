import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import cv2
import os
import h5py




def load_model():
    model=tf.keras.models.load_model(r'/Users/adarsh/Desktop/Streamlit_CNN/TestPatch_20211212_best.h5')
    return model

with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # Patch Reader
         """)
st.text("Adarsh Ravishankar, MD")

st.caption('This application provides only information, is not medical or treatment advice and may not be treated as such by the user. As such, this application may not be relied upon for the purposes of medical diagnosis or as a recommendation for medical care or treatment. The information on this application is not a substitute for professional medical advice, diagnosis or treatment. All content, including text, graphics, images and information, contained on or available through this application is for general information purposes only ')

file_uploaded = st.file_uploader("Choose the file", type = ['jpg', 'png', 'jpeg'])
st.set_option('deprecation.showfileUploaderEncoding', False)
model=tf.keras.models.load_model(r'/Users/adarsh/Desktop/Streamlit_CNN/TestPatch_20211212_best.h5')
my_threshold = 0.48115918


if file_uploaded is None:
    st.text("Please upload an image file")

else:
    image = Image.open(file_uploaded)
    figure = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    st.pyplot(figure)
    shape= ((100,100,3))
    test_image = image.resize((100,100))
    test_image = preprocessing.image.img_to_array(test_image)
    image_normalize = test_image/255
    image_final = np.expand_dims(image_normalize, axis=0)
    sigmoid_out = model.predict(image_final)
    print(sigmoid_out)
    st.header('The sigmoid output is')
    st.subheader(sigmoid_out)
    st.caption('The threshold is 0.48115918')
    if ( sigmoid_out> my_threshold):
        st.header('This model classifies this image as a REACTION')
    else:
        st.header('This model classifies this image as NOT A REACTION')
