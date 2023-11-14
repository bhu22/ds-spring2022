import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports
from multipage import MultiPage
from pages import slider_prediction, Data_visualization, DataDescription

# Create an instance of the app
app = MultiPage()

# Title of the main page
display = Image.open('Wine.jpg')
display = np.array(display)
# st.image(display, width = 400)
# st.title("Data Storyteller Application")
col1, col2 = st.columns(2)
col1.image(display, width = 350)
col2.title("Wine Quality Prediction Project")
# st.title("Wine Quality Prediction Project")

# Add all your application here
app.add_page("Dataset Visualization", Data_visualization.app)
app.add_page("Wine Quality Machine Learning Model Prediction ", slider_prediction.app)
app.add_page("Dataset Description",DataDescription.app)

# The main app
app.run()