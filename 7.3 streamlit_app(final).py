# Import streamlit
import numpy as np
import os
import streamlit as st
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dense, Input, Lambda
from keras.models import load_model
from keras.utils import custom_object_scope
from tcn import TCN
import numpy as np
import pandas as pd
from keras.models import load_model
import tensorflow as tf


filepath = r"C:\Users\Carel\Documents\Course NYP Specialist Diploma in Applied AI PDC2\SDAAI PDC2\ITI110 Project\App Deployment"
model_file = 'tcn_model_10.h5'
with custom_object_scope({'TCN': TCN}):
    #model = load_model(filepath + '0153-MAE-0.04-val_MAE-0.04-loss-0.04.h5')
    model = load_model(os.path.join(filepath, model_file))

# Create a Streamlit app
st.title('Your Energy Consumption Prediction App')

def predict():
  with st.sidebar:
    # Create sliders for each feature
    month = st.slider("Month of the year", min_value=1, max_value=12)
    hour = st.slider("Hour of the day", min_value=0, max_value=23)
    temperature = st.slider("Outside Temperature", min_value=0.0, max_value=50.0)
    humidity = st.slider("What is the Humidity", min_value=0.0, max_value=1.0)
    windSpeed = st.slider("Enter Windspeed", min_value=0.0, max_value=60.0)
    holiday = st.slider("Is it a Holiday", min_value=0, max_value=1)

    # Collect all features into a list
    input_data = [month, hour, temperature, humidity, windSpeed, holiday]

    # Convert the input data to a NumPy array
    input_data_as_numpy_array = np.asarray(input_data)
    input_array = np.array(input_data)

    # Reshape the input array to match the model's input shape
    input_reshaped = input_array.reshape((1,) + input_array.shape)

    # Make predictions
    predictions = model.predict(input_reshaped)

    return predictions

# Display the predictions on the Streamlit app
predictions = predict()

# Display the predictions on the Streamlit app
if st.button('Predict'):
    predictions = predict()
    st.write(predictions)
