import tensorflow as tf
print("TF:", tf.__version__)

from tensorflow.keras.models import load_model
print("TF Keras OK")

import streamlit as st
print("Streamlit:", st.__version__)

import google.protobuf
print("Protobuf:", google.protobuf.__version__)
