import streamlit as st
import tensorflow as tf
import numpy as np

st.title("TensorFlow Test")

if st.button("Test"):
    st.write("Loading model")

    model = tf.keras.models.load_model("model.h5")

    st.write("Loaded")

    x = np.zeros((1, 12), dtype=np.float32)

    st.write("Predicting")

    y = model.predict(x, verbose=0)

    st.write("Done")
    st.write(y)