"""
make sure to run this file using this command:

>>> streamlit run main.py

this will pop up a website at your default browser that you can
test the models on it.
"""

import numpy as np
import pandas as pd
import streamlit as st
from utils import LstmClassifier, GruClassifier, get_category


# for custom CSS style
with open("F:\BBC-Text-Classification\src\style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.header("BBC-News-Classification")

with st.form(key="input_form"):
    options = st.multiselect(
        'Select a model - اختر النموذج',
        ['LSTM', 'GRU'])

    text = st.text_area(label="Text - النص", height=200,
                        placeholder="Enter your text here - اكتب نصك هنا")

    submitted = st.form_submit_button("Submit")
    if submitted:
        if text != "":
            with st.spinner("Your text is being predicted..."):
                if "LSTM" in options and len(options) == 1:
                    model = LstmClassifier()
                    text_vector = model.convert_text_to_vector(text)
                    predictions = model.predict(text_vector)

                elif "GRU" in options and len(options) == 1:
                    model = GruClassifier()
                    text_vector = model.convert_text_to_vector(text)
                    predictions = model.predict(text_vector)

                elif len(options) == 2:
                    lstm_model = LstmClassifier()
                    lstm_text_vector = lstm_model.convert_text_to_vector(text)
                    lstm_predictions = lstm_model.predict(lstm_text_vector)

                    gru_model = GruClassifier()
                    gru_text_vector = gru_model.convert_text_to_vector(text)
                    gru_predictions = gru_model.predict(gru_text_vector)

                    predictions = (lstm_predictions + gru_predictions) / 2

                else:
                    st.error(
                        "Plesse, select a model to predict - من فضلك اختار النموذج")
                    st.stop()

            col1, col2 = st.columns(2)
            with col1:
                predicted_category = get_category(predictions)
                st.write("Predicted category - النتيجة المتوقعة:")
                st.success(predicted_category)

            with col2:
                st.write("Prediction score - نسبة التوقع:")
                st.success(f"{np.max(predictions)*100:.2f} %")

            st.write("Prediction plot - رسم بياني للتوقعات:")
            df = pd.DataFrame(data=predictions, columns=[
                              "Business", "Entertainment", "Politics", "Sports", "Technology"])
            st.bar_chart(df.T, height=550)
        else:
            st.error(
                'Please, enter a text wants to be predicted - من فضلك ادخل النص المراد توقع تصنيفه')
