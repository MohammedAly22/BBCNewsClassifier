import transformers
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
import streamlit as st


model = tf.keras.models.load_model('text-classifierr.h5', custom_objects={
                                   "TFBertModel": transformers.TFBertModel})
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

Ind2Label = {
    0: 'politics',
    1: 'entertainment',
    2: 'business',
    3: 'sport',
    4: 'technology'
}


def predict(model, text):
    bert_inputs = bert_tokenizer.encode_plus(
        text, max_length=128, padding='max_length', truncation=True)
    input_ids = np.array([bert_inputs['input_ids']])
    attention_mask = np.array([bert_inputs['attention_mask']])
    predictions = model.predict([input_ids, attention_mask])
    predicted_index = np.argmax(predictions)

    return Ind2Label[predicted_index], np.max(predictions)


# for custom CSS style
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.header('Text-classifier App')
with st.form(key='my_form', clear_on_submit=False):
    text = st.text_area(label='Enter your query')
    pressed = st.form_submit_button(label='submit')

    if pressed:
        if text != '':
            prediction, score = predict(model, text)
            st.success(
                f"Prediction is: {prediction}\t|\tscore: {score*100:.2f}")
        else:
            st.error(f"Please, enter a query")
