import streamlit as st
import matplotlib.pyplot as plt
import torch
import joblib
import time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append("../models")
sys.path.append("../src")
from helper import predict_string_with_ml, MyTinyBERT
from ltsm_helper import LSTMClassifier, BahdanauAttention

logreg = joblib.load('./models/logreg_model.joblib')
vectorizer = joblib.load('./models/tfidf_vectorizer.joblib')

@st.cache_resource  # This ensures the model is loaded only once
def load_bert():
    model = MyTinyBERT(2)
    model.load_state_dict(torch.load('./models/bert_best.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource  # This ensures the model is loaded only once
def load_lstm():
    model = LSTMClassifier()
    model.load_state_dict(torch.load('./models/lstm_weights.pth', map_location=torch.device('cpu')))
    # model = torch.load(torch.load('./models/lstm_entire.pt', weights_only=False, map_location=torch.device('cpu')))
    model.eval()
    return model

bert_cls = load_bert()
lstm = load_lstm()

MODELS = {
    'Логистическая Регрессия': logreg,
    'БЕРТ': bert_cls,
    'LSTM': lstm
}

def hospital_page():
    st.title('Классификация отзыва о работе поликлиники')
    model_name = st.selectbox('Выбери свою модель:', list(MODELS.keys()))
    model = MODELS[model_name]
    
    feedback = st.text_area('Оставте свой отзыв:')
    if st.button('Оценить'):
        start = time.time()
        
        pred = predict_string_with_ml(model, model_name,vectorizer, feedback)
        
        time_needed = time.time() - start
        st.write(
            f'Модель: {model_name}, потребовалось: {time_needed:.3f} секунд'
        )
        st.write(f'Это {pred} отзыв')