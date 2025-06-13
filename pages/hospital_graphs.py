import streamlit as st
import matplotlib.pyplot as plt


import sys
sys.path.append("../images")
sys.path.append("../src")

MODELS = {
    'Логистическая Регрессия': 1,
    'БЕРТ': 2,
    'LSTM': 3
}

def hospital_graph_page():
    st.title('Графики моделей классификации отзывов о работе поликлиники')
    model_name = st.selectbox('Выбери свою модель:', list(MODELS.keys()))
    
    if model_name == 'Логистическая Регрессия':
        st.header("Метрика логистической регресии + tfidf")
        st.image("./images/lr_tfidf_metrics.png", use_container_width=True)
    elif model_name == 'БЕРТ':
        st.header("Метрика БЕРТА")
        st.image("./images/bert_metrics.png", use_container_width=True)
    elif model_name == 'LSTM':
        st.header("Метрика LSTM + Bagdanau + Word2Vec")
        st.image("./images/lstm_acc_f1.png", use_container_width=True)
        st.image("./images/lstm_metrics.png", use_container_width=True)