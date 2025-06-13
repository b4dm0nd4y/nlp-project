import streamlit as st
import matplotlib.pyplot as plt
import time

import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

import sys
sys.path.append("../models")
sys.path.append("../src")



tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny-toxicity")
bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny-toxicity")

# 2. создать такую же модель
class ToxicClassifier(nn.Module):
    def __init__(self, model, hidden_dim=256):
        super().__init__()
        self.bert = model  # встроенный BERT
        self.classifier = nn.Sequential(
        nn.Linear(self.bert.config.hidden_size, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # НЕ обучаем BERT, только классификатор
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_vector = outputs.last_hidden_state[:, 0, :]  # берём [CLS]-вектор
        return self.classifier(cls_vector).squeeze(1)  # выход — один логит на пример

# 3. загрузить веса
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert = ToxicClassifier(bert_model).to(device)
bert.load_state_dict(torch.load("./models/toxic_bert.pt", map_location=device))
bert.eval()

def predict_toxic(text, model):
    encoded = tokenizer(
        text,
        max_length=50,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prob = output.item()
    
    return prob, prob >= 0.55



MODELS = {
    'Берта': bert
}

def toxic_page():
    st.title("Оценка степени токсичности пользовательского сообщения")
    model_name = st.selectbox('Выбери свою модель', list(MODELS.keys()))
    model = MODELS[model_name]
    
    comment = st.text_area('Введите свое сообщение:')
    if st.button('Начать оценку'):
        start = time.time()
        
        pred = predict_toxic(comment, model)
        
        time_needed = time.time() - start
        st.write(f'Модели {model_name} потребовалось: {time_needed:.3f} секунд')
        st.write(pred)  # placeholder for prediction