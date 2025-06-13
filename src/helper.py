import re
import numpy as np
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
from nltk.corpus import stopwords

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from ltsm_helper import tokenize_text

RUBERT_OUT = 312
SEQ_LEN = 512

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

stop = set(stopwords.words('russian')) - {'не', 'ни'}
tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')

def preprocess_string(string):
    if not isinstance(string, str) or len(string.strip()) == 0: # fool-protection
        return ''
    
    string = clean_string(string)
    string = lemmantize_words(string)
    string = filter_stop_words(string)
    
    return string


def filter_stop_words(text):
    words = []
    for word in text.split():
        if word in stop or len(word) < 2:
            continue
        words.append(word)
        
    return ' '.join(words)


def lemmantize_words(text):
    lemmas = []
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    try:
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
            lemmas.append(token.lemma)
                
    except Exception as e:
        print(f"Ошибка при обработке текста: {text[:30]}... → {e}")
        return ''
    
    return ' '.join(lemmas)


def clean_string(text):
    text = re.sub(r'<.*?>', ' ', text) # removing html-tags from text
    text = re.sub(r'http\S+|\S+@\S+', ' ', text) # removing links
    text = re.sub(r'[^а-яё\s]', ' ', text.lower()) # remove all non-letter symbols
    cleaned_text = re.sub(r'\s+', ' ', text).strip() # remove double or more spaces
    
    return cleaned_text

def predict_string_with_ml(model, model_name, vectorizer, text):
    decoder = {
        0:'negative',
        1:'positive'
    }
    
    cleaned = preprocess_string(text)
    
    
    if model_name == 'Логистическая Регрессия':
            X_input = vectorizer.transform([cleaned])
            pred = model.predict(X_input)
            pred = pred[0]
            
    elif model_name == 'БЕРТ':
        model.eval()
        encoding = tokenizer(
            cleaned,
            padding='max_length',
            truncation=True,
            max_length=SEQ_LEN,
            return_tensors='pt'
        )
        pred = model(encoding)
        pred = pred.argmax().item()
        
    elif model_name == 'LSTM':
        model.eval()
        text = tokenize_text(cleaned)
        pred = model(torch.tensor(text, dtype=torch.long))[0].item()
        pred = 1 / (1 + np.exp(-pred))
        pred = int(pred > 0.5)
        
    
    return decoder[pred]



class MyTinyBERT(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            'cointegrated/rubert-tiny2'
        )
        
        for name, param in self.bert.named_parameters():
            if 'encoder.layer.10' in name or 'encoder.layer.11' in name:
                param.requires_grad=True
            else:
                param.requires_grad=False
                
        self.norm = nn.LayerNorm(RUBERT_OUT)
        self.linear = nn.Sequential(
            nn.Linear(RUBERT_OUT, RUBERT_OUT),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(RUBERT_OUT, RUBERT_OUT//2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(RUBERT_OUT//2, n_classes)
        )
        
    def mean_pooling(self, model_output, attention_mask):
        token_embedings = model_output.last_hidden_state # [b:t:h]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embedings.size()
        )
        sum_embeddings = torch.sum(
            token_embedings * input_mask_expanded, dim=1
        )
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def forward(self, x):
        bert_out = self.bert(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask']
        )
        
        pooled=self.mean_pooling(bert_out, x['attention_mask'])
        pooled=self.norm(pooled)
        out=self.linear(pooled)
        
        return out
