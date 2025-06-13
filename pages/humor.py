import streamlit as st
import numpy as np
import textwrap
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import sys
sys.path.append("../models")
sys.path.append("../src")

tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
model_finetuned = GPT2LMHeadModel.from_pretrained(
    'sberbank-ai/rugpt3small_based_on_gpt2',
    output_attentions = False,
    output_hidden_states = False,
)
DEVICE = 'cpu'

WEIGHTS = {
    'Шутка Юмора': './models/humor_gpt.pt',
    'Шутка Юмора ПЛЮС': './models/better_humor_gpt.pt',
    'Туманное Фэнтези': './models/fantasy_gpt.pt',
}

def humor_page():
    st.title("Шутка-Минутка. Генератор Анекдотов!")
    weight_name = st.selectbox('Выбери свою модель:', list(WEIGHTS.keys()))
    weights = WEIGHTS[weight_name]
    
    model_finetuned.load_state_dict(torch.load(weights, map_location=DEVICE))
    
    prompt = st.text_area('Введите свой промт:')
    prompt = tokenizer(prompt, return_tensors='pt')
    masks = prompt.attention_mask
    prompt = prompt.input_ids
    
    temperature = st.slider('Температура', 1.0, 10.0, 1.0, 0.1)
    max_length = st.number_input(
        'Кол-во Токенов', min_value=10, max_value=1000, value=10
    )
    n_gen = st.number_input(
        'Кол-во генераций', min_value=1, max_value=5, value=1
    )
    sequence = st.selectbox(
        'Алгоритм Последовательности', ['Beam Search','Top-p','Greedy']
    )
    if st.button('Начинаем балаган'):
        for i in range(n_gen):
            params = {
                'input_ids': prompt,
                'attention_mask': masks,
                'temperature': temperature,
                'max_length': max_length,
                'num_return_sequences':1,
            }
            
            if sequence == 'Top-p':
                params['top_p'] = 0.9
            elif sequence == 'Beam Search':
                params['top_p'] = 0.95
                params['num_beams'] = 5
            elif sequence == 'Greedy':
                params['temperature'] = 1e-5
                params['top_p'] = 1.0
                
            out = model_finetuned.generate(
                do_sample=True,
                top_k=100,
                # сколько (постараться) не повторять n_gram подряд
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **params,
            ).numpy()
            
            if weight_name in ['Шутка Юмора', 'Шутка Юмора ПЛЮС'] :
                text = tokenizer.decode(out[0], skip_special_tokens=True)
                joke = text.strip().split("\n")[0]
                st.write(textwrap.fill(joke, 80))

            elif weight_name == 'Туманное Фэнтези': 
                for output in out:
                    text = tokenizer.decode(output, skip_special_tokens=True)
                    st.write(text)
            
            # response = openai.Completion.create(**params)
            # text = response.choices[0].text.strip()
            