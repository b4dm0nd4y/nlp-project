import streamlit as st
import matplotlib.pyplot as plt


import sys
sys.path.append("../images")
sys.path.append("../src")

def toxic_graph_page():
    st.title('Графики модели касательно Таксичности отзывов')
    
    st.header("Метрики")
    st.image("./images/toxic_conf.jpg", use_container_width=True)
    st.image("./images/toxic_f1.jpg", use_container_width=True)
    st.image("./images/toxic_top.jpg", use_container_width=True)
    st.image("./images/toxic_top_min.jpg", use_container_width=True)
    st.image("./images/toxic_compare.jpg", use_container_width=True)