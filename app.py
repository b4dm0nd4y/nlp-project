import streamlit as st

# Hide Streamlit's default pages navigation
st.set_page_config(
    page_title='Проект по NLP от Нияза, Олега и Сергея',
    layout='wide',
    initial_sidebar_state='auto'
)

hide_default_nav = """
<style>
/* hide auto-generated pages nav */
[data-testid="stSidebarNav"] {display: none;}
</style>
"""
st.markdown(hide_default_nav, unsafe_allow_html=True)


import sys
sys.path.append("./models")
sys.path.append('./notebooks')
sys.path.append("./src")
from src.helper import predict_string_with_ml


from pages.hospital import hospital_page
from pages.hospital_graphs import hospital_graph_page
from pages.toxic import toxic_page
from pages.toxic_graphs import toxic_graph_page
from pages.humor import humor_page

PAGES = {
    'Отзывы на поликлиники': hospital_page,
    'Графики отзывов': hospital_graph_page,
    'Оценка токсичности': toxic_page,
    'Графики токсичности': toxic_graph_page,
    'Шутка минутка': humor_page
}

def main():
    st.sidebar.title('Наши страницы')
    selection = st.sidebar.radio('Перейти на', list(PAGES.keys()))
    page = PAGES[selection]
    page()
    
if __name__ == '__main__':
    main()