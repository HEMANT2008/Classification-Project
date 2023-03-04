import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import base64
import pickle
from streamlit_lottie import st_lottie
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Liver Diseaase Prediction",layout="wide",initial_sidebar_state="collapsed")
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('img.png')

def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
side_bg = 'img_2.png'
sidebar_bg(side_bg)


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

st.header('Liver Disease Prediction')
patients=pd.read_csv('pre-processed_liver_disease.csv')

X=patients.iloc[:, patients.columns != "category"]
y = patients['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

LD = pickle.load(open('LDP_model.pkl','rb'))

menu = ['About','Dataset','Test your Sample','Test From Dataset']
chart = st.sidebar.radio("SELECT THE OPTION:-", menu)

if chart == 'About':
    col1,col2 = st.columns([2, 1])
    with col1:
     st.subheader('Business Objective :-')
     st.write('The variable to be predicted is categorical (no disease,suspect disease,hepatitis,fiborsis,cirrhosis)')
     st.write("""P-198 Liver Disease Prediction  :


                       1.Mr.Omprasad Kolhal
        2.Ms.Varsha Reddy
        3.Mr.Hemant Shinde
        4.Mr.Karan Patil
        5.Ms.Madhuri Varpe
        6.Mr.Junaid Pasha
               """)
    with col2:
        lottie = load_lottiefile('53911-online-medical-assistance-animation.json')
        st_lottie(lottie, key="Rating")
if chart == 'Dataset':
        st.write(patients)
        data = patients.shape
        st.success('Shape of Dataset :- {}'.format(data))



        #st.wr(data)

if chart == 'Test your Sample':
    col1, col2, col3 = st.columns(3)
    with col1:
        a = st.number_input('Enter your Age:',)
    with col2:
        bbb = ['1', '0']
        b = st.selectbox('Select Gender(Select 1 for male and 0 for Female):', bbb)
    with col3:
        c = st.number_input('Enter your albumin Count:',)
    col4, col5, col6 = st.columns(3)
    with col4:
        d = st.number_input('Enter your alkaline_phosphatase Count:',)
    with col5:
        e = st.number_input('Enter your alanine_aminotransferase Count:',)
    with col6:
        f = st.number_input('Enter your aspartate_aminotransferase Count:',)
    col7, col8, col9 = st.columns(3)
    with col7:
        g = st.number_input('Enter your bilirubin Count:', )
    with col8:
        h = st.number_input('Enter your cholinesterase Count:',)
    with col9:
        i = st.number_input('Enter your cholesterol Count:',)
    col10, col11, col12 = st.columns(3)
    with col10:
        j = st.number_input('Enter your creatinina Count:', )
    with col11:
        k = st.number_input('Enter your gamma_glutamyl_transferase Count:',)
    with col12:
        l = st.number_input('Enter your protien Count:', )

    z = (a, b, c, d, e, f, g, h, i, j, k, l)
    zz = [z]

    if st.button('predict'):
        st.subheader('Your Test Result:-')
        st.subheader(LD.predict(zz))
        #st.success('Your Test Result :- {}'.format())
if chart == 'Test From Dataset':
    zzz = st.number_input('Enter Row Number')
    if st.button('Predict'):
     zzzz = X.iloc[[zzz]]
     st.subheader('Test Result:-')
     st.subheader(LD.predict(zzzz))
     st.write(zzzz)

