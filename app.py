import streamlit as st
from fastai.vision.all import *
from tempfile import NamedTemporaryFile
import tkinter as tk
from tkinter import filedialog
#from keras.preprocessing.image import load_img

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_model():
    return load_learner('./model.pkl')

def label_func(f):
  if (f[0:4] == "real" or f[1:5] == "real" or f[2:6] == "real"):
      return False
  else:
    return True
    
def fake_or_real(model,image,accuracy = .99):
    predict = model.predict(image)
    real,fake = float(predict[2][0]),float(predict[2][1])
    if real >= accuracy:
        string_resposta = f"Temos {round(real*100,2)}% de certeza que essa imagem é real! \n \n"
    elif fake >= accuracy:
        string_resposta = f"Temos {round(fake*100,2)}% de certeza que essa imagem é fake! \n \n"
    else:
        string_resposta = f"Sem acurácia suficiente para dizer se é fake ou não!"
    return string_resposta


uploaded_files = st.file_uploader("Select images", accept_multiple_files=True)
model = load_model()
acertividade_num = st.slider('Acurácia necessária para predição? (em porcentagem)', 80.0, 100.0, 99.0)


cols = st.beta_columns(3)
for item,uploaded_file in enumerate(uploaded_files):
    bytes_data = uploaded_file.read()
    response = fake_or_real(model,bytes_data,accuracy=acertividade_num/100)
    cols[item%3].image(bytes_data, caption=response,width = 226)

