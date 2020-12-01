import streamlit as st
from fastai.vision.all import *
from tempfile import NamedTemporaryFile
import tkinter as tk
from tkinter import filedialog
#from keras.preprocessing.image import load_img


def label_func(f):
  if (f[0:4] == "real" or f[1:5] == "real" or f[2:6] == "real"):
      return False
  else:
    return True

def fake_or_real(model,image,accuracy = .99):

    predict = model.predict(image)
    real,fake = float(predict[2][0]),float(predict[2][1])

    if real >= accuracy:
        string_resposta = f"Temos {round(real*100,2)}% de certeza que essa imagem é real!"
    elif fake >= accuracy:
        string_resposta = f"Temos {round(fake*100,2)}% de certeza que essa imagem é fake!"
    else:
        string_resposta = f"Não temos precisão suficiente para informar se essa imagem é fake ou não! real = {round(fake*100,2)}; Fake {round(real*100,2)}"

    return string_resposta


uploaded_files = st.file_uploader("Select images", accept_multiple_files=True)
model = load_learner('./model.pkl')
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    response = fake_or_real(model,bytes_data)
    st.image(bytes_data, caption=response, use_column_width=True)

    #st.write(response)



