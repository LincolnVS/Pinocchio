import streamlit as st
from fastai.vision.all import *


def fake_or_real(model,image,accuracy = .99):

    predict = model.predict(image)
    real,fake = float(predict[2][0]),float(predict[2][1])

    if real >= acuracia:
        string_resposta = f"Temos {round(real*100,2)}% de certeza que essa imagem é real!"
    elif fake >= acuracia:
        string_resposta = f"Temos {round(fake*100,2)}% de certeza que essa imagem é fake!"
    else:
        string_resposta = f"Não temos precisão suficiente para informar se essa imagem é fake ou não! real = {round(fake*100,2)}; Fake {round(real*100,2)}"

    return string_resposta


model = load_learner('./model.pkl')
image = get_image_files('data/external-content.duckduckgo.com.jpeg')
print(image)
predict = fake_or_real(model,image)


st.title('Deep-fake detection')
