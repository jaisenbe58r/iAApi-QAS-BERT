import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from PIL import Image
import io
import logging


# interact with FastAPI endpoint
backend = 'http://127.0.0.1:8008/qas/'
# backend = 'http://localhost:8000/qas/'


def process(context: str, question: str, server_url: str):

    m = MultipartEncoder(
        fields={'context': context, 'question': question}
        )
    r = requests.post(server_url,
                      data=m,
                      params=m.fields,
                      headers={'Content-Type': m.content_type},
                      timeout=8000)
    print(r.url)

    return r


# construct UI layout
st.title('Question Answering')

st.write('''Question Answering.
         Visit this URL at `:8008/docs` for FastAPI documentation.''')  # description and instructions

user_input_context = st.text_area("Context:")
user_input_question = st.text_area("Question:")

if st.button('Get Answering'):

    if user_input_context and user_input_question:
        result = process(user_input_context, user_input_question, backend)
        st.write(result.content)

    elif user_input_context:
        # handle case with no image
        st.write("Insert question!")

    elif user_input_context:
        # handle case with no image
        st.write("Insert context!")

    else:
        # handle case with no image
        st.write("Insert context and question!")
