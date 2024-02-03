import os

from langchain.llms import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
OPENAI_API_KEY_VAR = os.getenv("OPENAI_API_KEY")
TEMPERATURE = 0.5
model_name = "gpt-3.5-turbo"


def get_openai_response(question):
    llm = OpenAI(openai_api_key=OPENAI_API_KEY_VAR, temperature=TEMPERATURE)
    response = llm.predict(question)
    return response


st.set_page_config(page_title="Q&A Demo")

st.header("Langchain Application")

input = st.text_input("Input: ", key="input")
response = get_openai_response(input)

submit = st.button("Ask the question")

## If ask button is clicked

if submit:
    st.subheader("The Response is")
    st.write(response)
