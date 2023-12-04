import streamlit as st
import numpy as np
import time

st.title("ICy's chatbot")

def get_response():
    with st.chat_message("user"):
        st.write(f"{prompt}" )

prompt = st.chat_input("Say something", on_submit=get_response())
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")
