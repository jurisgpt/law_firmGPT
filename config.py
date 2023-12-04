import streamlit as st
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# openAI
openai.api_key = st.secrets.get("OPENAI_API_KAY", "")

# anthropic
anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
anthropic = Anthropic(api_key=anthropic_api_key)

# assemblyAI
assemeblyAI_api_key = st.secrets.get("ASSEMBLYAI_API_KAY", "")