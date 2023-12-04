import streamlit as st
import time
import json
from pdf2image import convert_from_bytes
import base64, io
from google.cloud import vision
from google.cloud import storage
from annotated_text import annotated_text
import chatbots
from chatbots.claude import Claude, models, ASK_TEMPLATE
from chatbots.chatgpt import ChatGPT, models, ASK_TEMPLATE
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import *
from functools import reduce

# Set up the storage client using the JSON key file
bucket_name = 'juris-tsunzu-gpt-bucket'
blob_name = 'user-uploaded-files/'

storage_client = storage.Client.from_service_account_json('/root/workspace/gkey/juris-gpt-key.json')
vision_client = vision.ImageAnnotatorClient()

bucket = storage_client.bucket(bucket_name)

# Configures
st.set_page_config(layout="wide")
st.title("⚖️ 3Bugs' Law Firm Chatbot")

# Sidebar
st.sidebar.title("Settings")

llms = ["Claude", "ChatGPT"]
llm_type = st.sidebar.selectbox("LLM", llms)
if llm_type == "Claude":
    model = st.sidebar.selectbox("Model", chatbots.claude.models)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    system_prompt = st.sidebar.text_area("System Prompt", value=chatbots.claude.ASK_TEMPLATE)
    stream = st.sidebar.checkbox("Stream", value=True)
elif llm_type == "ChatGPT":
    model = st.sidebar.selectbox("Model", chatbots.chatgpt.models)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    system_prompt = st.sidebar.text_area("System Prompt", value=chatbots.chatgpt.ASK_TEMPLATE)
    stream = st.sidebar.checkbox("Stream", value=True)


# Main
# button: creat bot
# after click button, initial
def initialization():
    """
    Initialize the bot.
    """
    if llm_type == "Claude":
        st.session_state.bot = Claude(model=model, temperature=temperature, system_prompt=system_prompt, stream=stream)
    elif llm_type == "ChatGPT":
        st.session_state.bot = ChatGPT(model=model, temperature=temperature, system_prompt=system_prompt, stream=stream)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "text_map_prompt" not in st.session_state:
        st.session_state.text_map_prompt = ""
    if "text_reduce_prompt" not in st.session_state:
        st.session_state.text_reduce_prompt = ""
    if "full_response" not in st.session_state:
        st.session_state.full_response = ""
    if "st.session_state.uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

create_bot = st.sidebar.button("Create Bot", on_click=initialization)

# Function to clear the messages and bot object
def clear_session():
    """
    Clear the session.
    """
    st.session_state["uploaded_files"] = []
    st.session_state["messages"] = []
    st.session_state["bot"] = None
    st.session_state["text_map_prompt"] = ""
    st.session_state["full_response"] = ""
    
# Button: Clear
clear_button = st.sidebar.button("Clear", on_click=clear_session)

# restart = clear and initialize
def restart():
    clear_session()
    initialization()
    render()
# button restart
restart_button = st.sidebar.button("Restart", on_click=restart)

# Button: save history messages
def save_history():
    """
    Save the history messages and provide a download link.
    """
    if not st.session_state.messages:
        st.info("No Messages", icon="ℹ️")
        st.stop()

    # Generate save title
    save_title = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    messages_json = json.dumps(st.session_state["messages"])

    # Save the JSON string to a file
    filename = f"messages_{save_title}.json"
    with open(filename, "w") as file:
        file.write(messages_json)
    
    # Provide download link
    file_link = f"[Download History](./{filename})"
    st.markdown(file_link, unsafe_allow_html=True)
# button
download_button = st.sidebar.button("Download History", on_click=save_history)

if "bot" not in st.session_state:
    initialization()
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.get("bot") is None:
    st.info("Please create the bot first.")

"""
Welcome to 3Bugs' law firm chatbot PoC!
"""

# render messages
def render():
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"]=="user" else "🤖"):
            st.markdown(message["content"])
render()


# image to text using Google Vision API
def detect_labels(img_byte_arr, i):
    """Detects labels in the image file."""
    image = vision.Image(content=img_byte_arr)

    response = vision_client.text_detection(image=image)
    texts = response.text_annotations

    result = "\n_________________{}__________________\n".format(i)

    for text in texts[1:]:
        # print('\n"{}"'.format(text.description))
        result += text.description

    return result


def analyze_filelist(file_list):
    audios = []
    pdfs = []
    pics = []
    for filename in file_list:
        if filename.name.lower().endswith('.pdf'):
            print(f'{filename}: PDF')
            pdfs.append(filename)
        elif filename.name.lower().endswith(('.mp3', '.wav', '.ogg')):
            print(f'{filename}: Audio')
            audios.append(filename)
        elif filename.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            print(f'{filename}: Image')
            pics.append(filename)
        else:
            print(f'{filename}: Unknown')

    return {'audio': audios, 'pdf': pdfs, 'pics': pics}


disable_input = st.session_state.bot is None

st.header("Step 1: Please upload pdf files")

st.session_state.uploaded_files = st.file_uploader("Choose PDF files you want to upload", accept_multiple_files=True)

st.header("Step 2: Enter in your Map prompt")


st.session_state.text_map_input = st.text_area(
        "Map Reduce breaks your document into chunks then first runs the Map Prompt on each Chunk.\n\n Please include exactly one {text} token in your prompt.",
        "You are a research analyst. I will provide you with a section of a document and you will create a summary from it. You will use your editing and writing skills to create a summary in the style of a Confidential Information Memorandum. You will preserve as many details as possible. You will maintain context across the summary. Your section will be combined with the other sections to create summary of the entire document.\n\n Your summary must be no longer than 650 characters long. \n\n Input: {text} \n\n Output:",
        height=300
    )


st.header("Step 3: Enter in your Reduce prompt")


st.session_state.text_reduce_input = st.text_area(
        "Map Reduce then runs reduce on all the summarized chunks until the summary is 4,000 characters long. The current summary length is a hard limit based on the algorithm. Use Map Only for longer summaries.\n\nPlease include exactly one {text} token in your prompt.",
        "You are a copyeditor. Combine the below summaries. The combined output must be less than 4,000 characters long. You must keep the content and context preserved. \n\nInput: {text} \n\nOutput:",
        height=300
    )


def generate():
    print('analyzing the file list...')

    files = analyze_filelist(st.session_state.uploaded_files)
    pdf_source = files['pdf']
    audio_source = files['audio']
    pic_source = files['pics']
    docs = []

    ## parse pdf files
    
    if pdf_source is not []:
        st.toast("Parsing pdf files...")
        extracted_data = {}
        for pdf_id in range(len(pdf_source)):
            base64_pdf = base64.b64encode(pdf_source[pdf_id].getvalue()).decode('utf-8')
            pdf_source[pdf_id].seek(0)
            print("_____________\n")
            print(f"{blob_name}PDF/{pdf_source[pdf_id].name}")
            blob = bucket.blob(f"{blob_name}PDF/{pdf_source[pdf_id].name}")
            blob.upload_from_string(pdf_source[pdf_id].getvalue())
            images = convert_from_bytes(pdf_source[pdf_id].getvalue())
            extracted_data[pdf_source[pdf_id].name] = ""
            for i, image in enumerate(images):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                extracted_data[pdf_source[pdf_id].name] += detect_labels(img_byte_arr, i)
        print("Extracted data:", extracted_data)

        map_result = ""
        txt_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 160000,
                chunk_overlap  = 200
            )

        for pdf_idx in extracted_data:
            docs += txt_splitter.create_documents([extracted_data[pdf_idx]])

        print("\n\n_________________________________________\n\n", docs)

    else:
        st.toast("No Pdf files detected...skipping")


    ## parse audio files

    if audio_source is not []:
        st.toast("Parsing audio files...")
        extracted_data = {}
        for audio_id in range(len(audio_source)):
            base64_audio = base64.b64encode(audio_source[audio_id].getvalue()).decode('utf-8')
            audio_source[audio_id].seek(0)
            audios = convert_from_bytes(audio_source[audio_id].getvalue())
            extracted_data[audio_source[audio_id].name] = ""
            for idx, audio in enumerate(audio_source):
                audio_byte_arr = io.BytesIO()
                audio.save(img_byte_arr, format='.mp3')
                audio_byte_arr = audio_byte_arr.getvalue()
    else:
        st.toast("No audio source detected...skipping...")

    ## parse pic files

    if pic_source is not []:
        st.toast("Parsing pictures...")
    else:
        st.toast("No picture detected...skipping...")

    st.session_state.full_response = ""
    st.toast("Generating result...")
    st.toast("Mapping...")
    map_results = []
    for doc in docs:
        full_response = ""
        for response in st.session_state.bot.ask_llm(st.session_state.text_map_input.replace("{text}", doc.page_content), stream=True):
            full_response += response
        map_results.append(full_response)

    st.toast("Reducing...")
    reduce_result = ""
    for response in st.session_state.bot.ask_llm(st.session_state.text_reduce_input.replace("{text}", reduce(lambda x, y: x+y, map_results)), stream=True):
            reduce_result += response
    st.session_state.full_response = reduce_result

st.button("Generate", on_click=generate)


st.header("Result")


with st.chat_message("assistance", avatar = "🤖"):
    message_placeholder = st.empty()
    message_placeholder.write(st.session_state.full_response)
