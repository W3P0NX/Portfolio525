import spacy
print(f"spaCy version: {spacy.__version__}")
import pandas as pd
print(f"pandas version: {pd.__version__}")
import nltk
print(f"nltk version: {nltk.__version__}")
import kagglehub
print(f"kagglehub version: {kagglehub.__version__}")
import streamlit as st
import sklearn
import pickle
import ssl
import logging
import os

# from tqdm import tqdm
# tqdm.pandas()

from huggingface_hub import hf_hub_download

from spacy.language import Language
from spacy_langdetect import LanguageDetector

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_path = 'input'
output_path = 'output'

if not os.path.exists(input_path):
    os.mkdir(input_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)

model_file_path = os.path.join(output_path, 'classifier_model.pkl')
hug_model_path = hf_hub_download(repo_id="W3P0NX/ModelTest", filename="classifier_model.pkl")

@st.cache_resource
def load_classifier_model(a_file):
        print("Loading model from Hugging Face")
        return pickle.load(a_file)

# # Project Title Page
# st.title("CSC525 Portfolio Project")
#
# st.write("""
# ## American Airlines Twitter - NLP Chatbot
# This Chatbot was developed John Andrade, using the American Airlines Twitter Data from Kaggle.
#
# """)

if __name__ == "__main__":

    # Project Title Page
    st.title("CSC525 Portfolio Project")

    st.write("""
    ## American Airlines Twitter - NLP Chatbot
    This Chatbot was developed John Andrade, using the American Airlines Twitter Data from Kaggle.
    
    """)

    # Loading of Classifier Model
    with open(hug_model_path, 'rb') as f:
        st.write("""
        Please wait while data is downloaded.
        
        """)
        model = load_classifier_model(f)
        st.write("""
        Model download done.
        """)

    print(f"Loading....DONE")

    # Chat Session Stream Creation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is going on?: "):

        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate model response
        response = model.predict([prompt])[0]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
