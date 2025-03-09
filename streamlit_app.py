import streamlit as st
import sklearn
print(f"sklearn version: {sklearn.__version__}")
import pickle
import logging
import os

from huggingface_hub import hf_hub_download

# Logging Level
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Hugging Face Model Repository
hug_model_path = hf_hub_download(repo_id="W3P0NX/ModelTest", filename="classifier_model.pkl")

# Method to load Classifier Model
@st.cache_resource
def load_classifier_model(a_file):
        print("Loading model from Hugging Face")
        return pickle.load(a_file)


if __name__ == "__main__":

    # Source Data Citation
    twitter_data_url = "https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter"

    # Project Title Page
    st.title("CSC525 Portfolio Project")

    st.write("""
    ## American Airlines Twitter - NLP Chatbot
    This Chatbot was developed John Andrade using the American Airlines [Twitter Data from Kaggle](%s).
    
    """ % twitter_data_url)

    # Loading of Classifier Model
    with open(hug_model_path, 'rb') as f:
        st.write("""
        Please wait while model is downloaded.
        
        """)
        model = load_classifier_model(f)
        st.write("""
        Model download done.
        """)

    print(f"Loading....DONE!")

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