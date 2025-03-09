import spacy
print(f"spaCy version: {spacy.__version__}")
import pandas as pd
print(f"pandas version: {pd.__version__}")
import nltk
print(f"nltk version: {nltk.__version__}")
import matplotlib.pyplot as plt
import kagglehub
print(f"kagglehub version: {kagglehub.__version__}")
import streamlit as st
import sklearn
import pickle
import ssl
import logging
import os

from tqdm import tqdm
tqdm.pandas()

from huggingface_hub import hf_hub_download

from spacy.language import Language
from spacy_langdetect import LanguageDetector

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()

input_path = 'input'
output_path = 'output'

if not os.path.exists(input_path):
    os.mkdir(input_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)

model_file_path = os.path.join(output_path, 'classifier_model.pkl')
hug_model_path = hf_hub_download(repo_id="W3P0NX/ModelTest", filename="classifier_model.pkl")

@st.cache_data
def download_data():
    from spacy.cli import download
    download("en_core_web_sm")

    from nltk.corpus import stopwords
    nltk.download('stopwords')
    ", ".join(stopwords.words('english'))

    # Check for Model
    if not os.path.exists(model_file_path):
        # Create context to allow for access to URL without an SSL Certificate
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        local_file_path = "input/twcs.csv"
        if os.path.exists(local_file_path):
            data_file = local_file_path
            print("Path to dataset files:", input_path)
        else:
            path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
            print("Path to dataset files:", path)
            data_file = (os.path.join(path, 'twcs/twcs.csv'))

        df_all_data = pd.read_csv(data_file)
        df_all_data.to_csv(os.path.join(input_path, 'twcs.csv'), index=False)

        st.header("Data Statistics")
        st.write(df_all_data.describe())

        return df_all_data

@st.cache_data
def filter_aa_tweets(a_dataframe):
    a_dataframe['text'] = a_dataframe['text'].astype('string')
    a_dataframe['author_id'] = a_dataframe['author_id'].astype('string')

    first_inbound = a_dataframe[pd.isnull(a_dataframe.in_response_to_tweet_id) & a_dataframe.inbound]
    df = pd.merge(first_inbound, a_dataframe, left_on='tweet_id',
                  right_on='in_response_to_tweet_id')
    df['in_response_to_tweet_id_y'] = df['in_response_to_tweet_id_y'].astype('int64')

    questions_responses = df[["author_id_x", "created_at_x", "text_x", "author_id_y", "created_at_y", "text_y"]]
    aa = questions_responses[questions_responses["author_id_y"] == "AmericanAir"]

    st.header("Data Header")
    st.write(df.describe())

    return aa

@st.cache_resource
def load_classifier_model(a_file):
        print("Loading model from Hugging Face")
        return pickle.load(a_file)

def get_intent(text):
    return model.predict([text])[0]

def chatbot():
    print("\nWelcome to American Airlines Twitter Chat.  How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Thank you for flying American Airlines! Goodbye.")
            break
        intent = get_intent(user_input)
        print(f"Chatbot: {intent}\n")

st.title("Portfolio Project C525")

st.write("""
## American Airlines Twitter - NLP Chatbot
This Chatbot was developed using the American Airlines Twitter Data from Kaggle.  
Please wait while data is downloaded.
""")

if __name__ == "__main__":
    # df_all = download_data()
    # model = load_classifier_model()

    # @st.cache_data
    # @st.cache_data
    with open(hug_model_path, 'rb') as f:
        # print("Loading model from Hugging Face")
        model = load_classifier_model(f)
        # model = pickle.load(f)
        st.write("""
        Model download done
        """)

    print(f"Loading....DONE")

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

        response = model.predict([prompt])[0]
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
