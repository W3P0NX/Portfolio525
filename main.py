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
import ssl
import logging
import os

import variables

from tqdm import tqdm
tqdm.pandas()

from spacy.language import Language
from spacy_langdetect import LanguageDetector

# from spacy.cli import download
# download("en_core_web_sm")
#
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# ", ".join(stopwords.words('english'))

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

@st.cache_data
def load_data():
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

st.title("NLP Chatbot")

st.write("""
# American Airlines Twitter
Which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
st.write(dataset_name)

if __name__ == "__main__":
    load_data()
    print(f"{dataset_name} Loading....DONE")