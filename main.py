import spacy
import streamlit as st
import matplotlib.pyplot as plt

st.title("Streamlit Example")

st.write("""
# Explore Different Classifier
Which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
st.write(dataset_name)

if __name__ == "__main__":
    print(f"{dataset_name} Loading....DONE")