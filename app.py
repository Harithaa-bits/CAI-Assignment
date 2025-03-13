import os
import urllib.request
import zipfile
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths
NLTK_ZIP_URL = "https://github.com/Harithaa-bits/CAI-Assignment/raw/main/nltk_data_min.zip"
NLTK_ZIP_PATH = "nltk_data_min.zip"
NLTK_DATA_PATH = "nltk_data"

# Download and extract NLTK data if not present
if not os.path.exists(NLTK_DATA_PATH):
    st.write("Downloading and extracting NLTK data...")
    urllib.request.urlretrieve(NLTK_ZIP_URL, NLTK_ZIP_PATH)
    
    with zipfile.ZipFile(NLTK_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(NLTK_DATA_PATH)

    st.write("NLTK data successfully downloaded and extracted.")

# Set NLTK data path explicitly
nltk.data.path.append(NLTK_DATA_PATH)

# Verify punkt exists
if not os.path.exists(os.path.join(NLTK_DATA_PATH, "tokenizers", "punkt")):
    st.error("Missing NLTK punkt tokenizer. Extraction failed.")

# Load stopwords
nltk.download("stopwords", download_dir=NLTK_DATA_PATH)
stop_words = set(stopwords.words("english"))

