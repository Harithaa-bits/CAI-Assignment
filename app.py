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

# ------------------ Fix: Download and Extract NLTK Data ------------------
NLTK_ZIP_URL = "https://raw.githubusercontent.com/Harithaa-bits/CAI-Assignment/main/nltk_data_min.zip"
NLTK_ZIP_PATH = "nltk_data_min.zip"
NLTK_DATA_DIR = "nltk_data"

# Ensure NLTK data is available
if not os.path.exists(NLTK_DATA_DIR):
    st.write("Downloading and extracting NLTK data...")
    try:
        urllib.request.urlretrieve(NLTK_ZIP_URL, NLTK_ZIP_PATH)
        with zipfile.ZipFile(NLTK_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(".")  # Extracts `nltk_data`
        os.remove(NLTK_ZIP_PATH)  # Clean up the zip file
        st.write("NLTK data successfully downloaded and extracted.")
    except Exception as e:
        st.error(f"Failed to download or extract NLTK data: {e}")
        st.stop()

# Set the NLTK data path
nltk.data.path.append(os.path.abspath(NLTK_DATA_DIR))

# ------------------ Load Required NLTK Resources ------------------
nltk.download("stopwords", download_dir=NLTK_DATA_DIR)
nltk.download("punkt", download_dir=NLTK_DATA_DIR)

stop_words = set(stopwords.words("english"))

# ------------------ Financial Query Processing ------------------
FINANCIAL_KEYWORDS = ["revenue", "profit", "net income", "cash flow", "earnings", "financial report",
                      "balance sheet", "liabilities", "gross margin", "operating income", "expenses", "equity"]

FAISS_INDEX_URL = "https://raw.githubusercontent.com/Harithaa-bits/CAI-Assignment/main/faiss_index"
FAISS_INDEX_PATH = "faiss_index"

# Step 1: Ensure FAISS index is downloaded
if not os.path.exists(FAISS_INDEX_PATH):
    st.write("Downloading FAISS index from GitHub...")
    urllib.request.urlretrieve(FAISS_INDEX_URL, FAISS_INDEX_PATH)

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Load SentenceTransformer for embedding queries
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Dummy function for loading financial text data
def load_financial_data():
    return ["Amazon's revenue in 2023 was $500B.", "Apple's net income grew by 10% last year."]

financial_text = load_financial_data()

# Convert financial text into tokenized chunks for BM25
chunks = [sent_tokenize(text) for text in financial_text]
chunks = [item for sublist in chunks for item in sublist]  # Flatten the list

tokenized_corpus = [[word.lower() for word in word_tokenize(chunk) if word.lower() not in stop_words] for chunk in chunks]
bm25 = BM25Okapi(tokenized_corpus)

# Function to check if a query is financial
def is_financial_query(query):
    return any(word in query.lower() for word in FINANCIAL_KEYWORDS)

# BM25 Retrieval
def retrieve_bm25(query, top_k=3):
    if not is_financial_query(query):
        return ["⚠️ This is not a financial question. Please ask about financial topics."]

    query_tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(query_tokens)
    
    revenue_keywords = ["revenue", "income"]
    profit_keywords = ["profit", "net income", "operating profit"]

    for i, chunk in enumerate(chunks):
        if any(word in chunk.lower() for word in revenue_keywords):
            scores[i] *= 1.7
        elif any(word in chunk.lower() for word in profit_keywords):
            scores[i] *= 1.4

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]

# FAISS Retrieval
def retrieve_faiss(query, top_k=3):
    if not is_financial_query(query):
        return ["⚠️ This is not a financial question. Please ask about financial topics."]

    query_embedding = np.array(embed_model.encode([query]))
    distances, indices = index.search(query_embedding, top_k)

    filtered_results = [chunks[i] for i in indices[0]]
    return filtered_results if filtered_results else ["⚠️ No relevant financial data found."]

# Adaptive Retrieval
def adaptive_retrieve(query, top_k=3):
    if not is_financial_query(query):
        return ["⚠️ This is not a financial question. Please ask about financial topics."]

    bm25_results = retrieve_bm25(query, top_k)
    faiss_results = retrieve_faiss(query, top_k)

    combined_results = list(dict.fromkeys(bm25_results + faiss_results))
    
    if len(combined_results) < top_k:
        combined_results = bm25_results + faiss_results

    return combined_results[:top_k]

# Streamlit UI
st.title("Financial Query Retrieval System")

query = st.text_input("Enter your financial question:")

if query:
    results = adaptive_retrieve(query)
    st.subheader("Top Results:")
    for idx, res in enumerate(results):
        st.write(f"**{idx+1}.** {res}")
