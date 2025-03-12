import streamlit as st
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk

# Download necessary NLTK models
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Load financial data
with open("financial_data.txt", "r", encoding="utf-8") as f:
    financial_text = f.read()

# Step 1: Chunk Merging (Improves Context)
def merge_chunks(text, chunk_size=5, overlap=2):
    """Dynamically merges chunks for better retrieval context."""
    sentences = sent_tokenize(text)
    chunks = []
    
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks

chunks = merge_chunks(financial_text, chunk_size=5, overlap=2)
print(f"Merged into {len(chunks)} improved chunks.")

# Step 2: Financial Keywords for Guardrail
FINANCIAL_KEYWORDS = ["revenue", "profit", "net income", "cash flow", "earnings", 
                      "financial report", "balance sheet", "liabilities", "gross margin",
                      "operating income", "expenses", "equity", "advertising revenue"]

def is_financial_query(query):
    """Rejects non-financial queries by checking for financial keywords."""
    return any(word in query.lower() for word in FINANCIAL_KEYWORDS)

def mask_sensitive_data(text):
    """Redacts sensitive financial numbers to prevent leaks."""
    return re.sub(r'\b\d{6,}\b', '[REDACTED]', text)

# Step 3: Embed & Store in FAISS
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.array(embed_model.encode(chunks, normalize_embeddings=True), dtype="float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

def retrieve_adaptive(query, top_k=3):
    """Retrieves financial chunks using FAISS after chunk merging and computes confidence scores."""
    if not is_financial_query(query):
        return [("⚠️ This is not a financial question. Please ask about financial topics.", 0.0)]

    query_embedding = np.array(embed_model.encode([query]))
    distances, indices = index.search(query_embedding, top_k)

    retrieved_chunks = []
    confidence_scores = []

    for i, dist in zip(indices[0], distances[0]):
        if len(chunks[i]) > 20:
            retrieved_chunks.append(chunks[i])
            confidence_scores.append(dist)  # FAISS similarity score

    if not retrieved_chunks:
        return [("⚠️ No relevant financial data found.", 0.0)]

    # Normalize confidence scores (optional)
    max_score = max(confidence_scores) if confidence_scores else 1
    confidence_scores = [round(score / max_score, 2) for score in confidence_scores]

    return list(zip(retrieved_chunks, confidence_scores))

# Step 4: Streamlit UI
st.title("📊 Financial RAG Chatbot")

query = st.text_input("Ask a financial question:")

if query:
    if not is_financial_query(query):
        st.warning("⚠️ This question is not finance-related. Please ask a financial question.")
    else:
        results = retrieve_adaptive(query)

        if results[0][1] == 0.0:
            st.warning("⚠️ No relevant financial data found. Try rephrasing your query.")
        else:
            st.subheader("🔍 Retrieved Answers:")
            for idx, (res, score) in enumerate(results):
                masked_res = mask_sensitive_data(res)
                st.write(f"🔹 Response {idx+1} (Confidence: {score:.2f}): ")
                formatted_res = masked_res.replace("\n", " ")
                st.markdown(f"{formatted_res}\n\n")
