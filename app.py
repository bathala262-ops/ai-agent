import streamlit as st
from agent import ask_agent
from rag import create_vector_db
import os

st.set_page_config(page_title="AI Agent", layout="wide")

st.title("🤖 Free AI Agent (RAG + LLM)")

# Create DB if not exists
if not os.path.exists("db"):
    st.info("Creating vector database...")
    create_vector_db()
    st.success("Database ready!")

# Chat UI
query = st.text_input("Ask something:")

if query:
    with st.spinner("Thinking..."):
        response = ask_agent(query)
    st.write("### 🤖 Answer:")
    st.write(response)
