# 🏢 EstatePulse - Real Estate RAG Research Tool

**EstatePulse** is an AI-powered Real Estate Analytics & Market Intelligence platform built with **LangChain**, **Groq Llama-3.3-70B**, **Chroma Vector DB**, and **Streamlit**. It enables users to dynamically ingest real estate web articles, vectorize their contents, and perform retrieval-augmented question answering (RAG) with precise source citations.

---

## ✨ Features

- **🏢 Dynamic Source Ingestion**: Easily paste article URLs (CNBC, Bloomberg, Zillow, financial news, etc.) to index into Chroma Vector Store.
- **➕ Expandable URL Fields**: Start with a single input field and add as many dynamic URL fields as needed.
- **🧠 Advanced RAG Architecture**:
  - **LLM Engine**: Groq `llama-3.3-70b-versatile` for fast, accurate generation.
  - **Embeddings**: `Alibaba-NLP/gte-base-en-v1.5` via HuggingFace for dense vector representations.
  - **Vector DB**: ChromaDB with similarity retrieval.
- **📊 Real-time Diagnostics**: View active vector chunk counts, collection status, and system architecture specs.
- **💬 Interactive Q&A & Sources**: Get detailed market insights with clickable source link badges.
- **📜 Q&A Session History**: Track previous queries and answers during your research session.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Framework**: LangChain (`RetrievalQAWithSourcesChain`)
- **LLM**: Groq API (`llama-3.3-70b-versatile`)
- **Embeddings**: HuggingFace (`Alibaba-NLP/gte-base-en-v1.5`)
- **Vector DB**: ChromaDB
- **Document Loaders & Splitters**: `WebBaseLoader` & `RecursiveCharacterTextSplitter`

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Sai2475/EstatePulse_RealEstate_research_tool.git
cd EstatePulse_RealEstate_research_tool
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory (refer to `.env.example`):
```env
GROQ_API_KEY="your_groq_api_key_here"
USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/137.0 Safari/537.36"
```

### 5. Launch the Application
```bash
streamlit run main.py
```

---

## 📖 How to Use

1. Enter one or more real estate news or research article URLs in the sidebar.
2. Click **🚀 Process & Build Index** to ingest and chunk the articles into Chroma DB.
3. Type your question in the search input (e.g. *"What are the current trends in 30-year mortgage rates?"*).
4. Click **Generate Answer** to receive AI insights accompanied by source citations.
