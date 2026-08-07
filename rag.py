from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from transformers import AutoTokenizer

try:
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
except ImportError:
    try:
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
    except ImportError:
        create_retrieval_chain = None
        create_stuff_documents_chain = None

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except (ImportError, ModuleNotFoundError):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# ---------------- CONFIG ---------------- #

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "real_estate"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"

# Load tokenizer only once
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

llm = None
vector_store = None


# ---------------- INITIALIZATION ---------------- #

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=1024,
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR),
        )


def get_vectorstore_stats():
    """Returns vector store statistics (count of stored chunks, store status)."""
    try:
        initialize_components()
        if vector_store and hasattr(vector_store, "_collection"):
            count = vector_store._collection.count()
            return {
                "is_initialized": True,
                "chunk_count": count,
                "collection_name": COLLECTION_NAME,
                "dir": str(VECTORSTORE_DIR)
            }
    except Exception as e:
        pass
    return {
        "is_initialized": False,
        "chunk_count": 0,
        "collection_name": COLLECTION_NAME,
        "dir": str(VECTORSTORE_DIR)
    }


def reset_vectorstore():
    """Resets the Chroma vector store collection."""
    initialize_components()
    if vector_store:
        vector_store.reset_collection()


# ---------------- INGESTION ---------------- #

def process_urls(urls, progress_callback=None):
    """
    Ingest data from given URLs into Chroma vector store.
    Yields or calls progress_callback with (status_message, progress_percentage).
    Returns dict with summary statistics.
    """
    def log_progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)
        print(f"[{pct}%] {msg}")

    log_progress("Initializing AI models and vector store...", 10)
    initialize_components()

    log_progress("Clearing existing vector index...", 25)
    vector_store.reset_collection()

    log_progress(f"Fetching content from {len(urls)} web articles...", 40)
    loader = WebBaseLoader(urls)
    documents = loader.load()

    log_progress(f"Data loaded successfully ({len(documents)} document pages). Splitting into chunks...", 65)

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = text_splitter.split_documents(documents)

    log_progress(f"Created {len(docs)} chunks. Generating embeddings & indexing into Chroma DB...", 85)

    uuids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(documents=docs, ids=uuids)

    log_progress("Vector store update complete! Ready for research Q&A.", 100)

    return {
        "num_urls": len(urls),
        "num_docs": len(documents),
        "num_chunks": len(docs)
    }


# ---------------- QA ---------------- #

def generate_answer(query):

    initialize_components()

    if vector_store is None:
        raise RuntimeError("Vector store is not initialized.")

    stats = get_vectorstore_stats()
    if stats["chunk_count"] == 0:
        raise RuntimeError("Vector store is empty. Please process URLs first.")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    system_prompt = (
        "You are an expert real estate research analyst.\n"
        "Use the following pieces of retrieved context to answer the user's question accurately.\n"
        "If you don't know the answer or if the context doesn't contain relevant information, state that clearly.\n"
        "Include specific dates, rates, and numbers whenever available in the context.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    if create_stuff_documents_chain and create_retrieval_chain:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        result = rag_chain.invoke({"input": query})
        answer = result.get("answer", "No answer generated.")
        context_docs = result.get("context", [])
    else:
        # Fallback pure LCEL chain
        context_docs = retriever.invoke(query)
        formatted_context = "\n\n".join(doc.page_content for doc in context_docs)
        chain = prompt | llm
        response_msg = chain.invoke({"context": formatted_context, "input": query})
        answer = response_msg.content if hasattr(response_msg, "content") else str(response_msg)

    # Extract unique source URLs from retrieved context documents
    sources_set = set()
    for doc in context_docs:
        source_url = doc.metadata.get("source") or doc.metadata.get("url")
        if source_url:
            sources_set.add(source_url)

    sources_str = "\n".join(sorted(sources_set)) if sources_set else ""

    return answer, sources_str


# ---------------- MAIN ---------------- #

if __name__ == "__main__":

    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    process_urls(urls)

    answer, source = generate_answer(
        "Tell me what was the 30 year fixed mortagate rate along with the date?"
    )

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    print(source)