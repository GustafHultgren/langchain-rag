# https://github.com/pixegami/langchain-rag-tutorial

import os
import shutil

from dotenv import load_dotenv

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores.chroma import Chroma


DATA_PATH = 'data'
CHROMA_PATH = 'data/chroma'

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")


def init_chroma():
    documents = load_city_documents()
    chunks = split_text(documents)
    db = save_to_chroma(chunks)
    return db

def load_city_documents():
    loader = DirectoryLoader(DATA_PATH + "/cities", glob="*.md")
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def get_embedder():
    MODEL_NAME = "sentence-transformers/all-MiniLM-l6-v2"
    embedder = HuggingFaceInferenceAPIEmbeddings(
      api_key=HF_API_KEY, model_name=MODEL_NAME
    )

    return embedder


def save_to_chroma(chunks):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedder = get_embedder()

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embedder, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

    return db
