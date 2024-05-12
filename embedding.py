from langchain_nomic.embeddings import NomicEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

def get_embedding_function():
    embeddings_func = NomicEmbeddings(model="nomic-embed-text-v1.5",)
    return embeddings_func
