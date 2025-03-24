
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from tkinter import scrolledtext, messagebox
from transformers import AutoModel, AutoTokenizer
import pinecone

from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    VectorType
)

import os
import requests
import PyPDF2
import textwrap
import numpy as np
import streamlit as st
import tkinter as tk


# Important: Import pinecone-client properly
# Load environment variables from .env file
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
PINECONE_API = os.getenv("PINECONE_API")
PINECONE_ENV = os.getenv("PINECONE_ENV")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("PINECONE_API", PINECONE_API)


# Groq API settings
GROQ_EMBED_URL = "https://api.groq.com/openai/v1/embeddings"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
EMBEDDING_MODEL = "llama3-405b-8192-embed"
LLM_MODEL = "llama3-70b-8192"


# Configure headers for Groq API requests
GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}


# # PDF loader


def pdf_load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


documents = pdf_load_documents()
documents[0]


# def extract_text_from_pdf(pdf_path: str) -> str:
#     """Extract text from a PDF file."""
#     with open(pdf_path, 'r') as file:
#         pdf_reader = PyPDF2.PdfReader(file)
#         text = ""
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             text += page.extract_text() + "\n"
#     return text
# extract_text_from_pdf(DATA_PATH)


# # Text Splitting \ Chunking using Langchain


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False  # considers separators like '\n\n'if true
    )
    docs = text_splitter.split_documents(documents)
    return docs


# chunks = split_documents(documents)
# chunks


# # Init Pinecone




pc = Pinecone(api_key=PINECONE_API)
print(PINECONE_API)


# Connect to the index
index = pc.Index("test-index")


embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
#user_query = "user query"
# Function to generate embeddings without tokenization
def get_embedding(data):
    embeddings = embedding_model.encode(data).tolist()
    return embeddings

# def upsert_chunks_to_pinecone(index, chunks):
#   count = 1
#   for chunk in chunks:
#     #embedding = embedding_model.encode(chunk.page_content).tolist()
#     embedding = get_embedding(chunk.page_content)
#     # Extract metadata
#     metadata = chunk.metadata
#     text = chunk.page_content
#     # Create a unique vector ID for each chunk (e.g., based on count or some unique identifier)
#     vector_id = f"vec_{count}"

#     # Upsert the embedding along with its metadata
#     index.upsert(vectors=[(vector_id, embedding, metadata, text)])

#     print(f"Embedding {count} upserted to Pinecone with metadata")
#     count += 1
#       # Ensure data is written immediately
#   print(f"All {count} Embeddings have been upserted to pinecone")


def upsert_chunks_to_pinecone(index, chunks):
    count = 1
    for chunk in chunks:
        # Get the embedding for the chunk
        embedding = get_embedding(chunk.page_content)

        # Extract metadata and add text as part of the metadata
        metadata = chunk.metadata
        metadata["text"] = chunk.page_content  # Store text in metadata

        # Create a unique vector ID for each chunk (e.g., based on count or some unique identifier)
        vector_id = f"vec_{count}"

        # Upsert the embedding along with its metadata
        index.upsert(vectors=[(vector_id, embedding, metadata)])

        print(f"Embedding {count} upserted to Pinecone with metadata")
        count += 1

    print(f"All {count-1} Embeddings have been upserted to Pinecone")

# upsert_chunks_to_pinecone(index, chunks)

# query_embeddings = embedding_model.encode(user_query).tolist()
# query_embeddings


# # Update Vectors Function


def update_pinecone_chunks(index, chunks):
    count = 1
    for chunk in chunks:
        # Get updated embedding
        embedding = get_embedding(chunk.page_content)

        # Extract metadata and page content
        metadata = chunk.metadata
        text = chunk.page_content

        # Create a unique vector ID for each chunk (e.g., based on count or some unique identifier)
        vector_id = f"vec_{count}"

        # Update the embedding and metadata
        index.update(id=vector_id, values=embedding, set_metadata=metadata)

        print(f"Embedding {count} updated in Pinecone with new metadata")
        count += 1

    print(f"All {count-1} embeddings have been updated in Pinecone")

#update_pinecone_chunks(index, chunks)


# Since your application is designed to answer a wide range of student queries and suggest relevant material, you want to retrieve enough content to cover different facets of a topic without overwhelming the LLM with too much information.
# 
# # Starting Point:
# - A common starting point is to set top_k between **5 and 10.**
# - **top_k=5:** This can work well if your curated content is highly relevant and precise, ensuring that the top 5 matches are very close to the query.
# -  **top_k=10:** If you want the coach to consider a broader range of content—perhaps to provide diverse perspectives or cover a topic more comprehensively—increasing top_k to around 10 might be beneficial.
# 
# # Experiment and Adjust:
# - The “best” value depends on factors such as the diversity of your content, how densely your data covers the topics, and the quality of the embedding matches. It’s a good idea to experiment with different top_k values and evaluate the quality and relevance of the responses in your specific
# 

# # Query Pinecone
# 


# Function to query Pinecone index using embeddings
def query_pinecone(embedding):
    # Use keyword arguments to pass the embedding and other parameters
    result = index.query(vector=embedding, top_k=5, include_metadata=True)
    return result['matches']


# # Query Groq Inference



# Function to query Groq LLM
def query_groq(prompt: str) -> str:
    response = requests.post(
        GROQ_CHAT_URL,
        headers=GROQ_HEADERS,
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 8192 # max from groq website
        }
    )

    if response.status_code != 200:
        raise Exception(f"Error querying Groq: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


# Tokenizer to count number of tokens
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en")

def count_tokens(text: str) -> int:
    # Encode the text into tokens
    tokens = tokenizer.encode(text)
    return len(tokens)



# # Process User Query


# Main function to handle user query
# def process_user_query(user_query: str):
#     print(f"User Query Tokens : {count_tokens(user_query)}")
#     # Step 1: Generate embedding for the user query
#     embedding = get_embedding(user_query)

#     # Step 2: Query Pinecone for relevant chunks
#     relevant_chunks = query_pinecone(embedding)
#     print(f"Relevant Chunks : {relevant_chunks[0]}")
#     # Step 3: Prepare the content for the Groq LLM
#     context = "\n".join([chunk['metadata']["text"] for chunk in relevant_chunks])
#     print("------------------------------------ Context ------------------------------------------ : ", context)
#     # Step 4: Send the retrieved content as the prompt to Groq LLM
#     groq_response = query_groq(context)
#     print(f"Groq Response Tokens : {count_tokens(groq_response)}")
#     return groq_response


# # Example usage
# if __name__ == "__main__":
#     user_query = "What are the Link Layer?"
#     response = process_user_query(user_query)
#     print(response)

# def process_user_query(user_query: str):
#     print(f"User Query Tokens : {count_tokens(user_query)}")

#     # Step 1: Generate embedding for the user query
#     embedding = get_embedding(user_query)

#     # Step 2: Query Pinecone for relevant chunks
#     relevant_chunks = query_pinecone(embedding)
#     #print(f"Relevant Chunks : {relevant_chunks}")

#     # Step 3: Prepare the content (context) for the LLM
#     #context = "\n".join([chunk['metadata']["text"] for chunk in relevant_chunks])
#     #print("------------------------------------ Context ------------------------------------------ : ", context)

#     # Step 4: Craft a good coach prompt for the LLM
#     prompt = f"""
#     You are a knowledgeable and friendly coach. Your goal is to help students understand concepts in a detailed and easy-to-understand manner.
#     Be patient, ask guiding questions, and provide step-by-step explanations where needed. Adapt your responses to the student's knowledge level
#     and help them build confidence in their learning. Refer relevant material to the student and encourage them to explore further.

#     Context from the student's material:
#     {relevant_chunks}

#     The student has asked the following question:
#     "{user_query}"

#     Based on the context and the student's question, provide a thoughtful and detailed explanation. Encourage them to think about the topic and
#     offer further guidance if needed.
#     """

#     # Step 5: Send the prepared prompt (with context and user query) to the LLM
#     groq_response = query_groq(prompt)
#     print(f"Groq Response Tokens : {count_tokens(groq_response)}")

#     return groq_response


# # # Example usage
# if __name__ == "__main__":
#     while True:
#         user_query = input("Enter your query or press 0 to exit: ")
#         if user_query == "0":
#             break
#         response = process_user_query(user_query)
#         print(response)


# Initialize an empty list to store conversation history


def process_user_query(user_query: str, conversation_history: list):
    conversation_history = []

    print(f"User Query Tokens : {count_tokens(user_query)}")

    # Step 1: Generate embedding for the user query
    embedding = get_embedding(user_query)

    # Step 2: Query Pinecone for relevant chunks
    relevant_chunks = query_pinecone(embedding)

    # Prepare the context from relevant chunks
    context = "\n".join([chunk['metadata']["text"]
                        for chunk in relevant_chunks])
    print("CONTEXT: ",context)

    # Step 3: Combine conversation history with current user query
    conversation_history_str = "\n".join(conversation_history)

    # Step 4: Craft a good coach prompt for the LLM
    prompt = f"""
    You are a knowledgeable and friendly coach. Your goal is to help students understand concepts in a detailed and easy-to-understand manner. 
    Be patient, ask guiding questions, and provide step-by-step explanations where needed. Adapt your responses to the student's knowledge level 
    and help them build confidence in their learning. Refer relevant material to the student and encourage them to explore further.

    Context from the student's material:
    {context}

    Conversation history:
    {conversation_history_str}

    The student has asked the following question:
    "{user_query}"

    Based on the context and the student's question, provide a thoughtful and detailed explanation. Encourage them to think about the topic and 
    offer further guidance if needed.
    """

    # Step 5: Send the prepared prompt (with context and user query) to the LLM
    groq_response = query_groq(prompt)
    print(f"Groq Response Tokens : {count_tokens(groq_response)}")

    # Step 6: Append the user query and model's response to conversation history
    conversation_history.append(f"User: {user_query}")
    conversation_history.append(f"Coach: {groq_response}")

    return groq_response


# Example usage
if __name__ == "__main__":
    while True:
        print("------------------------------------------------------------------------------------------------------------------------")
        user_query = input("Enter your query or press 0 to exit: ")
        if user_query == "0":
            break
        response = process_user_query(user_query, conversation_history)
        print(response)


