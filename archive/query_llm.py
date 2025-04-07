from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from tkinter import scrolledtext, messagebox
from transformers import AutoModel, AutoTokenizer
# from pinecone import Pinecone, ServerlessSpec
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


pc = Pinecone(api_key=PINECONE_API)
print(PINECONE_API)


embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
#user_query = "user query"
# Function to generate embeddings without tokenization
def get_embedding(data):
    embeddings = embedding_model.encode(data).tolist()
    return embeddings


def query_pinecone(embedding):
    # Use keyword arguments to pass the embedding and other parameters
    result = index.query(vector=embedding, top_k=5, include_metadata=True)
    return result['matches']


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



def process_user_query():
    while True:
        conversation_history = []
        print("------------------------------------------------------------------------------------------------------------------------")
        user_query = input("Enter your query or press 0 to exit: ")
        if user_query == "0":
            break

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
    response = process_user_query()
    print(response)