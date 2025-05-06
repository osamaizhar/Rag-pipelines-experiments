from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

import numpy as np
import gradio as gr

import glob
import pandas as pd
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



# # Init Pinecone


pc = Pinecone(api_key=PINECONE_API)
print(PINECONE_API)


#  --------------- initialize pinecone -----------------------------
# pc.create_index_for_model(
#     name="test-index",
#     cloud="aws",
#     region="us-east-1",
#     embed={
#         "model":"llama-text-embed-v2",
#         "field_map":{"text": "page_content"}
#     }
# )



# Connect to the index
index = pc.Index("ai-coach")
#index = pc.Index("ahsan-400pg-pdf-doc-test")


embedding_model = AutoModel.from_pretrained(
    'jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
# user_query = "user query"
# Function to generate embeddings without tokenization


def get_embedding(data):
    embeddings = embedding_model.encode(data).tolist()
    return embeddings



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
    result = index.query(vector=embedding, top_k=20, include_metadata=True)
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
            "max_tokens": 8192  # max from groq website
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

# # Gradio GUI TEST


# system_message = f"""
#     You are a knowledgeable and friendly coach. Your goal is to help students understand concepts in a detailed and easy-to-understand manner. 
#     Be patient, ask guiding questions, and provide step-by-step explanations where needed. Adapt your responses to the student's knowledge level 
#     and help them build confidence in their learning. Refer relevant material to the student and encourage them to explore further.

#     Based on the context and the student's question, provide a thoughtful and detailed explanation. Encourage them to think about the topic and 
#     offer further guidance if needed.
#     """

# def gradio_interface(prompt,history =[]):
#     output = process_user_query(prompt,history)
#     history.append((prompt,output))
#     return history

# gr.Interface(fn=gradio_interface, inputs= ['text',"state"], outputs=["chatbot","state"]).launch(debug=True,share=True)


# ------------------------------------------- WORKING 1 -------------------------------------------

# # Function to be used by Gradio for handling the query
# def gradio_process(user_query):
#     response = process_user_query(user_query, conversation_history)
#     return response

# # Create Gradio interface
# interface = gr.Interface(fn=gradio_process, inputs="text", outputs="text", title="RAG-based Coaching System")

# # Launch Gradio app
# interface.launch()
# ------------------------------------------- WORKING 2 -------------------------------------------

# Initialize empty conversation history (list of tuples)
# conversation_history = []

# def process_user_query(user_query: str, conversation_history: list):
#     print(f"User Query Tokens: {count_tokens(user_query)}")

#     # Generate embedding and get relevant context
#     embedding = get_embedding(user_query)
#     relevant_chunks = query_pinecone(embedding)
#     context = "\n".join(chunk['metadata']["text"] for chunk in relevant_chunks)
#     print("CONTEXT:", context)

#     # Format conversation history for the prompt
#     history_str = "\n".join(
#         f"User: {user}\nCoach: {response}" 
#         for user, response in conversation_history
#     )

#     # Create structured prompt
#     prompt = f"""You are a knowledgeable and friendly coach. Follow these guidelines:
#     1. Provide clear, step-by-step explanations
#     2. Ask guiding questions to encourage critical thinking
#     3. Adapt to the student's knowledge level
#     4. Use examples from the provided context when relevant

#     Context from learning materials:
#     {context}

#     Conversation history:
#     {history_str}

#     New student question:
#     "{user_query}"

#     Provide a helpful response:"""

#     # Get LLM response
#     groq_response = query_groq(prompt)
#     print(f"Response Tokens: {count_tokens(groq_response)}")

#     # Return updated history with new interaction
#     return conversation_history + [(user_query, groq_response)]

# # Gradio Interface
# with gr.Blocks() as interface:
#     gr.Markdown("# 🧑‍🏫 AI Coaching Assistant")
#     gr.Markdown("Welcome! I'm here to help you learn. Type your question below.")

#     # State management
#     chat_history = gr.State(conversation_history)

#     with gr.Row():
#         chatbot = gr.Chatbot(height=500)
#         with gr.Column(scale=0.5):
#             context_display = gr.Textbox(label="Relevant Context", interactive=False)

#     user_input = gr.Textbox(label="Your Question", placeholder="Type here...")

#     with gr.Row():
#         submit_btn = gr.Button("Submit", variant="primary")
#         undo_btn = gr.Button("Undo Last")
#         clear_btn = gr.Button("Clear History")

#     def handle_submit(user_input, history):
#         if not user_input.strip():
#             return gr.update(), history, ""

#         # Process query and update history
#         new_history = process_user_query(user_input, history)

#         # Get latest context for display
#         latest_context = "\n".join([chunk['metadata']["text"] for chunk in query_pinecone(
#             get_embedding(user_input)
#         )][:3])  # Show top 3 context snippets

#         return "", new_history, latest_context

#     # Component interactions
#     submit_btn.click(
#         handle_submit,
#         [user_input, chat_history],
#         [user_input, chat_history, context_display]
#     ).then(
#         lambda x: x,
#         [chat_history],
#         [chatbot]
#     )

#     undo_btn.click(
#         lambda history: history[:-1] if history else [],
#         [chat_history],
#         [chat_history]
#     ).then(
#         lambda x: x,
#         [chat_history],
#         [chatbot]
#     )

#     clear_btn.click(
#         lambda: [],
#         None,
#         [chat_history]
#     ).then(
#         lambda: ([], ""),
#         None,
#         [chatbot, context_display]
#     )

# interface.launch(share=True)
# Just change the launch command to:
#interface.launch(share=True, auth=("username", "password"))  # Add basic auth


# self hosting

# # Run with:
# interface.launch(
#     server_name="0.0.0.0",
#     server_port=7860,
#     show_error=True
# )


# ------------------------------------------- WORKING 3 Enter key submits user query -------------------------------------------
# Initialize empty conversation history (list of tuples)
conversation_history = []

def process_user_query(user_query: str, conversation_history: list):
    print(f"User Query Tokens: {count_tokens(user_query)}")

    # Generate embedding and get relevant context
    embedding = get_embedding(user_query)
    relevant_chunks = query_pinecone(embedding)
    context = "\n".join(chunk['metadata']["text"] for chunk in relevant_chunks)
    print("CONTEXT:", context)

    # Format conversation history for the prompt
    history_str = "\n".join(
        f"User: {user}\nCoach: {response}" 
        for user, response in conversation_history
    )

    # Create structured prompt
    prompt = f"""You are a knowledgeable and friendly coach. Follow these guidelines:
    1. Provide clear, step-by-step explanations
    2. Ask guiding questions to encourage critical thinking
    3. Adapt to the student's knowledge level
    4. Use examples from the provided context when relevant

    Context from learning materials:
    {context}

    Conversation history:
    {history_str}

    New student question:
    "{user_query}"

    Provide a helpful response:"""

    # Get LLM response
    groq_response = query_groq(prompt)
    print(f"Response Tokens: {count_tokens(groq_response)}")

    # Return updated history with new interaction
    return conversation_history + [(user_query, groq_response)]

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("# 🧑‍🏫 AI Coaching Assistant")
    gr.Markdown("Welcome! I'm here to help you learn. Type your question below.")

    # State management
    chat_history = gr.State(conversation_history)

    with gr.Row():
        chatbot = gr.Chatbot(height=500)
        with gr.Column(scale=0.5):
            context_display = gr.Textbox(label="Relevant Context", interactive=False)

    user_input = gr.Textbox(label="Your Question", placeholder="Type here...")

    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        undo_btn = gr.Button("Undo Last")
        clear_btn = gr.Button("Clear History")

    def handle_submit(user_input, history):
        if not user_input.strip():
            return gr.update(), history, ""

        # Process query and update history
        new_history = process_user_query(user_input, history)

        # Get latest context for display
        latest_context = "\n".join([chunk['metadata']["text"] for chunk in query_pinecone(
            get_embedding(user_input)
        )][:3])  # Show top 3 context snippets

        return "", new_history, latest_context

    # Component interactions
    submit_btn.click(
        handle_submit,
        [user_input, chat_history],
        [user_input, chat_history, context_display]
    ).then(
        lambda x: x,
        [chat_history],
        [chatbot]
    )

    # Add submit on Enter key press
    user_input.submit(
        handle_submit,
        [user_input, chat_history],
        [user_input, chat_history, context_display]
    ).then(
        lambda x: x,
        [chat_history],
        [chatbot]
    )

    undo_btn.click(
        lambda history: history[:-1] if history else [],
        [chat_history],
        [chat_history]
    ).then(
        lambda x: x,
        [chat_history],
        [chatbot]
    )

    clear_btn.click(
        lambda: [],
        None,
        [chat_history]
    ).then(
        lambda: ([], ""),
        None,
        [chatbot, context_display]
    )

interface.launch(share=True)




