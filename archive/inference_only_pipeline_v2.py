'''
Changelog:
- Updated Embeddings Function nvidia/nv-embed-v1
- Optimized Code for fastest performance
- Made changes and optimizatinons to main inference function 
- Using paid apis now
'''


import gradio as gr
import os
import time

from dotenv import load_dotenv
from transformers import AutoTokenizer
from openai import OpenAI
from pinecone import Pinecone
from groq import Groq

load_dotenv()

PINECONE_API = os.getenv("PINECONE_API")
# PINECONE_ENV = os.getenv("PINECONE_ENV")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_URL = os.getenv("GROQ_CHAT_URL")

NVIDIA_API = os.getenv("NVIDIA_API")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")

# Configure headers for Groq API requests
GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}
# LLM_MODEL = "llama3-70b-8192"
LLM_MODEL = "llama-3.3-70b-versatile"


# NVidia Embedding import
client = OpenAI(
    api_key=NVIDIA_API,
    base_url=NVIDIA_BASE_URL,
)

"""
Input:
    - Context window: 128K
Ouput:
    - Output Max Tokens: 32,768

"""


def track_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[Time Tracker] `{func.__name__}` took {end - start:.4f} seconds")
        return result

    return wrapper

pc = Pinecone(api_key=PINECONE_API)

index = pc.Index("surgical-tech-complete")  # -- COMPLETE SURGICAL TECH BOOTCAMP


#@track_time
def get_embedding(text="None"):
    response = client.embeddings.create(
        input=text,
        model="nvidia/nv-embed-v1",
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"},
    )

    # print(response.data[0].embedding)
    # print(count_tokens(response.data[0].embedding))
    return response.data[0].embedding

get_embedding("None")

# Function to query Pinecone index using embeddings

#@track_time
def query_pinecone(embedding):
    # Use keyword arguments to pass the embedding and other parameters
    result = index.query(vector=embedding, top_k=5, include_metadata=True)
    return result["matches"]

#print(query_pinecone(get_embedding("Pediatric surgery definition")))


#  ------------------------ Modified query_groq function with more explicit streaming handling --------------------------------
#@track_time
def query_groq(prompt):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Always use streaming mode
    return client.chat.completions.create(
        model=LLM_MODEL,  # or whichever model you're using
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )


# Print all tool calls
# print(completion.choices[0].message.executed_tools)


# Tokenizer to count number of tokens
"""
Putting tokenizer outside of the function to avoid reinitialization and optimize performance.
"""
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en")


#@track_time
def count_tokens(text: str) -> int:
    # Encode the text into tokens
    tokens = tokenizer.encode(text)
    return len(tokens)



# --------------------------------------------------------- ## Groq and Gradio with Streaming Enabled -----------------------------------------------------
# Modified process_user_query to properly yield streaming updates
#@track_time
def process_user_query(user_query: str, conversation_history: list):
    print(f"User Query Tokens: {count_tokens(user_query)}")

    # Generate embedding and get relevant context
    embedding = get_embedding(user_query)
    relevant_chunks = query_pinecone(embedding)
    context = "\n".join(chunk["metadata"]["text"] for chunk in relevant_chunks)

    # Format conversation history for the prompt
    history_str = "\n".join(
        f"User: {user}\nCoach: {response}" for user, response in conversation_history
    )

    # Create structured prompt
    prompt = f"""You are an expert, knowledgeable, and friendly coach. Follow these guidelines carefully:

    1. Provide clear, step-by-step explanations to ensure deep understanding.
    2. Use chain-of-thought reasoning to thoroughly evaluate the provided context before responding.
    3. Ask guiding questions to encourage critical thinking.
    4. Adapt your explanation to match the student's knowledge level.
    5. Strictly use terminologies provided in the given context.
    6. Provide short, ideal examples (2-3) to illustrate your points clearly.
    7. Only answer based on the provided context—do not speculate or include external information.
    8. Always provide all specific relevant sources from the context in your responses urls, video names, video timestamps , links , resources , ebook names, lesson names , lesson numbers and anything else you think would be relevant to the user query.
    9. Perform sentiment analysis based on conversation history and user queries to adapt your responses empathetically and effectively.
    10. Must provide all relevant video timestamp from where to start watching and where to end watching 
    Context from learning materials:
    {context}

    Conversation history:
    {history_str}

    New student question:
    "{user_query}"
    
    Provide a thoughtful and contextually accurate response now:"""

    # Get streaming LLM response
    stream_response = query_groq(prompt)

    # The function now directly yields the stream chunks for the Gradio interface to use
    full_response = ""

    # First, yield a response with empty text to set up the message
    # This creates the user message immediately
    temp_history = conversation_history.copy()
    temp_history.append((user_query, ""))
    yield temp_history, context

    # Process the stream
    for chunk in stream_response:
        if (
            hasattr(chunk.choices[0].delta, "content")
            and chunk.choices[0].delta.content is not None
        ):
            content_chunk = chunk.choices[0].delta.content
            full_response += content_chunk

            # Create a temporary history with the current response
            temp_history = conversation_history.copy()
            temp_history.append((user_query, full_response))

            # Yield the updated history for display
            yield temp_history, context

    # Return the final history with the complete response
    final_history = conversation_history.copy()
    final_history.append((user_query, full_response))
    yield final_history, context


#@track_time
def create_gradio_interface(conversation_history):
    with gr.Blocks() as interface:
        gr.Markdown("# 🧑‍🏫 AI Coaching Assistant")
        gr.Markdown("Welcome! I'm here to help you learn. Type your question below.")

        # State management
        chat_history = gr.State(conversation_history)

        with gr.Row():
            chatbot = gr.Chatbot(height=500)
            with gr.Column(scale=0.5):
                context_display = gr.Textbox(
                    label="Relevant Context", interactive=False
                )

        user_input = gr.Textbox(label="Your Question", placeholder="Type here...")

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            undo_btn = gr.Button("Undo Last")
            clear_btn = gr.Button("Clear History")

        def handle_submit(user_query, history):
            if not user_query.strip():
                return gr.update(), history, ""

            # Use the generator directly from process_user_query
            # This will yield incremental updates as they arrive
            response_generator = process_user_query(user_query, history)

            for updated_history, context in response_generator:
                # Directly update the chatbot with each streaming chunk
                yield "", updated_history, context, updated_history

        # Component interactions with streaming support
        submit_btn.click(
            handle_submit,
            [user_input, chat_history],
            [user_input, chat_history, context_display, chatbot],
        )

        # Add submit on Enter key press
        user_input.submit(
            handle_submit,
            [user_input, chat_history],
            [user_input, chat_history, context_display, chatbot],
        )

        undo_btn.click(
            lambda history: history[:-1] if history else [],
            [chat_history],
            [chat_history],
        ).then(lambda x: x, [chat_history], [chatbot])

        clear_btn.click(lambda: [], None, [chat_history]).then(
            lambda: ([], ""), None, [chatbot, context_display]
        )

    return interface

def main():
    """
    Main entry point for the application.

    Initializes the conversation history with a welcome message,
    creates the Gradio interface, and launches the web app.
    """
    # Initialize conversation history with welcome message
    welcome_message = "Hi there! I'm your AI coach. I can help answer questions about your course materials, explain difficult concepts, and guide your learning journey. What would you like to know today?"
    initial_conversation_history = [("", welcome_message)]

    # Create and launch the interface
    interface = create_gradio_interface(initial_conversation_history)
    interface.launch(share=True)


if __name__ == "__main__":
    main()
