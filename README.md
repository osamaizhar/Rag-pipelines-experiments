```markdown
# 🔍 RAG Pipeline with Groq + LLaMA 3 70B + Pinecone

This repository is a personal lab for experimenting with **Retrieval-Augmented Generation (RAG)** pipelines using the blazing-fast `llama3-70b-8192` model via **Groq**, document embedding via **Hugging Face**, and vector storage via **Pinecone**.

> ✅ Latest notebook: [`gradio_test.ipynb`]
---

## ✨ What's Inside?

- 📄 Load and split documents into context chunks
- 🧠 Generate embeddings using Hugging Face’s `all-MiniLM-L6-v2`
- 🧬 Store and retrieve vectors with **Pinecone**
- ⚡ Query Groq’s `llama3-70b-8192` via **LangChain**
- 💬 Get context-aware LLM responses
- 🖼️ Interactive **Gradio GUI** with:
  - 💬 Conversation history
  - 🗑️ Delete last message
  - ♻️ Clear full chat

---

## 🛠️ Tech Stack

| Component        | Tool/Service                             |
|------------------|-------------------------------------------|
| Embedding Model  | `sentence-transformers/all-MiniLM-L6-v2`  |
| Vector Store     | [Pinecone](https://www.pinecone.io/)      |
| LLM Inference    | [Groq](https://console.groq.com/)         |
| LLM Model        | `llama3-70b-8192`                         |
| Orchestration    | LangChain                                 |
| UI               | [Gradio](https://www.gradio.app/)         |

---

## 📂 Repository Structure

```
.
├── groq_test.ipynb            # ✅ Main notebook: Groq + Pinecone + RAG + Gradio
├── data.txt                   # Input text file used for embedding
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/osamaizhar/Rag-pipelines-experiments.git
cd Rag-pipelines-experiments
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Set Your API Keys

You will need API keys for both Groq and Pinecone. Create a `.env` file or set the following environment variables:

```bash
export GROQ_API_KEY="your_groq_api_key"
export PINECONE_API_KEY="your_pinecone_api_key"
export PINECONE_ENVIRONMENT="your_pinecone_environment"  # e.g., "gcp-starter"
```

---

## ▶️ How to Use

1. Add your custom data to `data.txt`
2. Run all cells in `groq_test.ipynb`
3. Use the **Gradio chat interface** to:
   - Ask questions about your data
   - View **conversation history**
   - 🗑️ **Delete last message**
   - ♻️ **Clear entire chat**
4. Get fast, context-aware responses using LLaMA 3 on Groq

---

> ⚡ Built for experimentation and speed. Powered by Groq’s LPU inference, Pinecone vector DB, and LangChain orchestration.
```

---

