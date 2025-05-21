# RAG_Chatbot_streamlit
🤖 # RAG Chatbot with LangChain, Hugging Face Embeddings & ChatGroq (LLaMA 3)
A lightweight, fast, and efficient Retrieval-Augmented Generation (RAG) chatbot that processes PDF documents and allows users to query them using natural language — powered by LangChain, HuggingFace sentence transformers, FAISS vector store, and Groq’s LLaMA 3 model.


🚀 # Demo
Upload a PDF → Ask Questions → Get Concise, Contextual Answers (with Sources)

🧰 # Tech Stack
Layer	Tool/Libraryhttps://github.com/PreePrem
UI	-Streamlit
LLM	ChatGroq - LLaMA 3 8B Instant
Framework -	LangChain
Embeddings -	HuggingFace Transformers - all-MiniLM-L6-v2
Document Loader -	PyPDFLoader (from langchain_community)
Text Chunking -	RecursiveCharacterTextSplitter
Vector Store -	FAISS
Prompt Template -	Custom LangChain prompt

📂 Features
✅ Upload and parse PDFs

✅ Intelligent chunking for better context understanding

✅ Semantic search with FAISS + HuggingFace embeddings

✅ Fast, low-latency LLM inference using ChatGroq (LLaMA 3)

✅ Clean and user-friendly Streamlit interface

✅ Displays both answer and source text

📦 Installation
Clone the repository

git clone https://github.com/PreePrem/rag-chatbot-langchain

cd rag-chatbot-langchain

Install dependencies

pip install -r requirements.txt

Set API Key

🔑 # Environment Variables
Create a .env file or export:
export GROQ_API_KEY="your-groq-api-key"

Run the Streamlit app
streamlit run app.py

📘 # Example Use Case
Upload a 100-page user manual → Ask: “How do I reset the device to factory settings?”
⟶ Get an accurate, grounded answer with the paragraph source from the document.

🛠️ TODO
- Support multiple file formats (TXT, DOCX)

- Add metadata filtering (date, author, section)

- Deploy via Streamlit Sharing / Hugging Face Spaces
