import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
#from langchain.llms import HuggingFaceHub
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os



st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 RAG Chatbot using LangChain + HuggingFace Embeddings")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF..."):
        # Save uploaded file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF and split
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Embedding and FAISS
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)

        retriever = db.as_retriever(search_kwargs={"k": 3})

        # LLM
        #llm = HuggingFaceHub(repo_id="deepseek-ai/DeepSeek-V3-0324", model_kwargs={"temperature": 0.5, "max_length": 512})
        #llm = HuggingFaceHub(repo_id="deepseek-ai/DeepSeek-V3-0324")

        prompt = PromptTemplate(
                    template="""You are an assistant for question-answering tasks.
                    Use the following documents to answer the question.
                    If you don't know the answer, just say that you don't know.
                    Use three sentences maximum and keep the answer concise:
                    Question: {question}
                    Documents: {documents}
                    Answer:
                    """,
                    input_variables=["query", "documents"],
                )
        
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_retries=2,
            )

        print(f"Test statement {llm}") #test
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

        st.success("PDF Processed Successfully! You can now ask questions.")

        query = st.text_input("Ask a question based on your PDF" )

        if query:
            with st.spinner("Thinking..."):
                result = chain(query)

                st.markdown("### 📌 Answer")
                st.write(result["result"])

                st.markdown("### 📚 Source Chunks")
                for doc in result["source_documents"]:
                    st.info(doc.page_content[:300] + "...")

