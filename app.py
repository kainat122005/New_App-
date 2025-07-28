import streamlit as st
import os

# LangChain Loaders
from langchain.document_loaders import CSVLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore

# Streamlit UI
st.title("📄 Docs2txt Chatbot — CSV, TXT, DOCX")
st.subheader("Upload a document and send it to Qdrant with Gemini embeddings")

uploaded_file = st.file_uploader("Upload your document", type=["csv", "txt", "docx"])

if uploaded_file is not None:
    st.success("✅ File uploaded!")

    file_name = uploaded_file.name
    with open(file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Auto detect file type
    try:
        if file_name.endswith(".csv"):
            loader = CSVLoader(file_name)
        elif file_name.endswith(".txt"):
            loader = TextLoader(file_name)
        elif file_name.endswith(".docx"):
            loader = Docx2txtLoader(file_name)
        else:
            st.error("❌ Unsupported file format.")
            st.stop()

        docs = loader.load()
        st.success("📄 Document loaded and parsed.")
    except Exception as e:
        st.error(f"❌ Failed to load document: {e}")
        st.stop()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Gemini embeddings
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key="AIzaSyAaI6cEtck9zu9Vb0UphPTez2BkFRzFXdw"
        )
    except Exception as e:
        st.error(f"❌ Gemini API error: {e}")
        st.stop()

    # Upload to Qdrant
    try:
        qdrant = QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            url="https://fe58f34e-8a11-44b7-bc37-b36c7b67f516.us-west-1-0.aws.cloud.qdrant.io:6333",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ZOuPanOWtPTZX6-ixCgGJ-SytMMUBco320lUIenAOgk",
            collection_name="hope_cluster"
        )
        st.success("✅ Data embedded and uploaded to Qdrant.")
    except Exception as e:
        st.error(f"❌ Qdrant upload error: {e}")
else:
    st.info("📥 Please upload a file to begin.")
