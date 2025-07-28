import streamlit as st
import asyncio
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant

st.title("DOCX to Qdrant Upload")
st.subheader("Upload your .docx file")

file = st.file_uploader("Choose a DOCX file", type="docx")

if file:
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())

    loader = Docx2txtLoader(file.name)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyAaI6cEtck9zu9Vb0UphPTez2BkFRzFXdw"
    )

    qdrant = Qdrant.from_documents(
        chunks,
        embeddings,
        url="https://fe58f34e-8a11-44b7-bc37-b36c7b67f516.us-west-1-0.aws.cloud.qdrant.io:6333",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ZOuPanOWtPTZX6-ixCgGJ-SytMMUBco320lUIenAOgk",
        collection_name="hope_cluster"
    )

    st.success("Uploaded and embedded successfully.")
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

query = st.text_input("Ask a question about your document")

if query:
    retriever = qdrant.as_retriever()
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key="AIzaSyAaI6cEtck9zu9Vb0UphPTez2BkFRzFXdw"
    )
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = chain.run(query)
    st.write("Answer:", result)
