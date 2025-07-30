import streamlit as st
import asyncio
from langchain.document_loaders import Docx2txtLoader,PyPDFLoader,CSVLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant

st.title("Chatbot")
st.subheader("Upload your any type of file")

file = st.file_uploader("Choose any file", type=["docx","pdf","txt","CSV"])
if not file and "qdrant" in st.session_state:
    del st.session_state.qdrant
if "chat_history" not in st.session_state:
   st.session_state.chat_history=[]

if file:
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())
    #File detecting
    file_type=file.type
    
    if file_type== "application/pdf":
        loader = PyPDFLoader(file.name)
    
    elif file_type=="text/plain":
        loader = TextLoader(file.name)
    
    elif file_type=="text/csv":
        loader = CSVLoader(file.name)
        
    elif file_type=="application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(file.name)
    else:
        st.warning("Unsupported File Type")
        st.stop()
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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
    #We are storing our qdrant in session state so we apply condition in future
    st.session_state.qdrant=qdrant
    st.success("Uploaded and embedded successfully.")
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
if "chat_history" in st.session_state:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
if "qdrant" in st.session_state:
    query = st.chat_input("Ask a question about your document")
    if query:
        retriever = st.session_state.qdrant.as_retriever()
       llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key="AIzaSyAaI6cEtck9zu9Vb0UphPTez2BkFRzFXdw"
        )
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        result = chain.run(query)
        st.write("Answer:", result)
        # Adding history alignment 
        # User asking question
        st.session_state.chat_history.append({"role": "user","content": query})
        # assistant answer
        st.session_state.chat_history.append({"role": "assistant","content": result})
        # Display response
        with st.chat_message("assistant"):
            st.markdown(result)
else:
    st.warning("Upload any document first")
    st.stop()
     
  
