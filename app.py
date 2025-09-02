import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import Docx2txtLoader,PyPDFLoader,CSVLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.title("Chatbot")
st.subheader("Upload your any type of file")

file = st.file_uploader("Choose any file", type=["docx","pdf","txt","csv"])
if not file and "qdrant" in st.session_state:
    del st.session_state["qdrant"]
if "chat_history" not in st.session_state:
   st.session_state.chat_history=[]

if file:
    file_type = file.type
    
    import tempfile
    # Save uploaded file into a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
        tmp_file.write(file.getbuffer())  
        tmp_path = tmp_file.name

    if file_type == "application/pdf":
        loader = PyPDFLoader(tmp_path)

    elif file_type == "text/plain":
        loader = TextLoader(tmp_path)

    elif file_type == "text/csv":
        loader = CSVLoader(tmp_path)

    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(tmp_path)

    else:
        st.warning("Unsupported File Type")
        st.stop()

    docs = loader.load()

    import os
    os.remove(tmp_path)


    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    import nest_asyncio
    import asyncio
    nest_asyncio.apply()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyAaI6cEtck9zu9Vb0UphPTez2BkFRzFXdw"
    )

    qdrant = Qdrant.from_documents(
        chunks,
        embeddings,
        url="https://e728f0c3-8330-4c60-ac27-45c5d17b556b.us-east4-0.gcp.cloud.qdrant.io",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.IqLcQc2ydNpwPibyqMqwQKQKU_Xmd4NaewxIg7irGmA",
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
        from langchain.prompts import PromptTemplate

        template = """
        You are a helpful assistant for students. 
        Always answer directly, correctly, and concisely. 

       - If the answer is clearly defined in the provided text, use that.  
       - If the provided text does not fully answer the question, then use your own knowledge to complete it.  
       - Never give incomplete or misleading answers.  

        Question: {query}
        """
        prompt = PromptTemplate(template=template, input_variables=["query"])
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )

        result = chain({"query": query})

        answer = result["result"]
        sources = result["source_documents"]

        st.write("Answer:", answer)
        # Adding history alignment 
        # User asking question
        st.session_state.chat_history.append({"role": "user","content": query})
        # assistant answer
        st.session_state.chat_history.append({"role": "assistant","content": answer})
        # Display response
        with st.chat_message("assistant"):
                st.markdown(answer)
else:
    st.warning("Upload any document first")
    st.stop()
     

  
