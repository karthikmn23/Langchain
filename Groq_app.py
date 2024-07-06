import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_coummunity.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.core.prompts import ChatPromptTemplate

##for vectorestore
from langchain_community.vectorstores import FAISS

##Retrieval_chain_creation
from langchain.chains import create_retrieval_chain
import time


##for loading env variables

from dotenv import load_dotenv

load_dotenv()

##load the groq api key
groq_api_key = os.environ["Groq_API_KEY"]

if "vector" not in st.session.state:
    st.session_state.embedding = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecrusiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs, st.session_state.embeddings
    )
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, st.session_state.embeddings
    )

st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-It")

prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("input you prompt here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response time :", time.process_time().start)
    st.write(response["answer"])

    # with a streamlit expander
    with st.expander("Document Similarity Search"):
        # find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------")
