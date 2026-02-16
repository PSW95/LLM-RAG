# app.py

import os
import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="Cricket Player Recognition + RAG", layout="wide")
st.title("🏏 Cricket Player Recognition Using LLM & RAG System")


def recognize_player(image):

    image.save("temp.jpg")

    for file in os.listdir("players"):
        result = DeepFace.verify(
            img1_path="temp.jpg",
            img2_path=f"players/{file}",
            enforce_detection=False
        )

        if result["verified"]:
            return file.split(".")[0]

    return None

@st.cache_resource
def create_vector_db():

    documents = []

    if not os.path.exists("data"):
        st.error("Data folder not found! Put all PDF player info inside 'data' folder.")
        return None

    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"data/{file}")
            docs = loader.load()
            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

vectorstore = create_vector_db()


@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=150,
        temperature=0.7,
        repetition_penalty=1.2,
        do_sample=True,
        device=-1
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


llm = load_llm()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)


uploaded_file = st.file_uploader(
    "Upload Player Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Recognize Player"):

        with st.spinner("Detecting Face..."):
            player = recognize_player(image)

        if player:
            st.session_state["player"] = player
            st.success(f"Recognized Player: {player}")
        else:
            st.error("Player Not Recognized")

if "player" in st.session_state:

    player = st.session_state["player"]

    question = st.text_input(
        "Ask about this player:",
        value=f"Tell me achievements and ICC ranking of {player}"
    )

    if st.button("Get Player Info"):

        with st.spinner("Retrieving from RAG..."):
            docs = vectorstore.similarity_search(question, k=1)

            if docs:
                content = docs[0].page_content

                if "ICC Rankings" in content:
                    content = content.split("ICC Rankings")[0] + "ICC Rankings"

                st.markdown("## Player Information ##")
                st.write(content)

            else:
                st.write("No information found.")

