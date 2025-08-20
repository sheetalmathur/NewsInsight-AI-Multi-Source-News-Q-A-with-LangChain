import os

import langchain
import streamlit as st
import pickle
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.9, max_tokens=1000)


import requests
from bs4 import BeautifulSoup

from langchain.schema import Document
def fetch_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join(p.get_text() for p in paragraphs if p.get_text())



if process_url_clicked:

    main_placeholder.text("Scraping articles...🔍")
    texts = [fetch_article_text(url) for url in urls if url]

    docs = [Document(page_content=text, metadata={"source": url}) for text, url in zip(texts, urls) if text]
    if not docs:
        st.error("Failed to fetch or parse any articles.")
        st.stop()

    # split data
    main_placeholder.text("Splitting text...📄")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    split_docs = text_splitter.split_documents(docs)

    if not split_docs:
        st.error("No usable document chunks created.")
        st.stop()

    # Step 3: Generate embeddings and store in FAISS
    main_placeholder.text("Generating embeddings...⚙️")
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(split_docs, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

    main_placeholder.success("Done! You can now ask questions below.")

# Step 4: Accept user query
query = main_placeholder.text_input("Ask a question about the articles:")
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
        qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        retriever = vectorstore.as_retriever()
        chain = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, retriever=retriever)

        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])


