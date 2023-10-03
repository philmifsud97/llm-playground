import os
from langchain import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

if __name__ == "__main__":
    print("hi")
    pdf_path = "vectorstore-in-memory\pdfs\paper.pdf"

    # create a PyPdf instance and load the pdf file
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    # text splitter, set it into chunks to not hit token limit in LMM
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    texts = text_splitter.split_documents(documents)
    print(len(texts))

    # create an embedding and a FAISS instance, this is a vectorstore that contains the documents in memory efficiently
    # save the vector store locally
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
    vectorstore.save_local("faiss_index_gpt")

    # load the vector store
    new_vectorstore = FAISS.load_local("faiss_index_gpt", embeddings)

    # feed the vectorstore to a ReqreivalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )
    res = qa.run("What is the gist of this paper?")

    print(res)
