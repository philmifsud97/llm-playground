import os
from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains import RetrievalQA

pinecone.init()


if __name__ == "__main__":
    print("hello vectorstore!")

    # read file in a text loader
    loader = TextLoader(
        "intro-to-vector-db\mediumblogs\mediumblog1.txt", encoding="utf-8"
    )
    document = loader.load()

    # text splitter, set it into chunks to not hit token limit in LMM
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    # create embeddings object
    embeddings = OpenAIEmbeddings()

    # vector db pinecone, use the index created and load the doc as vectors
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    # this will chain the embeddings from the vectorstore and send it to the llm as extra context
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa({"query": query})

    print(result)
