from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os
from dotenv import load_dotenv
from consts import INDEX_NAME

''''
    The langchain docs are read and saved into a vector store as embeddings
    When we ask it a question it will:
    - take the question from our prompt and embed it in the vector store
    - retrieve the similar vectors as context for the LLM
    - the prompt is modified with the context for information to answer the question
'''

load_dotenv()

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_ENVIRONMENT_REGION')
)

def ingest_docs()->None:
    #Read the docs by a loader
    loader = ReadTheDocsLoader(path="documentation-helper/langchain-docs/langchain.readthedocs.io/en/latest", encoding='utf-8',)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    #Split the docs into chunks, do not create a big chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    #replace the prefix 
    for doc in documents:
        old_path = doc.metadata['source']
        new_url = old_path.replace('langchain-docs', 'https:/')
        doc.metadata.update({'source': new_url})

    #Initialise an embedding and persist the documents to pinecone
    print(f'Going to insert {len(documents)} to Pinecone')
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    Pinecone.from_documents(documents=documents, embedding=embeddings, index_name=INDEX_NAME)
    print('Persisted documents in vector store')

if __name__ == '__main__':
    ingest_docs()
