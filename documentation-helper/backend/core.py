import os
from typing import Any
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
from consts import INDEX_NAME

load_dotenv()

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_ENVIRONMENT_REGION')
)

def run_llm(query: str) -> Any:
    '''
        feed the llm with the embeddings and query
    '''
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    docsearch = Pinecone.from_existing_index(
        index_name= INDEX_NAME, embedding= embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(llm = chat, chain_type='stuff', retriever= docsearch.as_retriever(), return_source_documents=True)
    return qa({"query":query})


# if __name__ == "__main__":
#     print(run_llm(query= "What is RetrievalQA chain?"))