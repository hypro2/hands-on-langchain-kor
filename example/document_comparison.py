import os
from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain.agents import AgentType, initialize_agent
from langchain.embeddings import HuggingFaceEmbeddings


class DocumentInput(BaseModel):
    question: str = Field()

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cuda'},
)

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-0613",
    openai_api_key=openai_api_key
)

tools = []
files = [
    {
        "name": "alphabet-earnings",
        "path": "../dataset/2023Q1_alphabet_earnings_release.pdf",
    },
    {
        "name": "tesla-earnings",
        "path": "../dataset/TSLA-Q1-2023-Update.pdf",
    },
]

for file in files:
    loader = PyPDFLoader(file["path"])
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = hf_embeddings
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=file["name"],
            description=f"useful when you want to answer questions about {file['name']}",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
        )
    )



agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=tools,
    llm=llm,
    verbose=True,
)

print(agent({"input": "did alphabet or tesla have more revenue?"}))