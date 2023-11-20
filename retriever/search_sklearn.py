import os
from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import SVMRetriever #pip install lark
from langchain.retrievers import KNNRetriever, TFIDFRetriever


# 데이터 준비
with open('../dataset/akazukin_all.txt', encoding='utf-8') as f:
    akazukin_all = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # 청크의 최대 문자 수
    chunk_overlap=20,  # 최대 오버랩 문자 수
)
texts = text_splitter.split_text(akazukin_all)

# 메타데이터 준비
metadatas = [
    {"source": "1장"},
    {"source": "2장"},
    {"source": "3장"},
    {"source": "4장"},
    {"source": "5~6장"},
    {"source": "7장"}
]



"""
벡터 머신 지원 ( SVM ) 분류, 회귀 및 특이 치 탐지에 사용되는 일련의 감독 학습 방법입니다.
"""

retriever = SVMRetriever.from_texts(texts,
                                    OpenAIEmbeddings(openai_api_key=openai_api_key),
                                    k=3)

query = "미코의 친구는?"
found_docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(found_docs):
    print(f"{i + 1}.", doc.page_content, "\n")


"""
KNN
"""


retriever = KNNRetriever.from_texts(texts,
                                    OpenAIEmbeddings(openai_api_key=openai_api_key),
                                    k=3)

query = "미코의 친구는?"
found_docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(found_docs):
    print(f"{i + 1}.", doc.page_content, "\n")


"""
TF-IDF
"""


retriever = TFIDFRetriever.from_texts(texts, k=3)

query = "미코의 친구는?"
found_docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(found_docs):
    print(f"{i + 1}.", doc.page_content, "\n")

# retriever.save_local("testing.pkl")