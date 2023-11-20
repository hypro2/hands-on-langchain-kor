import os
from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever # pip install rank_bm25


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
키워드 기반의 랭킹 알고리즘 - BM25 
BM25(a.k.a Okapi BM25)는 주어진 쿼리에 대해 문서와의 연관성을 평가하는 랭킹 함수로 사용되는 알고리즘으로,TF-IDF 계열의 검색 알고리즘 중  SOTA 인 것으로 알려져 있다. 
IR 서비스를 제공하는 대표적인 기업인 엘라스틱서치에서도 ElasticSearch 5.0서부터 기본(default) 유사도 알고리즘으로 BM25 알고리즘을 채택하였다. 
"""


retriever = BM25Retriever.from_texts(texts)

query = "미코의 친구는?"
found_docs = retriever.get_relevant_documents(query)[:2]

for i, doc in enumerate(found_docs):
    print(f"{i + 1}.", doc.page_content, "\n")