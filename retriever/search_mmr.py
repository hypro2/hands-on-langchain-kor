import os
from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS


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

# Faiss 벡터 인덱스 생성
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
docsearch = FAISS.from_texts(texts=texts,  # 청크 배열
                             embedding=embeddings,  # 임베딩
                             metadatas=metadatas  # 메타데이터
                             )

query = "미코의 친구는?"
found_docs = docsearch.max_marginal_relevance_search(query,
                                                     k=2, # 최종 가져올 문서
                                                     fetch_k=10 # 후보 문서 (fetch_k > k)
                                                     )

for i, doc in enumerate(found_docs):
    print(f"{i + 1}.", doc.page_content, "\n")