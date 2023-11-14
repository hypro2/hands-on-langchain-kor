import os

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def summarize():
    # 데이터 준비
    with open('../dataset/akazukin_all.txt', encoding='utf-8') as f:
        akazukin_all = f.read()

    # 청크 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, # 청크의 최대 문자 수
        chunk_overlap=20, # 최대 오버랩 문자 수
    )
    texts = text_splitter.split_text(akazukin_all)

    # 확인
    print(len(texts))
    for text in texts:
        print(text[:10], ":", len(text))

    # 청크 배열을 문서 배열로 변환
    docs = [Document(page_content=t) for t in texts]

    chain = load_summarize_chain(
        llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
        chain_type="map_reduce",  # stuff, map_reduce, refine
        verbose=True,
    )

    print(chain.run(docs))

if __name__=="__main__":
    summarize()