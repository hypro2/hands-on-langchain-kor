import os

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

text = "안녕하세요! 해변에 갈 시간입니다"
text_embedding = embeddings.embed_query(text)

print (f"임베딩 길이 : {len(text_embedding)}")
print (f"샘플은 다움과 같습니다 : {text_embedding[:5]}...")