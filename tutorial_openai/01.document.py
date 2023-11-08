import os

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

# my document
my_page = Document(
  page_content="이 문서는 제 문서입니다. 다른 곳에서 수집한 텍스트로 가득합니다.",
  lookup_str='',
  metadata={
    'my_document_id': 234234,
    'my_document_source': 'The LangChain Papers',
    'my_document_create_time': 1680013019
  },
  lookup_index=0
)

# pdf document
loader = PyPDFLoader("../dataset/2013101000021.pdf")
pages = loader.load_and_split()