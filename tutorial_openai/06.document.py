import os
from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

import pandas as pd
from langchain.document_loaders import PyPDFLoader, DataFrameLoader, BSHTMLLoader
from langchain.schema import Document


# document 만들기
def my_doc():
  my_page = Document(
    page_content="이 문서는 제 문서입니다. 다른 곳에서 수집한 텍스트로 가득합니다.",
    metadata={'explain': 'The LangChain Papers'})
  print(my_page)

def my_docs():
  my_list = [
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
  ]

  my_pages = [Document(page_content = i) for i in my_list]
  print(my_pages)


# PyPDF Loader # pip install pypdf
def pdf_doc():
  loader = PyPDFLoader("../dataset/2013101000021.pdf")
  pages = loader.load_and_split()
  print(pages[:5])


# DataFrame Loader
def df_doc():
  df = pd.read_csv("../dataset/mlb_teams_2012.csv")
  loader = DataFrameLoader(df, page_content_column="Team")
  pages = loader.load_and_split()
  print(pages[:5])


# BS4 HTML Loader
def html_doc():
  loader = BSHTMLLoader("../dataset/fake-content.html")
  pages = loader.load_and_split()
  print(pages[:5])


if __name__=="__main__":
    my_doc()
    my_docs()
    pdf_doc()
    df_doc()
    html_doc()
