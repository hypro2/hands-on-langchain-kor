from IPython.display import Image
from langchain_core.runnables.graph import  MermaidDrawMethod

from PIL import Image as PILImage
import io

from util import config_util
import os

def openai_api_key():
    config = config_util.ConfigClsf().get_config()
    openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])
    return openai_api_key


def make_graph_img(graph):
    # Mermaid 그래프 이미지 생성
    img_data = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    # IPython Image 객체 생성
    img = Image(img_data)
    # 바이트 데이터를 PIL 이미지로 변환 후 열기
    pimg = PILImage.open(io.BytesIO(img.data))
    pimg.show()


import feedparser
from urllib.parse import quote
from typing import List, Dict, Optional


class GoogleNews:
    def __init__(self):
        self.base_url = "https://news.google.com/rss"

    def _fetch_news(self, url: str, k: int = 3) -> List[Dict[str, str]]:
        news_data = feedparser.parse(url)
        return [{"title": entry.title, "link": entry.link} for entry in news_data.entries[:k]]

    def _collect_news(self, news_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not news_list:
            print("해당 키워드의 뉴스가 없습니다.")
            return []

        result = []
        for news in news_list:
            result.append({"url": news["link"], "content": news["title"]})

        return result

    def search_latest(self, k: int = 3) -> List[Dict[str, str]]:
        url = f"{self.base_url}?hl=ko&gl=KR&ceid=KR:ko"
        news_list = self._fetch_news(url, k)
        return self._collect_news(news_list)

    def search_by_keyword(self, keyword: Optional[str] = None, k: int = 3) -> List[Dict[str, str]]:
        if keyword:
            encoded_keyword = quote(keyword)
            url = f"{self.base_url}/search?q={encoded_keyword}&hl=ko&gl=KR&ceid=KR:ko"
        else:
            url = f"{self.base_url}?hl=ko&gl=KR&ceid=KR:ko"
        news_list = self._fetch_news(url, k)
        return self._collect_news(news_list)
