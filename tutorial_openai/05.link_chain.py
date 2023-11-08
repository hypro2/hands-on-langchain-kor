import os
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain import PromptTemplate

template = """
신생 회사의 네이밍 컨설턴트 역할을 해 주셨으면 합니다.
다음은 좋은 회사 이름의 몇 가지 예입니다:
- 검색 엔진, Google 
- 소셜 미디어, Facebook 
- 동영상 공유, YouTube

이름은 짧고 눈에 잘 띄며 기억하기 쉬워야 합니다.
{product}을 만드는 회사의 좋은 이름은 무엇인가요?
"""



model = OpenAI(temperature=0, openai_api_key=openai_api_key)

parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
    output_parser=parser
)

print(LLMChain(llm=model,prompt=prompt).run({"product":"funny"}))

#chain 연결
chain = prompt | model | parser

output = chain.invoke({"product": "funny"})

print(output)

