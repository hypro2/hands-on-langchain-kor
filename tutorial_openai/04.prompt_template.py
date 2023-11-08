import os

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain import PromptTemplate, ChatPromptTemplate

template = """
신생 회사의 네이밍 컨설턴트 역할을 해 주셨으면 합니다.
다음은 좋은 회사 이름의 몇 가지 예입니다:
- 검색 엔진, Google 
- 소셜 미디어, Facebook 
- 동영상 공유, YouTube

이름은 짧고 눈에 잘 띄며 기억하기 쉬워야 합니다.
{product}을 만드는 회사의 좋은 이름은 무엇인가요?
"""

prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)

# 입력 변수가 없는 프롬프트 예제
no_input_prompt = PromptTemplate(input_variables=[], template="Tell me a joke.")
print(no_input_prompt.format())

# 하나의 입력 변수가 있는 예제 프롬프트
one_input_prompt = PromptTemplate(template="Tell me a {adjective} joke.", input_variables=["adjective"],)
print(one_input_prompt.format(adjective="funny"))

# 여러 입력 변수가 있는 프롬프트 예제
multiple_input_prompt = PromptTemplate(
    template="Tell me a {adjective} joke about {content}.",
    input_variables=["adjective", "content"],
)
print(multiple_input_prompt.format(adjective="funny", content="chickens"))

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
print(prompt.format(product="colorful socks"))

# ChatPrompt

template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")