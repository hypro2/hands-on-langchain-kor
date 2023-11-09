import os
from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.llms import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser

"""
가장 일반적이고 귀중한 구성은 다음과 같습니다:
PromptTemplate / ChatPromptTemplate -> LLM / ChatModel -> OutputParser
구축 한 거의 모든 체인이이 빌딩 블록을 사용합니다.
"""

model = OpenAI(temperature=0, openai_api_key=openai_api_key)


def llm_chain():
    # 프롬프트 템플릿 생성
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""Q: {question}\nA:"""
    )

    # LLMChain 생성
    llm_chain = LLMChain(
        llm=model,
        prompt=prompt,
        verbose=True
    )

    # LLMChain 실행
    question = "기타를 잘 치는 방법은?"
    print(llm_chain.predict(question=question))


# Runnable Sequence chain 연결
# |를 통해서 간단히 연결 할 수 있고, 대신 invoke를 사용한다.
def runnable_chain():
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""Q: {question}\nA:"""
    )
    question = "기타를 잘 치는 방법은?"
    parser = CommaSeparatedListOutputParser()
    chain = prompt | model | parser
    output = chain.invoke({"question": question})
    print(output)


# SimpleSequentialChain
def simple_sequential_chain():
    template1 = """당신은 극작가입니다. 연극 제목이 주어졌을 때, 그 줄거리를 작성하는 것이 당신의 임무입니다.
    
    제목:{title}
    시놉시스:"""
    prompt1 = PromptTemplate(input_variables=["title"], template=template1)
    chain1 = LLMChain(llm=model, prompt=prompt1)

    template2 = """당신은 연극 평론가입니다. 연극의 시놉시스가 주어지면 그 리뷰를 작성하는 것이 당신의 임무입니다.
    
    시놉시스:
    {synopsis}
    리뷰:"""
    prompt2 = PromptTemplate(input_variables=["synopsis"], template=template2)
    chain2 = LLMChain(llm=model,prompt=prompt2)

    # SimpleSequentialChain으로 두 개의 체인을 연결
    overall_chain = SimpleSequentialChain(
        chains=[chain1, chain2],
        verbose=True
    )
    print(overall_chain("서울 랩소디"))


def sequential_chain():
    template1 = """당신은 극작가입니다. 연극 제목이 주어졌을 때, 그 줄거리를 작성하는 것이 당신의 임무입니다.

    제목:{title}
    시놉시스:"""
    prompt1 = PromptTemplate(input_variables=["title"], template=template1)
    chain1 = LLMChain(llm=model, prompt=prompt1, output_key="synopsis")

    template2 = """당신은 연극 평론가입니다. 연극의 시놉시스가 주어지면 그 리뷰를 작성하는 것이 당신의 임무입니다.

    시놉시스:
    {synopsis}
    리뷰:"""
    prompt2 = PromptTemplate(input_variables=["synopsis"], template=template2)
    chain2 = LLMChain(llm=model, prompt=prompt2, output_key="review")

    overall_chain = SequentialChain(
        chains=[chain1, chain2],
        input_variables=["title"],
        output_variables=["synopsis", "review"],
        verbose=True
    )
    print(overall_chain("서울 랩소디"))

if __name__=="__main__":
    llm_chain()
    runnable_chain()
    simple_sequential_chain()
    sequential_chain()
