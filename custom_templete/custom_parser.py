import os
from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser


class CustomSpaceSeparatedListOutputParser(BaseOutputParser):

    def parse(self, text: str):
        """LLM 결과 물을 파싱"""
        return text.strip().split(" ")

    def get_format_instructions(self) -> str:
        """LLM에 input으로 사용될 수 있는 예시 작성."""
        return (
            "Your response should be a list of space separated values, "
            "eg: `foo bar baz`"
        )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _type(self) -> str:
        return "space-separated-list"



def main():
    parser = CustomSpaceSeparatedListOutputParser()

    prompt = PromptTemplate(
        template="Answer the user query.\n\n{query}\n",
        input_variables=["query"],
    )

    model = OpenAI(temperature=0,
                   callbacks=([StreamingStdOutCallbackHandler()]),
                   streaming=True ,
                   verbose=True,
                   openai_api_key=openai_api_key)

    prompt_and_model = prompt | model | parser

    output = prompt_and_model.invoke({"query": "Tell me a joke."})
    print(output)
    return output

if __name__=="__main__":
    main()
