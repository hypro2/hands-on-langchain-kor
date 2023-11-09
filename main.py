import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

chat = ChatOpenAI(temperature=.7,
                  callbacks=([StreamingStdOutCallbackHandler()]),
                  streaming=True,
                  verbose=True,
                  openai_api_key=openai_api_key
                  )


response = chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
        HumanMessage(content="I like tomatoes, what should I eat?")
    ]
)

print(response)

pass