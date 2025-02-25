from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import add_messages



# 기본 사용 예시
# 두 개의 메시지 리스트를 병합합니다.
# add_messages 함수는 2개의 인자(left, right)를 받으며 좌, 우 메시지를 병합하는 방식으로 동작합니다.
msgs1 = [HumanMessage(content="안녕하세요?", id="1")]
msgs2 = [AIMessage(content="반갑습니다~", id="2")]

result1 = add_messages(msgs1, msgs2)
print(result1)

# 동일한 ID를 가진 메시지 대체 예시
msgs1 = [HumanMessage(content="안녕하세요?", id="1")]
msgs2 = [HumanMessage(content="반갑습니다~", id="1")]

result2 = add_messages(msgs1, msgs2)
print(result2)