from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from functions import make_graph_img, openai_api_key
from langchain_core.messages import HumanMessage
from langchain_core.messages import RemoveMessage
from langgraph.graph import END



# 웹 검색 기능을 모방하는 도구 함수 정의
@tool
def search(query: str):
    """Call to surf on the web."""
    return "웹 검색 결과 : 창팝은 신창섭의 노래를 부른다."


# LLM 모델 호출 및 응답 처리 함수
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}


# 메시지 개수가 3개 초과 시 오래된 메시지 삭제 및 최신 메시지만 유지
def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 3:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-3]]}


# 메시지 상태에 따른 다음 실행 노드 결정 로직
def should_continue(state: MessagesState) -> Literal["action", "delete_messages"]:
    """Return the next node to execute."""

    last_message = state["messages"][-1]
    # 함수 호출이 없는 경우 메시지 삭제 함수 실행

    if not last_message.tool_calls:
        return "delete_messages"

    # 함수 호출이 있는 경우 액션 실행
    return "action"


tools = [search]
tool_node = ToolNode(tools)
api_key = openai_api_key()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
bound_model = llm.bind_tools(tools)

# 체크포인트 저장을 위한 메모리 객체 초기화
memory = MemorySaver()

# 메시지 상태 기반 워크플로우 그래프 정의
workflow = StateGraph(MessagesState)

# 노드
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node(delete_messages)

# 엣지
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("action", "agent")
workflow.add_edge("delete_messages", END)

# 메모리 체크포인터를 사용하여 워크플로우 컴파일
app = workflow.compile(checkpointer=memory)
make_graph_img(app)

# 스레드 ID가 포함된 설정 객체 초기화
config = {"configurable": {"thread_id": "2"}}

# 1번째 질문 수행
input_message = HumanMessage(content="안녕하세요! 제 이름은 신창섭입니다. 잘 부탁드립니다.")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    print([(message.type, message.content) for message in event["messages"]])

input_message = HumanMessage(content="내 이름이 뭐라고요?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    print([(message.type, message.content) for message in event["messages"]])


print("--"*39)

# 앱 상태에서 메시지 목록 추출 및 저장
messages = app.get_state(config).values["messages"]
# 메시지 목록 반환
for message in messages:
    message.pretty_print()