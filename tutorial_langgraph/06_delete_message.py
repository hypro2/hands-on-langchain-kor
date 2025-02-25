from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from functions import make_graph_img, openai_api_key
from langchain_core.messages import HumanMessage
from langchain_core.messages import RemoveMessage


# 웹 검색 기능을 모방하는 도구 함수 정의
@tool
def search(query: str):
    """Call to surf on the web."""
    return "웹 검색 결과 : 창팝은 신창섭의 노래를 부른다."


# # 대화 상태에 따른 다음 실행 노드 결정 함수
def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "tool"


# LLM 모델 호출 및 응답 처리 함수
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}


# 도구 목록 생성 및 도구 노드 초기화
tools = [search]
tool_node = ToolNode(tools)

# 모델 초기화 및 도구 바인딩
api_key = openai_api_key()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
bound_model = llm.bind_tools(tools)

# 체크포인트 저장을 위한 메모리 객체 초기화
memory = MemorySaver()

# 상태 기반 워크플로우 그래프 초기화
workflow = StateGraph(MessagesState)

# 에이전트와 액션 노드 추가
workflow.add_node("agent", call_model)
workflow.add_node("tool", tool_node)

# 시작점을 에이전트 노드로 설정
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tool": "tool", END: END})
workflow.add_edge("tool", "agent")

# 체크포인터가 포함된 최종 실행 가능한 워크플로우 컴파일
app = workflow.compile(checkpointer=memory)
make_graph_img(app)

# 스레드 ID가 1인 기본 설정 객체 초기화
config = {"configurable": {"thread_id": "1"}}

# 질문 수행
input_message = HumanMessage(content="안녕하세요! 제 이름은 신창섭입니다. 잘 부탁드립니다.")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

input_message = HumanMessage(content="내 이름이 뭐라고요?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

print("--"*39)

# 앱 상태에서 메시지 목록 추출 및 저장
messages = app.get_state(config).values["messages"]
app.update_state(config, {"messages": RemoveMessage(id=messages[0].id)})

messages = app.get_state(config).values["messages"]
for message in messages:
    message.pretty_print()