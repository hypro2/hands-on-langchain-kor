from langchain_openai import ChatOpenAI
from functions import make_graph_img, openai_api_key, GoogleNews
from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from typing import List, Dict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# 도구 생성
@tool
def search_news(query: str) -> List[Dict[str, str]]:
    """Search Google News by input keyword"""
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)


@tool
def python_code_interpreter(code: str):
    """Call to execute python code."""
    return PythonAstREPLTool().invoke(code)


# LLM 모델을 사용하여 메시지 처리 및 응답 생성, 도구 호출이 포함된 응답 반환
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


tools = [search_news, python_code_interpreter]
api_key = openai_api_key()
model_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key).bind_tools(tools)

# ToolNode 초기화
tool_node = ToolNode(tools)

# 메시지 상태 기반 워크플로우 그래프 초기화
workflow = StateGraph(MessagesState)

# 에이전트와 도구 노드 정의 및 워크플로우 그래프에 추가
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 워크플로우 시작점에서 에이전트 노드로 연결
workflow.add_edge(START, "agent")

# 에이전트 노드에서 조건부 분기 설정, 도구 노드 또는 종료 지점으로 연결
workflow.add_conditional_edges("agent", tools_condition)

# 도구 노드에서 에이전트 노드로 순환 연결
workflow.add_edge("tools", "agent")

# 에이전트 노드에서 종료 지점으로 연결
workflow.add_edge("agent", END)

# 정의된 워크플로우 그래프 컴파일 및 실행 가능한 애플리케이션 생성
app = workflow.compile()

make_graph_img(app)

for chunk in app.stream({"messages": [("human", "처음 5개의 소수를 출력하는 python code 를 작성해줘")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

for chunk in app.stream({"messages": [("human", "search google news about AI")]},stream_mode="values"):
    chunk["messages"][-1].pretty_print()

for chunk in app.stream({"messages": [("human", "안녕? 반가워")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()