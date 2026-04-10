# 介绍

LangGraph 作为 LangChain 生态系统的重要扩展，代表了智能体框架设计的一个全新方向。

与前面介绍的**基于“对话”的框架（如 AutoGen 和 CAMEL）不同**，

LangGraph 将智能体的**执行流程建模**为一种**状态机（State Machine）**，并将其表示为 **有向图（Directed Graph）**

在这种范式中，图的 **节点（Nodes）代表一个具体的计算步骤**（如调用 LLM、执行工具），

而 **边（Edges）则定义了从一个节点到另一个节点的跳转逻辑**。

这种设计的**革命性**之处在于它**天然支持循环**，使得构建能够**进行迭代、反思和自我修正的复杂智能体工作流**变得前所未有的直观和简单

# 构成要素

## 全局状态（State）

整个图的执行过程都**围绕一个共享的状态对象进行**。

这个状态通常被定义为一个 Python 的 `TypedDict`，它**可以包含任何你需要追踪的信息**，如对话历史、中间结果、迭代次数等。

**所有的节点都能读取和更新这个中心状态**。

```python
from typing import TypedDict, List

# 定义全局状态的数据结构
class AgentState(TypedDict):
    messages: List[str]      # 对话历史
    current_task: str        # 当前任务
    final_answer: str        # 最终答案
    # ... 任何其他需要追踪的状态

```

## 节点（Nodes）

每个节点都是一个**接收当前状态作为输入、并返回一个更新后的状态作为输出**的 Python 函数。

节点是**执行具体工作的单元**。

```python
# 定义一个“规划者”节点函数
def planner_node(state: AgentState) -> AgentState:
    """根据当前任务制定计划，并更新状态。"""
    current_task = state["current_task"]
    # ... 调用LLM生成计划 ...
    plan = f"为任务 '{current_task}' 生成的计划..."
  
    # 将新消息追加到状态中
    state["messages"].append(plan)
    return state

# 定义一个“执行者”节点函数
def executor_node(state: AgentState) -> AgentState:
    """执行最新计划，并更新状态。"""
    latest_plan = state["messages"][-1]
    # ... 执行计划并获得结果 ...
    result = f"执行计划 '{latest_plan}' 的结果..."
  
    state["messages"].append(result)
    return state

```

## 边（Edges）

边负责**连接节点，定义工作流的方向**。

最简单的边是常规边，它**指定了一个节点的输出总是流向另一个固定的节点**。

LangGraph 最强大的功能在于**条件边（Conditional Edges）**。

- 它通过一个函数来**判断当前的状态，然后动态地决定下一步应该跳转到哪个节点**。
- 这正是实现循环和复杂逻辑分支的关键。

```python
def should_continue(state: AgentState) -> str:
    """条件函数：根据状态决定下一步路由。"""
    # 假设如果消息少于3条，则需要继续规划
    if len(state["messages"]) < 3:
        # 返回的字符串需要与添加条件边时定义的键匹配
        return "continue_to_planner"
    else:
        state["final_answer"] = state["messages"][-1]
        return "end_workflow"

```

## 工作流示例

```python
from langgraph.graph import StateGraph, END

# 初始化一个状态图，并绑定我们定义的状态结构
workflow = StateGraph(AgentState)

# 将节点函数添加到图中
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)

# 设置图的入口点
workflow.set_entry_point("planner")

# 添加常规边，连接 planner 和 executor
workflow.add_edge("planner", "executor")

# 添加条件边，实现动态路由
workflow.add_conditional_edges(
    # 起始节点
    "executor",
    # 判断函数
    should_continue,
    # 路由映射：将判断函数的返回值映射到目标节点
    {
        "continue_to_planner": "planner", # 如果返回"continue_to_planner"，则跳回planner节点
        "end_workflow": END               # 如果返回"end_workflow"，则结束流程
    }
)

# 编译图，生成可执行的应用
app = workflow.compile()

# 运行图
inputs = {"current_task": "分析最近的AI行业新闻", "messages": []}
for event in app.stream(inputs):
    print(event)

```

# 实践：三步问答助手

## 项目说明


构建一个简化的问答对话助手，它会遵循一个清晰、固定的三步流程来回答用户的问题：

1. **理解 (Understand)**：首先，分析**用户的查询意图**。
2. **搜索 (Search)**：然后，**模拟搜索与意图相关的信息**。
3. **回答 (Answer)**：最后，**基于意图和搜索到的信息，生成最终答案**。

将代码分解为四个核心步骤：定义状态、创建节点、构建图、以及运行应用。

## 项目初始化设置

项目的初始化设置，包括加载环境变量和实例化大语言模型。

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient

# 加载 .env 文件中的环境变量
load_dotenv()

# 初始化模型
# 我们将使用这个 llm 实例来驱动所有节点的智能
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    temperature=0.7
)
# 初始化Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

```

## 定义全局状态

首先，我们需要定义一个贯穿整个工作流的全局状态。

这是一个**共享的数据结构，它在图的每个节点之间传递，作为工作流的持久化上下文**。

每个节点都可以读取该结构中的数据，并对其进行更新。

创建了 `SearchState` 这个 `TypedDict`，为状态对象定义了一个清晰的数据模式（Schema）。一个关键的设计是同时包含了 `user_query` 和 `search_query` 字段。这允许智能体先将用户的自然语言提问，优化成更适合搜索引擎的精炼关键词，从而显著提升搜索结果的质量。

创建了 `SearchState` 这个 `TypedDict`，为状态对象定义了一个清晰的数据模式（Schema）。

一个关键的设计是同时包含了 `user_query` 和 `search_query` 字段。

> 这允许智能体先将用户的自然语言提问，优化成更适合搜索引擎的精炼关键词，从而显著提升搜索结果的质量。

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str      # 经过LLM理解后的用户需求总结
    search_query: str    # 优化后用于Tavily API的搜索查询
    search_results: str  # Tavily搜索返回的结果
    final_answer: str    # 最终生成的答案
    step: str            # 标记当前步骤

```

## 定义工作流节点

在 LangGraph 中，每个节点都是一个执行具体任务的 Python 函数。

**这些函数接收当前的状态对象作为输入，并返回一个包含更新后字段的字典**。

### 理解与查询节点

工作流的第一步，此节点的职责是**理解用户意图，并为其生成一个最优化的搜索查询**。

该节点通过一个结构化的提示，要求 LLM 同时完成“意图理解”和“关键词生成”两个任务，并将解析出的专用搜索关键词更新到状态的 `search_query` 字段中，为下一步的精确搜索做好准备。

```python
def understand_query_node(state: SearchState) -> dict:
    """步骤1：理解用户查询并生成搜索关键词"""
    user_message = state["messages"][-1].content
  
    understand_prompt = f"""分析用户的查询："{user_message}"
请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索引擎的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    response = llm.invoke([SystemMessage(content=understand_prompt)])
    response_text = response.content
  
    # 解析LLM的输出，提取搜索关键词
    search_query = user_message # 默认使用原始查询
    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()
  
    return {
        "user_query": response_text,
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"我将为您搜索：{search_query}")]
    }

```

### 搜索节点

该节点负责执行智能体的“工具使用”能力，它将调用 Tavily API 进行真实的互联网搜索，并具备基础的错误处理功能。

此节点通过 `tavily_client.search` 发起真实的 API 调用。

它被包裹在 `try...except` 块中，用于捕获可能的异常。

如果搜索失败，它会更新 `step` 状态为 `"search_failed"`，这个状态将被下一个节点用来触发备用方案。

```python
def tavily_search_node(state: SearchState) -> dict:
    """步骤2：使用Tavily API进行真实搜索"""
    search_query = state["search_query"]
    try:
        print(f"🔍 正在搜索: {search_query}")
        response = tavily_client.search(
            query=search_query, search_depth="basic", max_results=5, include_answer=True
        )
        # ... (处理和格式化搜索结果) ...
        search_results = ... # 格式化后的结果字符串
    
        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [AIMessage(content="✅ 搜索完成！正在整理答案...")]
        }
    except Exception as e:
        # ... (处理错误) ...
        return {
            "search_results": f"搜索失败：{e}",
            "step": "search_failed",
            "messages": [AIMessage(content="❌ 搜索遇到问题...")]
        }

```

### 回答节点

回答节点能够根据上一步的搜索是否成功，来选择不同的回答策略，具备了一定的弹性。

该节点通过检查 `state["step"]` 的值来执行条件逻辑。

- 如果搜索失败，它会利用 LLM 的内部知识回答并告知用户情况。
- 如果搜索成功，它则会使用包含实时搜索结果的提示，来生成一个有时效性且有据可依的回答。

```python
def generate_answer_node(state: SearchState) -> dict:
    """步骤3：基于搜索结果生成最终答案"""
    if state["step"] == "search_failed":
        # 如果搜索失败，执行回退策略，基于LLM自身知识回答
        fallback_prompt = f"搜索API暂时不可用，请基于您的知识回答用户的问题：\n用户问题：{state['user_query']}"
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
    else:
        # 搜索成功，基于搜索结果生成答案
        answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：
用户问题：{state['user_query']}
搜索结果：\n{state['search_results']}
请综合搜索结果，提供准确、有用的回答..."""
        response = llm.invoke([SystemMessage(content=answer_prompt)])
  
    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)]
    }

```



### 构建图

将所有节点连接起来。


```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

def create_search_assistant():
    workflow = StateGraph(SearchState)
  
    # 添加节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)
  
    # 设置线性流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
  
    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

```

# 优势

最大**优势：高度的可控性与可预测性，开发者可以精确地规划智能体的每一步行为**，这对于构建需要高可靠性和可审计性的生产级应用至关重要

最强大的**特性**在于：**循环（Cycles）的原生支持，**通过条件边，我们可以轻松构建“反思-修正”循环

**高度的模块化：每个节点都是一个独立的 Python 函数**


# 局限性

**对于简单任务而言，开发过程显得更为繁琐**

**缺**少了对话式智能体那种**动态的、“涌现”式的交互**

它的**强项在于执行一个确定的、可靠的流程，而非模拟开放式的、不可预测的社会性协作**。

**调试过程同样存在挑战**：虽然流程比对话历史更清晰，但问题可能出在多个环节：某个节点内部的逻辑错误、在节点间传递的状态数据发生异变，或是边跳转的条件判断失误。这要求开发者对整个图的运行机制有全局性的理解。
