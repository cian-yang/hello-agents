# 案例背景

在实际工作中，我们经常需要处理大量的技术文档、研究论文、产品手册等PDF文件。**传统的文档阅读方式效率低下，难以快速定位关键信息，更无法建立知识间的关联**。

# 目标

基于Datawhale另外一门动手学大模型教程Happy-LLM的公测PDF文档 `Happy-LLM-0727.pdf`为例，构建一个**基于Gradio的Web应用**，展示如何使用**RAGTool和MemoryTool构建完整的交互式学习助手**

希望实现以下功能：

1. **智能文档处理**：使用MarkItDown**实现PDF到Markdown的统一转换**，基于Markdown结构的**智能分块策略**，高效的**向量化和索引构建**
2. **高级检索问答**：**多查询扩展（MQE）提升召回率，假设文档嵌入（HyDE）改善检索精度**，上下文感知的智能问答
3. **多层次记忆管理**：工作记忆管理当前学习任务和上下文，情景记忆记录学习事件和查询历史，语义记忆存储概念知识和理解，感知记忆处理文档特征和多模态信息
4. **个性化学习支持**：**基于学习历史的个性化推荐，记忆整合和选择性遗忘，学习报告生成和进度追踪**

# 工作流程

五个步骤之间的关系和数据流动。五个步骤形成了一个完整的闭环：

1. 步骤1将**PDF文档处理后的信息记录到记忆系统**，
2. 步骤2的**检索结果也会记录到记忆系统**，
3. 步骤3展示记忆系统的完整功能（添加、检索、整合、遗忘），
4. 步骤4**整合RAG和Memory提供智能路由**，
5. 步骤5**收集所有统计信息生成学习报告**。

# Web应用划分

整个应用分为三个核心部分：

1. **核心助手类（PDFLearningAssistant）**：封装RAGTool和MemoryTool的调用逻辑
2. **Gradio Web界面**：提供友好的**用户交互界面**，这个部分可以参考示例代码学习
3. **其他核心功能**：笔记记录、学习回顾、统计查看和报告生成

# 核心助手类

实现核心的助手类 `PDFLearningAssistant`，它封装了RAGTool和MemoryTool的调用逻辑。

## 类的初始化

在这个初始化过程中，我们做了几个关键的设计决策：

- **MemoryTool的初始化**：通过 **`user_id`参数实现用户级别的记忆隔离**。不同用户的学习记忆是完全独立的，每个用户都有自己的工作记忆、情景记忆、语义记忆和感知记忆空间。
- **RAGTool的初始化** ：通过 **`rag_namespace`参数实现知识库的命名空间隔离**。使用 `f"pdf_{user_id}"`作为命名空间，每个用户都有自己独立的PDF知识库。
- **会话管理** ：`session_id`用于追踪单次学习会话的完整过程，便于后续的学习历程回顾和分析。
- **统计信息**：`stats`字典记录关键的学习指标，用于生成学习报告。

```python
class PDFLearningAssistant:
    """智能文档问答助手"""

    def __init__(self, user_id: str = "default_user"):
        """初始化学习助手

        Args:
            user_id: 用户ID，用于隔离不同用户的数据
        """
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 初始化工具
        self.memory_tool = MemoryTool(user_id=user_id)
        self.rag_tool = RAGTool(rag_namespace=f"pdf_{user_id}")

        # 学习统计
        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0
        }

        # 当前加载的文档
        self.current_document = None

```

## 加载PDF文档


```python
def load_document(self, pdf_path: str) -> Dict[str, Any]:
    """加载PDF文档到知识库

    Args:
        pdf_path: PDF文件路径

    Returns:
        Dict: 包含success和message的结果
    """
    if not os.path.exists(pdf_path):
        return {"success": False, "message": f"文件不存在: {pdf_path}"}

    start_time = time.time()

    # 【RAGTool】处理PDF: MarkItDown转换 → 智能分块 → 向量化
    result = self.rag_tool.execute(
        "add_document",
        file_path=pdf_path,
        chunk_size=1000,
        chunk_overlap=200
    )

    process_time = time.time() - start_time

    if result.get("success", False):
        self.current_document = os.path.basename(pdf_path)
        self.stats["documents_loaded"] += 1

        # 【MemoryTool】记录到学习记忆
        self.memory_tool.execute(
            "add",
            content=f"加载了文档《{self.current_document}》",
            memory_type="episodic",
            importance=0.9,
            event_type="document_loaded",
            session_id=self.session_id
        )

        return {
            "success": True,
            "message": f"加载成功！(耗时: {process_time:.1f}秒)",
            "document": self.current_document
        }
    else:
        return {
            "success": False,
            "message": f"加载失败: {result.get('error', '未知错误')}"
        }

```



我们通过一行代码就能完成PDF的处理：

```python
result = self.rag_tool.execute(
    "add_document",
    file_path=pdf_path,
    chunk_size=1000,
    chunk_overlap=200
)
```

这个调用会触发RAGTool的完整处理流程（MarkItDown转换、增强处理、智能分块、向量化存储），这些内部细节在8.3节已经详细介绍过。我们只需要关注：

- 操作类型：`"add_document"` - 添加文档到知识库
- 文件路径：`file_path` - PDF文件的路径
- 分块参数：`chunk_size=1000, chunk_overlap=200` - 控制文本分块
- 返回结果：包含处理状态和统计信息的字典

文档加载成功后，我们使用MemoryTool记录到情景记忆：

```python
self.memory_tool.execute(
    "add",
    content=f"加载了文档《{self.current_document}》",
    memory_type="episodic",
    importance=0.9,
    event_type="document_loaded",
    session_id=self.session_id
)
```

**为什么用情景记忆？因为这是一个具体的、有时间戳的事件，适合用情景记忆记录。`session_id`参数将这个事件关联到当前学习会话，便于后续回顾学习历程**。

这个记忆记录为后续的个性化服务奠定了基础：

- 用户询问"我之前加载过哪些文档？" → 从情景记忆中检索
- 系统可以追踪用户的学习历程和文档使用情况

# 智能问答功能

实现一个 `ask`方法来处理用户的问题：

```python
def ask(self, question: str, use_advanced_search: bool = True) -> str:
    """向文档提问

    Args:
        question: 用户问题
        use_advanced_search: 是否使用高级检索（MQE + HyDE）

    Returns:
        str: 答案
    """
    if not self.current_document:
        return "⚠️ 请先加载文档！"

    # 【MemoryTool】记录问题到工作记忆
    self.memory_tool.execute(
        "add",
        content=f"提问: {question}",
        memory_type="working",
        importance=0.6,
        session_id=self.session_id
    )

    # 【RAGTool】使用高级检索获取答案
    answer = self.rag_tool.execute(
        "ask",
        question=question,
        limit=5,
        enable_advanced_search=use_advanced_search,
        enable_mqe=use_advanced_search,
        enable_hyde=use_advanced_search
    )

    # 【MemoryTool】记录到情景记忆
    self.memory_tool.execute(
        "add",
        content=f"关于'{question}'的学习",
        memory_type="episodic",
        importance=0.7,
        event_type="qa_interaction",
        session_id=self.session_id
    )

    self.stats["questions_asked"] += 1

    return answer

```

当我们调用 `self.rag_tool.execute("ask", ...)`时，RAGTool内部执行了以下高级检索流程：

- 多查询扩展（MQE）：MQE通过LLM生成语义等价但表述不同的查询，从多个角度理解用户意图，提升召回率30%-50%。
- 假设文档嵌入（HyDE）：
  - 生成假设答案文档，桥接查询和文档的语义鸿沟
  - 使用假设答案的向量进行检索

# 其他核心功能

除了加载文档和智能问答，我们还需要实现笔记记录、学习回顾、统计查看和报告生成等功能：


这些方法分别实现了：

- add_note ：将学习笔记保存到语义记忆
- recall：从记忆系统中检索学习历程
- get_stats ：获取当前会话的统计信息
- generate_report ：生成详细的学习报告并保存为JSON文件

```python
def add_note(self, content: str, concept: Optional[str] = None):
    """添加学习笔记"""
    self.memory_tool.execute(
        "add",
        content=content,
        memory_type="semantic",
        importance=0.8,
        concept=concept or "general",
        session_id=self.session_id
    )
    self.stats["concepts_learned"] += 1

def recall(self, query: str, limit: int = 5) -> str:
    """回顾学习历程"""
    result = self.memory_tool.execute(
        "search",
        query=query,
        limit=limit
    )
    return result

def get_stats(self) -> Dict[str, Any]:
    """获取学习统计"""
    duration = (datetime.now() - self.stats["session_start"]).total_seconds()
    return {
        "会话时长": f"{duration:.0f}秒",
        "加载文档": self.stats["documents_loaded"],
        "提问次数": self.stats["questions_asked"],
        "学习笔记": self.stats["concepts_learned"],
        "当前文档": self.current_document or "未加载"
    }

def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
    """生成学习报告"""
    memory_summary = self.memory_tool.execute("summary", limit=10)
    rag_stats = self.rag_tool.execute("stats")

    duration = (datetime.now() - self.stats["session_start"]).total_seconds()
    report = {
        "session_info": {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "start_time": self.stats["session_start"].isoformat(),
            "duration_seconds": duration
        },
        "learning_metrics": {
            "documents_loaded": self.stats["documents_loaded"],
            "questions_asked": self.stats["questions_asked"],
            "concepts_learned": self.stats["concepts_learned"]
        },
        "memory_summary": memory_summary,
        "rag_status": rag_stats
    }

    if save_to_file:
        report_file = f"learning_report_{self.session_id}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        report["report_file"] = report_file

    return report
```
