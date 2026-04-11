# 目的

**从零开始，逐步构建一个智能体框架**——HelloAgents

# 为何需要自建Agent框架

**市面框架的快速迭代与局限性**：智能体领域是一个快速发展的领域，随时会有新的概念产生，对于智能体的设计每个框架都有自己的定位和理解，不过智能体的核心知识点是一致的。

- **过度抽象的复杂性**：
- **快速迭代带来的不稳定性**：
- **黑盒化的实现逻辑**：
- **依赖关系的复杂性**：

**从使用者到构建者的能力跃迁**：构建自己的Agent框架，实际上是一个从"使用者"向"构建者"转变的过程。这种转变带来的价值是长远的。

* **深度理解Agent工作原理**：
* **获得完全的控制权**：
* **培养系统设计能力**：

定制化需求与深度掌握的必要性：在实际应用中，不同场景对智能体的需求差异巨大，往往都需要在通用框架基础上做二次开发。

- 特定领域的优化需求
- **性能与资源的精确控制**：通用框架的"一刀切"方案往往无法满足精细化需求，直接修改已安装的库源码是一种不被推荐的做法，因为它会使后续的库升级变得困难。
- **学习与教学的透明性要求**：教学场景中，学习者更期待的是清晰地看到智能体的每一步构建过程，理解不同范式的工作机制，这要求框架具有高度的可观测性和可解释性。

# MyAgents框架的设计理念

在功能完整性和学习友好性之间找到平衡点，形成了四个核心的设计理念。

- 轻量级与教学友好的平衡
- 基于标准API的务实选择：OpenAI的API已经成为了行业标准，在这个标准之上构建
- 渐进式学习路径的精心设计
- **统一的“工具”抽象：万物皆为工具**，除了核心的Agent类，一切皆为Tools。

# LLM扩展

## 目标

主要围绕以下三个目标展开：

1. **多提供商支持**：实现对 OpenAI、ModelScope、智谱 AI 等多种主流 LLM 服务商的无缝切换，避免框架与特定供应商绑定。
2. **本地模型集成**：引入 VLLM 和 Ollama 这两种高性能本地部署方案，作为对第 3.2.3 节中 Hugging Face Transformers 方案的生产级补充，满足数据隐私和成本控制的需求。
3. **自动检测机制**：建立一套自动识别机制，使框架能根据环境信息智能推断所使用的 LLM 服务类型，简化用户的配置过程。

## 支持多提供商

问题：实际应用中，不同的服务商在环境变量命名、默认 API 地址和推荐模型等方面都存在差异。

> 如果每次切换服务商都需要用户手动查询并修改代码，会极大影响开发效率。

改进思路是：引入 `provider`，让 `MyAgentsLLM` 在内部处理不同服务商的配置细节，从而为用户提供一个统一、简洁的调用体验。

### 1、创建自定义LLM类并继承

从 `hello_agents` 库中导入 `HelloAgentsLLM` 基类，然后创建一个名为 `MyLLM` 的新类继承它。

```python
# my_llm.py
import os
from typing import Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    """
    一个自定义的LLM客户端，通过继承增加了对ModelScope的支持。
    """
    pass # 暂时留空

```

### 2、重写 `__init__` 方法以支持新供应商

在 `MyLLM` 类中重写 `__init__` 方法。

我们的目标是：

- 当用户传入 `provider="modelscope"` 时，执行我们自定义的逻辑；
- 否则，就调用父类 `HelloAgentsLLM` 的原始逻辑，使其能够继续支持 OpenAI 等其他内置的供应商。

“重写”的思想：我们拦截了 `provider="modelscope"` 的情况并进行了特殊处理，对于其他所有情况，则通过 `super().__init__(...)` 交还给父类，保留了原有框架的全部功能。

```python
class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ):
        # 检查provider是否为我们想处理的'modelscope'
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider")
            self.provider = "modelscope"
        
            # 解析 ModelScope 的凭证
            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"
        
            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("ModelScope API key not found. Please set MODELSCOPE_API_KEY environment variable.")

            # 设置默认模型和其他参数
            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)
        
            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        else:
            # 如果不是 modelscope, 则完全使用父类的原始逻辑来处理
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)


```


### 3、使用自定义的 `MyLLM` 类

首先，在 `.env` 文件中配置 ModelScope 的 API 密钥：

```
# .env file
MODELSCOPE_API_KEY="your-modelscope-api-key"

```

然后，在主程序中导入并使用 `MyLLM`：

```python
# my_main.py
from dotenv import load_dotenv
from my_llm import MyLLM # 注意:这里导入我们自己的类

# 加载环境变量
load_dotenv()

# 实例化我们重写的客户端，并指定provider
llm = MyLLM(provider="modelscope") 

# 准备消息
messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]

# 发起调用，think等方法都已从父类继承，无需重写
response_stream = llm.think(messages)

# 打印响应
print("ModelScope Response:")
for chunk in response_stream:
    # chunk在my_llm库中已经打印过一遍，这里只需要pass即可
    # print(chunk, end="", flush=True)
    pass

```

## 本地模型调用

本地模型安装，使用VLLM、Ollama等（查看另外一个文章查看详细）

## 自动检测机制 

为了尽可能减少用户的配置负担并遵循“约定优于配置”的原则

`elloAgentsLLM` 内部设计了两个核心辅助方法：`_auto_detect_provider` 和 `_resolve_credentials`。

- `_auto_detect_provider` 负责**根据环境信息推断服务商**，
- 而 `_resolve_credentials` 则**根据推断结果完成具体的参数配置**

### `_auto_detect_provider`

`_auto_detect_provider` 方法负责根据环境信息，按照下述优先级顺序，尝试自动推断服务商：

1. **最高优先级：检查特定服务商的环境变量** 。框架会依次检查 `MODELSCOPE_API_KEY`, `OPENAI_API_KEY`, `ZHIPU_API_KEY` 等环境变量是否存在。一旦发现任何一个，就会立即确定对应的服务商。
2. **次高优先级：根据 `base_url` 进行判断** 如果用户没有设置特定服务商的密钥，但设置了通用的 `LLM_BASE_URL`，框架会转而解析这个 URL。
   - **域名匹配**：通过检查 URL 中是否包含 `"api-inference.modelscope.cn"`, `"api.openai.com"` 等特征字符串来识别云服务商。
   - **端口匹配**：通过检查 URL 中是否包含 `:11434` (Ollama), `:8000` (VLLM) 等本地服务的标准端口来识别本地部署方案。
3. **辅助判断：分析 API 密钥的格式** 在某些情况下，如果上述两种方式都无法确定，**框架会尝试分析通用环境变量 `LLM_API_KEY` 的格式**。
   * 例如，某些服务商的 API 密钥有固定的前缀或独特的编码格式。
   * 不过，由于这种方式可能存在模糊性（例如多个服务商的密钥格式相似），
   * 因此它的优先级较低，仅作为辅助手段。

核心代码如下

```python
def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
    """
    自动检测LLM提供商
    """
    # 1. 检查特定提供商的环境变量 (最高优先级)
    if os.getenv("MODELSCOPE_API_KEY"): return "modelscope"
    if os.getenv("OPENAI_API_KEY"): return "openai"
    if os.getenv("ZHIPU_API_KEY"): return "zhipu"
    # ... 其他服务商的环境变量检查

    # 获取通用的环境变量
    actual_api_key = api_key or os.getenv("LLM_API_KEY")
    actual_base_url = base_url or os.getenv("LLM_BASE_URL")

    # 2. 根据 base_url 判断
    if actual_base_url:
        base_url_lower = actual_base_url.lower()
        if "api-inference.modelscope.cn" in base_url_lower: return "modelscope"
        if "open.bigmodel.cn" in base_url_lower: return "zhipu"
        if "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
            if ":11434" in base_url_lower: return "ollama"
            if ":8000" in base_url_lower: return "vllm"
            return "local" # 其他本地端口

    # 3. 根据 API 密钥格式辅助判断
    if actual_api_key:
        if actual_api_key.startswith("ms-"): return "modelscope"
        # ... 其他密钥格式判断

    # 4. 默认返回 'auto'，使用通用配置
    return "auto"

```

### `_resolve_credentials`

一旦 `provider` 被确定（无论是用户指定还是自动检测），`_resolve_credentials` 方法便会接手处理服务商的差异化配置。

它会根据 `provider` 的值，去主动查找对应的环境变量，并为其设置默认的 `base_url`。

核心代码如下

```python
def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
    """根据provider解析API密钥和base_url"""
    if self.provider == "openai":
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
        return resolved_api_key, resolved_base_url

    elif self.provider == "modelscope":
        resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
        return resolved_api_key, resolved_base_url
  
    # ... 其他服务商的逻辑

```

### 示例

假设一个用户想要使用本地的 Ollama 服务，他只需在 `.env` 文件中进行如下配置：

```python
LLM_BASE_URL="http://localhost:11434/v1"
LLM_MODEL_ID="llama3"

```

他完全不需要配置 `LLM_API_KEY` 或在代码中指定 `provider`。然后，在 Python 代码中，他只需简单地实例化 `HelloAgentsLLM` 即可：

```python
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

load_dotenv()

# 无需传入 provider，框架会自动检测
llm = HelloAgentsLLM() 
# 框架内部日志会显示检测到 provider 为 'ollama'

# 后续调用方式完全不变
messages = [{"role": "user", "content": "你好！"}]
for chunk in llm.think(messages):
    print(chunk, end="")


```

在这个过程中，`_auto_detect_provider` 方法通过解析 `LLM_BASE_URL` 中的 `"localhost"` 和 `:11434`，成功地将 `provider` 推断为 `"ollama"`。随后，`_resolve_credentials` 方法会为 Ollama 设置正确的默认参数。

# 框架接口实现

## 目标

定义一系列配套的接口和组件来处理数据流、管理配置、应对异常，并为上层应用的构建提供一个清晰、统一的结构。

三个核心文件：

- `message.py`： 定义了框架内**统一的消息格式**，确保了智能体与模型之间信息传递的标准化。
- `config.py`： 提供了一个**中心化的配置管理方案**，使框架的行为易于调整和扩展。
- `agent.py`： 定义了**所有智能体的抽象基类（`Agent`）**，为后续实现不同类型的智能体提供了统一的接口和规范。

## Message 类

在智能体与大语言模型的交互中，**对话历史是至关重要的上下文**

该类的设计有几个关键点。

1. 首先，我们通过 `typing.Literal` 将 **`role` 字段的取值严格限制为 `"user"`, `"assistant"`, `"system"`, `"tool"` 四种**，这直接对应 OpenAI API 的规范，保证了类型安全。
2. 除了 `content` 和 `role` 这两个核心字段外，我们还**增加了 `timestamp` 和 `metadata`，为日志记录和未来功能扩展预留了空间**。
3. 最后，**`to_dict()` 方法是其核心功能**之一，负责**将内部使用的 `Message` 对象转换为与 OpenAI API 兼容的字典格式**，体现了“对内丰富，对外兼容”的设计原则。

```python
"""消息系统"""
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel

# 定义消息角色的类型，限制其取值
MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    """消息类"""
  
    content: str
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
  
    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )
  
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（OpenAI API格式）"""
        return {
            "role": self.role,
            "content": self.content
        }
  
    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"

```

## Config 类

`Config` 类的职责是**将代码中硬编码配置参数集中起来**，并**支持从环境变量中读取**。

关键内容如下：

* 首先，我们**将配置项按逻辑划分**为 `LLM配置`、`系统配置` 等，使结构一目了然。
* 其次，**每个配置项都设有合理的默认值**，保证了框架在零配置下也能工作。
* 最**核心的是 `from_env()` 类**方法，它**允许用户通过设置环境变量来覆盖默认配置，无需修改代码**，这在部署到不同环境时尤其有用。

```python
"""配置管理"""
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    """HelloAgents配置类"""
  
    # LLM配置
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
  
    # 系统配置
    debug: bool = False
    log_level: str = "INFO"
  
    # 其他配置
    max_history_length: int = 100
  
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )
  
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()

```


## Agent 抽象基类

`Agent` 类是**整个框架的顶层抽象**。

它**定义了一个智能体应该具备的通用行为和属性，但并不关心具体的实现方式**。

我们通过 Python 的 **`abc` (Abstract Base Classes) 模块来实现**它，这**强制所有具体的智能体实现都必须遵循同一个“接口”**。

该类的设计体现了面向对象中的抽象原则。

- 首先，它通过**继承 `ABC` 被定义为一个不能直接实例化的抽象类**。
- 其构造函数 **`__init__` 清晰地定义了 Agent 的核心依赖**：名称、LLM 实例、系统提示词和配置。
- 最重要的部分是**使用 `@abstractmethod` 装饰的 `run` 方法**，它**强制所有子类必须实现此方法**，从而保证了所有智能体都有统一的执行入口。
- 此外，**基类还提供了通用的历史记录管理方法，这些方法与 `Message` 类协同工作，体现了组件间的联系**。

```python
"""Agent基类"""
from abc import ABC, abstractmethod
from typing import Optional, Any
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config

class Agent(ABC):
    """Agent基类"""
  
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []
  
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass
  
    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)
  
    def clear_history(self):
        """清空历史记录"""
        self._history.clear()
  
    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()
  
    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"

```

# Agent范式的框架化实现

## 目标

三种经典Agent范式（ReAct、Plan-and-Solve、Reflection）基础上进行框架化重构，并**新增SimpleAgent作为基础对话范式**

**将把这些独立的Agent实现，改造为基于统一架构的框架组件**。

所有Agent现在都共享统一的基础架构，同时保持了各自的特色和优势。

三个核心目标展开：

1. **提示词工程的系统性提升**：对第四章中的提示词进行深度优化，从特定任务导向转向通用化设计，同时增强格式约束和角色定义。
2. **接口与格式的标准化统一**：建立**统一的Agent基类和标准化的运行接口**，**所有Agent都遵循相同**的初始化参数、方法签名和历史管理机制。
3. **高度可配置的自定义能力**：支持用户自定义提示词模板、配置参数和执行策略。

## SimpleAgent

SimpleAgent是最基础的Agent实现，它展示了如何在框架基础上构建一个完整的对话智能体。

通过继承 `Agent` 基类，成功构建了一个功能完备且遵循框架规范的基础对话智能体 `MySimpleAgent`。它不仅支持基础对话，还具备可选的工具调用能力、流式响应和便利的工具管理方法。

### 1、继承框架基类

我们将通过继承框架基类来重写SimpleAgent。

首先，在你的项目目录中创建一个 `my_simple_agent.py`文件：

```python
# my_simple_agent.py
from typing import Optional, Iterator
from hello_agents import SimpleAgent, HelloAgentsLLM, Config, Message

class MySimpleAgent(SimpleAgent):
    """
    重写的简单对话Agent
    展示如何基于框架基类构建自定义Agent
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"✅ {name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")

```

### 2、重写Agent基类:run

接下来，我们需要重写Agent基类的抽象方法 `run`。

SimpleAgent支持可选的工具调用功能，也方便后续章节的扩展：

```python
# 继续在 my_simple_agent.py 中添加
import re

class MySimpleAgent(SimpleAgent):
    # ... 前面的 __init__ 方法

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        重写的运行方法 - 实现简单对话逻辑，支持可选工具调用
        """
        print(f"🤖 {self.name} 正在处理: {input_text}")

        # 构建消息列表
        messages = []

        # 添加系统消息（可能包含工具信息）
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户消息
        messages.append({"role": "user", "content": input_text})

        # 如果没有启用工具调用，使用简单对话逻辑
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            print(f"✅ {self.name} 响应完成")
            return response

        # 支持多轮工具调用的逻辑
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题:\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式:\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

        return base_prompt + tools_section

```

### 3、实现工具调用

实现工具调用的核心逻辑：

```python
# 继续在 my_simple_agent.py 中添加
class MySimpleAgent(SimpleAgent):
    # ... 前面的方法

    def _run_with_tools(self, messages: list, input_text: str, max_tool_iterations: int, **kwargs) -> str:
        """支持工具调用的运行逻辑"""
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            # 调用LLM
            response = self.llm.invoke(messages, **kwargs)

            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                print(f"🔧 检测到 {len(tool_calls)} 个工具调用")
                # 执行所有工具调用并收集结果
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    # 从响应中移除工具调用标记
                    clean_response = clean_response.replace(call['original'], "")

                # 构建包含工具结果的消息
                messages.append({"role": "assistant", "content": clean_response})

                # 添加工具结果
                tool_results_text = "\n\n".join(tool_results)
                messages.append({"role": "user", "content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

                current_iteration += 1
                continue

            # 没有工具调用，这是最终回答
            final_response = response
            break

        # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        print(f"✅ {self.name} 响应完成")

        return final_response

    def _parse_tool_calls(self, text: str) -> list:
        """解析文本中的工具调用"""
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })

        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """执行工具调用"""
        if not self.tool_registry:
            return f"❌ 错误:未配置工具注册表"

        try:
            # 智能参数解析
            if tool_name == 'calculator':
                # 计算器工具直接传入表达式
                result = self.tool_registry.execute_tool(tool_name, parameters)
            else:
                # 其他工具使用智能参数解析
                param_dict = self._parse_tool_parameters(tool_name, parameters)
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"❌ 错误:未找到工具 '{tool_name}'"
                result = tool.run(param_dict)

            return f"🔧 工具 {tool_name} 执行结果:\n{result}"

        except Exception as e:
            return f"❌ 工具调用失败:{str(e)}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """智能解析工具参数"""
        param_dict = {}

        if '=' in parameters:
            # 格式: key=value 或 action=search,query=Python
            if ',' in parameters:
                # 多个参数:action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 单个参数:key=value
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()
        else:
            # 直接传入参数，根据工具类型智能推断
            if tool_name == 'search':
                param_dict = {'query': parameters}
            elif tool_name == 'memory':
                param_dict = {'action': 'search', 'query': parameters}
            else:
                param_dict = {'input': parameters}

        return param_dict

```

### 4、添加流式响应功能和便利方法

自定义Agent添加流式响应功能和便利方法：

```python
# 继续在 my_simple_agent.py 中添加
class MySimpleAgent(SimpleAgent):
    # ... 前面的方法

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        自定义的流式运行方法
        """
        print(f"🌊 {self.name} 开始流式处理: {input_text}")

        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        # 流式调用LLM
        full_response = ""
        print("📝 实时响应: ", end="")
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            print(chunk, end="", flush=True)
            yield chunk

        print()  # 换行

        # 保存完整对话到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
        print(f"✅ {self.name} 流式响应完成")

    def add_tool(self, tool) -> None:
        """添加工具到Agent（便利方法）"""
        if not self.tool_registry:
            from hello_agents import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        self.tool_registry.register_tool(tool)
        print(f"🔧 工具 '{tool.name}' 已添加")

    def has_tools(self) -> bool:
        """检查是否有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None
  
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法）"""
        if self.tool_registry:
            self.tool_registry.unregister(tool_name)
            return True
        return False
  
    def list_tools(self) -> list:
        """列出所有可用工具"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

```

### 5、测试文件

创建一个测试文件 `test_simple_agent.py`：

```python
# test_simple_agent.py
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ToolRegistry
from hello_agents.tools import CalculatorTool
from my_simple_agent import MySimpleAgent

# 加载环境变量
load_dotenv()

# 创建LLM实例
llm = HelloAgentsLLM()

# 测试1:基础对话Agent（无工具）
print("=== 测试1:基础对话 ===")
basic_agent = MySimpleAgent(
    name="基础助手",
    llm=llm,
    system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
)

response1 = basic_agent.run("你好，请介绍一下自己")
print(f"基础对话响应: {response1}\n")

# 测试2:带工具的Agent
print("=== 测试2:工具增强对话 ===")
tool_registry = ToolRegistry()
calculator = CalculatorTool()
tool_registry.register_tool(calculator)

enhanced_agent = MySimpleAgent(
    name="增强助手",
    llm=llm,
    system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
    tool_registry=tool_registry,
    enable_tool_calling=True
)

response2 = enhanced_agent.run("请帮我计算 15 * 8 + 32")
print(f"工具增强响应: {response2}\n")

# 测试3:流式响应
print("=== 测试3:流式响应 ===")
print("流式响应: ", end="")
for chunk in basic_agent.stream_run("请解释什么是人工智能"):
    pass  # 内容已在stream_run中实时打印

# 测试4:动态添加工具
print("\n=== 测试4:动态工具管理 ===")
print(f"添加工具前: {basic_agent.has_tools()}")
basic_agent.add_tool(calculator)
print(f"添加工具后: {basic_agent.has_tools()}")
print(f"可用工具: {basic_agent.list_tools()}")

# 查看对话历史
print(f"\n对话历史: {len(basic_agent.get_history())} 条消息")

```

## ReActAgent

框架化的 ReActAgent 在保持核心逻辑不变的同时，提升了代码的组织性和可维护性，主要是通过提示词优化和与框架工具系统的集成。

将 ReAct 范式成功地集成到了框架中。

核心改进在于**利用了统一的 `ToolRegistry` 接口，并通过一个可配置、格式更严谨的提示词模板，提升了智能体执行思考-行动循环的稳定性**。

对于ReAct的测试案例，由于需要调用工具，所以统一放在文末提供测试代码。

### 提示词模板的改进

保持了原有的格式要求，强调"每次只能执行一个步骤"，避免混乱，并明确了两种Action的使用场景。

```python
MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""

```

### 重写ReActAgent的完整实现

#### 1、初始化

创建 `my_react_agent.py`文件来重写ReActAgent：其初始化参数的含义如下：

- `name`： Agent的名称。
- `llm`： `HelloAgentsLLM`的实例，负责与大语言模型通信。
- `tool_registry`： `ToolRegistry`的实例，用于管理和执行Agent可用的工具。
- `system_prompt`： 系统提示词，用于设定Agent的角色和行为准则。
- `config`： 配置对象，用于传递框架级的设置。
- `max_steps`： ReAct循环的最大执行步数，防止无限循环。
- `custom_prompt`： 自定义的提示词模板，用于替换默认的ReAct提示词。

```python
# my_react_agent.py
import re
from typing import Optional, List, Tuple
from hello_agents import ReActAgent, HelloAgentsLLM, Config, Message, ToolRegistry

class MyReActAgent(ReActAgent):
    """
    重写的ReAct Agent - 推理与行动结合的智能体
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else MY_REACT_PROMPT
        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")

```

#### 2、run

框架化的ReActAgent将执行流程分解为清晰的步骤：

```python
def run(self, input_text: str, **kwargs) -> str:
    """运行ReAct Agent"""
    self.current_history = []
    current_step = 0

    print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

    while current_step < self.max_steps:
        current_step += 1
        print(f"\n--- 第 {current_step} 步 ---")

        # 1. 构建提示词
        tools_desc = self.tool_registry.get_tools_description()
        history_str = "\n".join(self.current_history)
        prompt = self.prompt_template.format(
            tools=tools_desc,
            question=input_text,
            history=history_str
        )

        # 2. 调用LLM
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm.invoke(messages, **kwargs)

        # 3. 解析输出
        thought, action = self._parse_output(response_text)

        # 4. 检查完成条件
        if action and action.startswith("Finish"):
            final_answer = self._parse_action_input(action)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            return final_answer

        # 5. 执行工具调用
        if action:
            tool_name, tool_input = self._parse_action(action)
            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

    # 达到最大步数
    final_answer = "抱歉，我无法在限定步数内完成这个任务。"
    self.add_message(Message(input_text, "user"))
    self.add_message(Message(final_answer, "assistant"))
    return final_answer

```

## ReflectionAgent

与之前专门针对代码生成的提示词不同，框架化的版本采用了通用化设计，使其适用于文本生成、分析、创作等多种场景，并通过 `custom_prompts`参数支持用户深度定制。

### 提示词

```python
DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}

```



### 重写MyReflectionAgent

可以尝试根据之前的代码，以及上文ReAct的实现，构建出自己的MyReflectionAgent

```python

```

### 测试

测试代码如下

```python
# test_reflection_agent.py
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from my_reflection_agent import MyReflectionAgent

load_dotenv()
llm = HelloAgentsLLM()

# 使用默认通用提示词
general_agent = MyReflectionAgent(name="我的反思助手", llm=llm)

# 使用自定义代码生成提示词（类似第四章）
code_prompts = {
    "initial": "你是Python专家，请编写函数:{task}",
    "reflect": "请审查代码的算法效率:\n任务:{task}\n代码:{content}",
    "refine": "请根据反馈优化代码:\n任务:{task}\n反馈:{feedback}"
}
code_agent = MyReflectionAgent(
    name="我的代码生成助手",
    llm=llm,
    custom_prompts=code_prompts
)

# 测试使用
result = general_agent.run("写一篇关于人工智能发展历程的简短文章")
print(f"最终结果: {result}")

```

## PlanAndSolveAgent

**与之前的自由文本的计划输出不同，框架化版本强制要求Planner以Python列表的格式输出计划**，并提供了完整的异常处理机制，确保了后续步骤能够稳定执行。

### 提示词

框架化的Plan-and-Solve提示词：

````bash
# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""
````

### 重写PlanAndSolveAgent


### 测试代码

```python
# test_plan_solve_agent.py
from dotenv import load_dotenv
from hello_agents.core.llm import HelloAgentsLLM
from my_plan_solve_agent import MyPlanAndSolveAgent

# 加载环境变量
load_dotenv()

# 创建LLM实例
llm = HelloAgentsLLM()

# 创建自定义PlanAndSolveAgent
agent = MyPlanAndSolveAgent(
    name="我的规划执行助手",
    llm=llm
)

# 测试复杂问题
question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"

result = agent.run(question)
print(f"\n最终结果: {result}")

# 查看对话历史
print(f"对话历史: {len(agent.get_history())} 条消息")

```


### 其他提示词



补充一款新的提示词，可以尝试实现 `custom_prompt`载入自定义提示词。

```python
# 创建专门用于数学问题的自定义提示词
math_prompts = {
    "planner": """
你是数学问题规划专家。请将数学问题分解为计算步骤:

问题: {question}

输出格式:
python
["计算步骤1", "计算步骤2", "求总和"]

""",
    "executor": """
你是数学计算专家。请计算当前步骤:

问题: {question}
计划: {plan}
历史: {history}
当前步骤: {current_step}

请只输出数值结果:
"""
}

# 使用自定义提示词创建数学专用Agent
math_agent = MyPlanAndSolveAgent(
    name="数学计算助手",
    llm=llm,
    custom_prompts=math_prompts
)

# 测试数学问题
math_result = math_agent.run(question)
print(f"数学专用Agent结果: {math_result}")
```

## FunctionCallAgent

基于OpenAI原生函数调用机制的Agent，展示了如何使用OpenAI的函数调用机制来构建Agent


支持以下功能：

- _build_tool_schemas:通过工具的description构建OpenAI的function calling schema
- _extract_message_content:从OpenAI的响应中提取文本
- _parse_function_call_arguments:解析模型返回的JSON字符串参数
- _convert_parameter_types:转换参数类型

这些功能**可以使其具备原生的OpenAI Function Calling的能力**，对比使用prompt约束的方式，具备更强的鲁棒性。

如下

```python
def _invoke_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: Union[str, dict], **kwargs):
        """调用底层OpenAI客户端执行函数调用"""
        client = getattr(self.llm, "_client", None)
        if client is None:
            raise RuntimeError("HelloAgentsLLM 未正确初始化客户端，无法执行函数调用。")

        client_kwargs = dict(kwargs)
        client_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            client_kwargs.setdefault("max_tokens", self.llm.max_tokens)

        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **client_kwargs,
        )

#内部逻辑是对Openai 原生的functioncall作再封装
#OpenAI 原生functioncall示例
from openai import OpenAI
client = OpenAI()

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
          },
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
      },
    }
  }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
completion = client.chat.completions.create(
  model="gpt-5",
  messages=messages,
  tools=tools,
  tool_choice="auto"
)

print(completion)

```
