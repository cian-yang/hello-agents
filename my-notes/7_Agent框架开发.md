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
