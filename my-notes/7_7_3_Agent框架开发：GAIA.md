# GAIA 基准介绍

GAIA (General AI Assistants)：通用 AI 助手能力评估，专注于评估 AI 助手的 **通用能力**

与 **BFCL 专注于工具调用**不同，**GAIA 评估的是智能体在真实世界任务中的综合表现**。

GAIA 的设计理念是：**真实世界的问题往往需要多种能力的综合运用**

一个优秀的 AI 助手不仅需要调用工具，还需要：

- **多步推理**：将复杂问题分解为多个子问题
- **知识运用**：利用内置知识和外部知识库
- **多模态理解**：处理文本、图片、文件等多种输入
- **网页浏览**：从互联网获取最新信息
- **文件操作**：读取和处理各种格式的文件

# GAIA 数据集结构

GAIA 包含 466 个精心设计的真实世界问题，

这些问题按照复杂度和所需推理步骤分为三个难度级别，从简单的零步推理任务到需要多步复杂推理的困难任务，

全面覆盖了智能体在实际应用中可能遇到的各种场景

| 级别    | 描述     | 推理步骤 | 样本数 | 示例                             |
| ------- | -------- | -------- | ------ | -------------------------------- |
| Level 1 | 简单任务 | 0步      | 165    | “2023年诺贝尔物理学奖得主是谁？ |
| Level 2 | 中等任务 | 1-5步    | 184    | 比较最近三年GDP增长最快的国家    |
| Level 3 | 困难任务 | 5+步     | 117    | 分析某公司财报并预测下季度表现   |

关于 GAIA 数据集的样本示例可以参考下面的代码片段：关键字段说明：

- `Question`: 问题描述
- `Level`: 难度级别（1-3）
- `Final answer`: 标准答案（可能是数字、文本或文件）
- `file_name/file_path`: 附件文件（如果有）
- `Annotator Metadata`: 标注者提供的元数据（推理步骤、所需工具等）

```json
{
  "task_id": "gaia_001",
  "Question": "What is the total population of the top 3 most populous cities in California?",
  "Level": 2,
  "Final answer": "12847521",
  "file_name": "",
  "file_path": "",
  "Annotator Metadata": {
    "Steps": [
      "Search for most populous cities in California",
      "Get population data for top 3 cities",
      "Sum the populations"
    ],
    "Number of steps": 3,
    "How long did this take?": "5 minutes",
    "Tools": ["web_search", "calculator"]
  }
}

```

# 准精确匹配介绍

GAIA 使用 准精确匹配（Quasi Exact Match）评估算法，这是 GAIA 官方定义的评估标准。

该算法的核心思想是：**先对答案进行归一化处理，然后进行精确匹配**

给定预测答案 $A_{\text{pred}}$ 和标准答案 $A_{\text{true}}$，准精确匹配函数定义为：

$$
\text{Quasi\_Exact\_Match}(A_{\text{pred}}, A_{\text{true}}) = \begin{cases}
1 & \text{if } \mathcal{N}(A_{\text{pred}}) = \mathcal{N}(A_{\text{true}}) \\
0 & \text{otherwise}
\end{cases}
$$

其中 $\mathcal{N}(\cdot)$ 是归一化函数，根据答案类型应用不同的规则。

归一化函数根据答案类型应用不同的规则。

- 对于**数字类型**，需要**移除逗号分隔**符（`1,000` → `1000`）和**单位符号**（`$100` → `100`，`50%` → `50`），例如 `"$1,234.56"`归一化为 `"1234.56"`。
- 对于**字符串类型**，需要**转换为小写**（`"Apple"` → `"apple"`）、移除**冠词**（`"the apple"` → `"apple"`）、移除**多余空格**（`"hello  world"` → `"hello world"`）和移除**末尾标点**（`"hello."` → `"hello"`），例如 `"The United States"`归一化为 `"united states"`。
- 对于**列表类型**，需要按逗号分隔元素，对每个元素应用字符串归一化，按字母顺序排序后重新连接，例如 `"Paris, London, Berlin"`归一化为 `"berlin,london,paris"`。

归一化示例：

```python
# 数字答案
原始答案: "$1,234.56"
归一化后: "1234.56"

# 字符串答案
原始答案: "The United States of America"
归一化后: "united states of america"

# 列表答案
原始答案: "Paris, London, Berlin"
归一化后: "berlin, london, paris"
```

# GAIA 评估指标

GAIA 使用以下指标评估智能体性能：

## 精确匹配率 (Exact Match Rate)

精确匹配率是 GAIA 的核心指标，定义为准精确匹配成功的样本比例：

$$
\text{Exact Match Rate} = \frac{1}{N} \sum_{i=1}^{N} \text{Quasi\_Exact\_Match}(A_{\text{pred},i}, A_{\text{true},i})
$$

其中：

- $N$ 是总样本数
- $A_{\text{pred},i}$ 是第 $i$ 个样本的预测答案
- $A_{\text{true},i}$ 是第 $i$ 个样本的标准答案
- $\text{Quasi\_Exact\_Match}(\cdot, \cdot) \in \{0, 1\}$ 是准精确匹配函数

## 分级准确率 (Level-wise Accuracy)

对于每个难度级别 $\ell \in \{1, 2, 3\}$，计算该级别的准确率：

$$
\text{Accuracy}_\ell = \frac{1}{|D_\ell|} \sum_{i \in D_\ell} \text{Quasi\_Exact\_Match}(A_{\text{pred},i}, A_{\text{true},i})
$$

其中 $D_\ell$ 是难度级别 $\ell$ 的样本集合，$|D_\ell|$ 是该级别的样本数。

## 难度递进下降率 (Difficulty Progression Drop Rate)

衡量智能体在难度增加时的性能衰减：

$$
\text{Drop Rate}_{\ell \to \ell+1} = \frac{\text{Accuracy}_\ell - \text{Accuracy}_{\ell+1}}{\text{Accuracy}_\ell}
$$

- $\text{Drop Rate}_{1 \to 2}$：从 Level 1 到 Level 2 的下降率
- $\text{Drop Rate}_{2 \to 3}$：从 Level 2 到 Level 3 的下降率

## 平均推理步骤数 (Average Reasoning Steps)

评估智能体完成任务所需的平均步骤数：

$$
\text{Avg Steps} = \frac{1}{N_{\text{correct}}} \sum_{i \in \text{Correct}} \text{steps}_i
$$

其中 $N_{\text{correct}}$ 是正确回答的样本数，$\text{steps}_i$ 是第 $i$ 个样本的推理步骤数。

指标解释：

- Exact Match Rate = 1.0 ：所有样本都完全正确
- Exact Match Rate = 0.5 ：50%的样本正确，50%的样本错误
- Drop Rate = 0.3：难度增加导致准确率下降 30%
- Drop Rate = 0.0：难度增加不影响准确率（理想情况）

# GAIA 官方系统提示词

GAIA 对答案格式有严格的要求：

- 答案必须以 `FINAL ANSWER: [答案]`的格式给出；
- 对于数字类型的答案，不使用逗号分隔符和单位符号；
- 对于字符串类型的答案，不使用冠词和缩写；
- 对于列表类型的答案，使用逗号分隔并按字母顺序排列。

GAIA 要求使用特定的系统提示词，确保模型输出符合评估格式：

```python
GAIA_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.

If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.

If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.

If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""
```

# 获取 GAIA 数据集

## 权限申请

GAIA 是受限数据集（Gated Dataset），需要先在 HuggingFace 上申请访问权限。

步骤 1：申请访问权限

1. 访问 https://huggingface.co/datasets/gaia-benchmark/GAIA
2. 点击"Request access"按钮
3. 填写申请表单（通常会在几秒内批准）
4. 获取你的 HuggingFace Token：https://huggingface.co/settings/tokens

步骤 2：配置环境变量

在 `.env`文件中添加你的 HuggingFace Token：

```bash
# HuggingFace API 配置
HF_TOKEN=hf_your_token_here
```

## HelloAgents（推荐）

方法 1：使用 HelloAgents 自动下载（推荐）

**工作原理**：

- 首次运行时，使用 `snapshot_download`下载整个数据集到 `./data/gaia/`
- 数据集包含 114 个文件（问题、图片、PDF 等材料）
- 后续使用直接从本地加载，速度很快

数据集目录结构：

```
./data/gaia/
├── 2023/
│   ├── validation/
│   │   ├── metadata.jsonl  (165个问题)
│   │   ├── *.png, *.pdf, *.csv, *.xlsx  (附件文件)
│   └── test/
│       ├── metadata.jsonl  (301个问题)
│       └── ... (附件文件)
├── GAIA.py
└── README.md
```

HelloAgents 会自动处理 GAIA 数据集的下载和缓存：

```python
from hello_agents.evaluation import GAIADataset
import os

# 确保设置了HF_TOKEN，如果设置了.env无需这一行
os.environ["HF_TOKEN"] = "hf_your_token_here"

# 自动下载到 ./data/gaia/
dataset = GAIADataset(
    dataset_name="gaia-benchmark/GAIA",
    split="validation",  # 或 "test"
    level=1  # 可选: 1, 2, 3, None(全部)
)
items = dataset.load()

print(f"加载了 {len(items)} 个测试样本")
# 输出: 加载了 53 个测试样本 (Level 1)
```

## 手动下载

方法 2：手动下载

如果你想手动下载数据集：

```python
from huggingface_hub import snapshot_download
import os

# 设置Token
os.environ["HF_TOKEN"] = "hf_your_token_here"

# 下载数据集
snapshot_download(
    repo_id="gaia-benchmark/GAIA",
    repo_type="dataset",
    local_dir="./data/gaia",
    token=os.getenv("HF_TOKEN")
)
```

查看数据集统计：

```python
# 查看数据集统计
stats = dataset.get_statistics()
print(f"总样本数: {stats['total_samples']}")
print(f"级别分布: {stats['level_distribution']}")
# 输出:
# 总样本数: 165
# 级别分布: {1: 53, 2: 62, 3: 50}
```

# HelloAgents 中实现 GAIA 评估

与 BFCL 类似，我们提供两种评估方式，推荐使用GAIAEvaluationTool一键评估

## GAIAEvaluationTool

方式 1：使用 GAIAEvaluationTool 一键评估

这是最简单的方式，自动完成数据集下载、评估执行、结果导出和报告生成：

```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import GAIAEvaluationTool

# GAIA官方系统提示词（来自论文）
GAIA_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.

If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.

If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.

If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

# 1. 创建智能体（使用GAIA官方系统提示词）
llm = HelloAgentsLLM()
agent = SimpleAgent(
    name="TestAgent",
    llm=llm,
    system_prompt=GAIA_SYSTEM_PROMPT  # 关键：使用GAIA官方提示词
)

# 2. 创建GAIA评估工具
gaia_tool = GAIAEvaluationTool()

# 3. 一键运行评估
results = gaia_tool.run(
    agent=agent,
    level=1,  # Level 1: 简单任务
    max_samples=5,  # 评估5个样本
    export_results=True,  # 导出GAIA格式结果
    generate_report=True  # 生成评估报告
)

# 4. 查看结果
print(f"精确匹配率: {results['exact_match_rate']:.2%}")
print(f"部分匹配率: {results['partial_match_rate']:.2%}")
print(f"正确数: {results['exact_matches']}/{results['total_samples']}")
```

注意 ：如果你发现生成的评估结果不理想（例如准确率较低），这是正常现象。

- 虽然 Level 1 是一步推理任务，但仍然需要智能体具备工具调用能力（如搜索引擎、计算器等）才能正确回答问题。
- 我们当前使用的 SimpleAgent 主要用于演示评估流程，在工具调用能力上还有提升空间。

## Dataset + Evaluator

方式 2：使用 Dataset + Evaluator（灵活定制）

如果需要更细粒度的控制，可以直接使用底层组件：

```python
from hello_agents.evaluation import GAIADataset, GAIAEvaluator

# 1. 加载数据集
dataset = GAIADataset(level=1)
items = dataset.load()
print(f"加载了 {len(items)} 个样本")

# 2. 创建评估器
evaluator = GAIAEvaluator(dataset=dataset, level=1)

# 3. 运行评估
results = evaluator.evaluate(agent, max_samples=5)

# 4. 导出GAIA格式结果
evaluator.export_to_gaia_format(
    results,
    "gaia_results.jsonl",
    include_reasoning=True
)
```

# 核心组件实现

## （1）GAIADataset


（1）GAIADataset：支持多模态的数据加载器 

GAIA 数据集的特殊之处在于它包含多模态数据（文本、文件、图片等）：

````python
class GAIADataset:
    """GAIA数据集加载器

    支持从HuggingFace加载GAIA数据集（受限数据集）
    """

    def __init__(
        self,
        level: Optional[int] = None,
        split: str = "validation",
        local_data_dir: Optional[str] = None
    ):
        self.level = level
        self.split = split
        self.local_data_dir = local_data_dir or "./data/gaia"
        self.data = []

    def load(self) -> List[Dict[str, Any]]:
        """加载数据集"""
        # 从HuggingFace下载
        items = self._load_from_huggingface()

        # 按级别过滤
        if self.level:
            items = [item for item in items if item.get("level") == self.level]

        self.data = items
        return items

    def _load_from_huggingface(self) -> List[Dict[str, Any]]:
        """从HuggingFace下载GAIA数据集"""
        from huggingface_hub import snapshot_download
        import json

        # 下载数据集
        repo_id = "gaia-benchmark/GAIA"
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=self.local_data_dir,
            local_dir_use_symlinks=False
        )

        # 加载JSONL文件
        data_file = Path(local_dir) / "2023" / self.split / "metadata.jsonl"
        items = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                items.append(self._standardize_item(item))

        return items
````

## （2）GAIAEvaluator


（2）GAIAEvaluator：实现 GAIA 官方评估算法 

### 归一化和匹配

GAIA 的评估使用准精确匹配（Quasi Exact Match）算法，需要特殊的答案归一化和匹配逻辑：

````python
class GAIAEvaluator:
    """GAIA评估器

    实现GAIA官方的准精确匹配（Quasi Exact Match）评估算法
    """

    def evaluate(self, agent: Any, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """执行评估"""
        dataset_items = self.dataset.load()

        if max_samples:
            dataset_items = dataset_items[:max_samples]

        results = []
        for i, item in enumerate(dataset_items, 1):
            # 1. 构造提示词
            prompt = self._build_prompt(item["question"], item)

            # 2. 调用智能体
            response = agent.run(prompt)

            # 3. 提取答案（GAIA格式：FINAL ANSWER: [答案]）
            predicted_answer = self._extract_answer(response)

            # 4. 归一化答案（GAIA官方规则）
            normalized_pred = self._normalize_answer(predicted_answer)
            normalized_truth = self._normalize_answer(item["final_answer"])

            # 5. 准精确匹配
            exact_match = (normalized_pred == normalized_truth)

            results.append({
                "task_id": item["task_id"],
                "predicted": predicted_answer,
                "expected": item["final_answer"],
                "exact_match": exact_match,
                "level": item.get("level", 0)
            })

        return self._format_results(results)
````

### 处理不同类型答案

GAIA 使用特定的归一化规则来处理不同类型的答案：

```python
def _normalize_answer(self, answer: str) -> str:
    """标准化答案字符串（GAIA官方标准化规则）

    规则：
    1. 数字：移除逗号分隔符和单位符号
    2. 字符串：移除冠词、转小写、移除多余空格
    3. 列表：逗号分隔，按字母顺序排序
    """
    if not answer:
        return ""

    answer = answer.strip()

    # 检查是否是逗号分隔的列表
    if ',' in answer:
        parts = [self._normalize_single_answer(p.strip()) for p in answer.split(',')]
        parts.sort()  # GAIA要求按字母顺序排序
        return ','.join(parts)
    else:
        return self._normalize_single_answer(answer)

def _normalize_single_answer(self, answer: str) -> str:
    """标准化单个答案（不包含逗号的答案）"""
    answer = answer.strip().lower()

    # 移除常见的冠词
    articles = ['the', 'a', 'an']
    words = answer.split()
    if words and words[0] in articles:
        words = words[1:]
        answer = ' '.join(words)

    # 移除货币符号和百分号
    answer = answer.replace('$', '').replace('%', '').replace('€', '').replace('£', '')

    # 移除数字中的逗号分隔符
    answer = re.sub(r'(\d),(\d)', r'\1\2', answer)

    # 移除多余空格
    answer = ' '.join(answer.split())

    # 移除末尾的标点符号
    answer = answer.rstrip('.,;:!?')

    return answer
```

### 输出

GAIA 要求模型输出格式为 `FINAL ANSWER: [答案]`：

```python
def _extract_answer(self, response: str) -> str:
    """从响应中提取答案（GAIA格式）

    GAIA要求答案格式为：FINAL ANSWER: [答案]
    """
    # 首先尝试提取GAIA官方格式的答案
    final_answer_pattern = r'FINAL ANSWER:\s*(.+?)(?:\n|$)'
    match = re.search(final_answer_pattern, response, re.IGNORECASE | re.MULTILINE)
    if match:
        answer = match.group(1).strip()
        # 移除可能的方括号
        answer = answer.strip('[]')
        return answer

    # 备用方案：查找其他答案标记
    answer_patterns = [
        r'答案[：:]\s*(.+)',
        r'最终答案[：:]\s*(.+)',
        r'Final answer[：:]\s*(.+)',
        r'Answer[：:]\s*(.+)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # 如果没有找到标记，返回最后一个非空行
    lines = response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('#'):
            return line

    return response.strip()
```

### 导出

评估完成后，可以导出为 GAIA 官方要求的 JSONL 格式：

```python
def export_to_gaia_format(
    self,
    results: Dict[str, Any],
    output_path: Union[str, Path],
    include_reasoning: bool = True
) -> None:
    """导出为GAIA官方格式（JSONL）

    GAIA要求的格式：
    {"task_id": "xxx", "model_answer": "答案", "reasoning_trace": "推理过程"}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results.get("detailed_results", []):
            entry = {
                "task_id": result["task_id"],
                "model_answer": result["predicted"]
            }

            if include_reasoning:
                entry["reasoning_trace"] = result.get("response", result["predicted"])

            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
```

## （3）GAIAEvaluationTool


（3）GAIAEvaluationTool：一键评估工具

GAIAEvaluationTool 封装了完整的评估流程，提供一键评估功能：

````python
class GAIAEvaluationTool(Tool):
    """GAIA评估工具

    提供一键评估功能：
    1. 运行HelloAgents评估
    2. 导出GAIA格式结果
    3. 生成评估报告
    4. 生成提交说明
    """

    def run(
        self,
        agent: Any,
        level: Optional[int] = None,
        max_samples: Optional[int] = None,
        local_data_dir: Optional[str] = None,
        export_results: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """执行GAIA一键评估"""
        # 步骤1: 运行HelloAgents评估
        results = self._run_evaluation(agent, level, max_samples, local_data_dir)

        # 步骤2: 导出GAIA格式结果
        if export_results:
            self._export_results(results)

        # 步骤3: 生成评估报告
        if generate_report:
            self.generate_report(results)

        return results
````

GAIAEvaluationTool 会自动生成评估报告：

```python
def generate_report(
    self,
    results: Dict[str, Any],
    output_file: Optional[Union[str, Path]] = None
) -> str:
    """生成评估报告"""
    report = f"""# GAIA评估报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 评估概览

- **智能体**: {results.get("agent_name", "Unknown")}
- **难度级别**: {results.get("level_filter") or '全部'}
- **总样本数**: {results.get("total_samples", 0)}
- **精确匹配数**: {results.get("exact_matches", 0)}
- **精确匹配率**: {results.get("exact_match_rate", 0):.2%}

## 📈 详细指标

### 分级准确率

{self._format_level_metrics(results.get("level_metrics", {}))}

## 📝 样本详情（前10个）

{self._format_sample_details(results.get("detailed_results", [])[:10])}

## 📊 准确率可视化

{self._format_visualization(results.get("exact_match_rate", 0))}

## 💡 建议

{self._format_suggestions(results.get("exact_match_rate", 0))}
"""

    # 保存报告
    if output_file is None:
        output_dir = Path("./evaluation_reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"gaia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    return report
```
