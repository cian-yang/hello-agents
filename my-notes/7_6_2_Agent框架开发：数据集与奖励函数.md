# 背景

数据集和奖励函数是强化学习训练的两大基石。

- **数据集定义了智能体要学习的任务**，
- **奖励函数定义了什么是好的行为**。

# GSM8K 数学推理数据集

数学推理是评估 LLM 推理能力的理想任务。

## 为什么使用数学推理

首先，**数学问题有明确的正确答案**，**可以自动评估**，不需要人工标注或复杂的奖励模型。

其次，**解决数学问题需要分解问题、逐步推导**，这正是**多步推理的典型场景**。

最后，**学到的推理能力可以迁移到其他领域**，具有很强的**泛化性**。

相比之下，开放式问答任务(如"如何学习编程?")的答案质量难以客观评估，需要大量人工标注。

## GSM8K 是什么

GSM8K(Grade School Math 8K)是一个高质量的小学数学应用题数据集。

- 数据集包含 7，473 个训练样本和 1，319 个测试样本，
- 难度为小学数学水平(2-8 年级)，
- 题型为应用题，
- 需要 2-8 步推理才能得出答案。

典型的 GSM8K 问题：这个问题需要两步推理:首先计算 5 月份卖出的数量(48 的一半)，然后计算总数(4 月+5 月)。

- 答案中的 `<<48/2=24>>`是中间计算步骤的标记，
- `#### 72`标记最终答案

```
问题: Natalia sold clips to 48 of her friends in April, and then she sold half 
      as many clips in May. How many clips did Natalia sell altogether in April 
      and May?

答案: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
      Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
      #### 72

最终答案: 72

```

## 格式转换

GSM8K 数据集需要转换为不同的格式，以适应不同的训练方法，

![1776348009530](image/1776348009530.png)

### 原始格式


原始格式直接来自数据集，包含问题(question)和答案(answer，含解题步骤)，适合人类阅读。SFT 格式用于监督微调，将问题转换为对话格式的 prompt，将完整解答作为 completion。

例如:

```python
{
    "prompt": "<|im_start|>user\nNatalia sold clips to 48 of her friends...<|im_end|>\n<|im_start|>assistant\n",
    "completion": "Let me solve this step by step.\n\nStep 1: ...\n\nFinal Answer: 72<|im_end|>"
}
```

关键点如下，这样模型可以学习如何格式化输出、如何分步推理。

- 使用模型的对话模板(如 Qwen 的 `<|im_start|>`标记)，
- prompt 包含用户问题，
- completion 包含完整的解题过程和答案。

### RL 格式


RL 格式用于强化学习，只提供问题和正确答案，不提供解题过程。

prompt 与 SFT 相同，但 ground_truth 只包含最终答案(用于计算奖励)，模型需要自己生成完整的推理过程。

这种设计迫使模型学会自主推理，而不是简单地记忆答案。

例如:

```python
{
    "prompt": "<|im_start|>user\nNatalia sold clips to 48 of her friends...<|im_end|>\n<|im_start|>assistant\n",
    "ground_truth": "72"
}
```



### 格式选择


| 格式 | 用途     | 标签内容       | 特点     |
| ---- | -------- | -------------- | -------- |
| 原始 | 数据存储 | answcr(含步骤) | 人类可读 |
| SFT  | 监督学习 | 完整解答       | 学习格式 |
| RL   | 强化学习 | 仅答案         | 自主推理 |


### 示例

通过代码来加载和查看数据集:

- SFT 格式包含完整的解题过程，用于监督学习;
- RL 格式只包含最终答案，模型需要自己生成推理过程。
- `max_samples`参数控制加载的样本数量，方便快速测试。

```python
from hello_agents.tools import RLTrainingTool
import json

# 创建工具
rl_tool = RLTrainingTool()

# 1. 加载SFT格式数据集
sft_result = rl_tool.run({
    "action": "load_dataset",
    "format": "sft",
    "max_samples": 5  # 只加载5个样本查看
})
sft_data = json.loads(sft_result)

print(f"数据集大小: {sft_data['dataset_size']}")
print(f"数据格式: {sft_data['format']}")
print(f"样本字段: {sft_data['sample_keys']}")

# 2. 加载RL格式数据集
rl_result = rl_tool.run({
    "action": "load_dataset",
    "format": "rl",
    "max_samples": 5
})
rl_data = json.loads(rl_result)

print(f"数据集大小: {rl_data['dataset_size']}")
print(f"数据格式: {rl_data['format']}")
print(f"样本字段: {rl_data['sample_keys']}")

```

# 奖励函数设计

## 概念

奖励函数是强化学习的核心，它定义了什么是"好的行为"。奖励函数的设计直接影响训练效果。

- 好的奖励函数应该能清楚地定义什么是成功、能够提供梯度信号、不会产生过大的方差、容易调整和组合。
- 糟糕的奖励函数可能只在任务结束时给奖励，中间步骤无反馈、存在奖励欺骗，使得智能体找到"作弊"方式获得高奖励、多个目标相互矛盾、方差过大，训练不收敛。

## 公式

在强化学习中，奖励函数 $r(s, a)$ 或 $r(s, a, s')$ 为智能体的每个行动分配一个数值奖励。

**智能体的目标是最大化累积奖励**:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]
$$

对于数学推理任务，我们可以简化为:

- 其中 $q$ 是问题，
- $a$ 是模型生成的答案，
- $a^*$ 是正确答案，
- $f$ 是评估函数。

$$
r(q, a) = f(a, a^*)
$$

## 奖励函数

HelloAgents 提供了三种内置奖励函数，可以单独使用或组合使用

![1776348836860](image/1776348836860.png)

### 准确率奖励

准确率奖励(AccuracyReward)是最基础的奖励函数，它只关心答案是否正确。

数学定义为：这是一个二值奖励函数，答案正确得 1 分，错误得 0 分。

- 其中 $a$ 是模型生成的答案，
- $a^*$ 是正确答案。

$$
r_{\text{acc}}(a, a^*) = \begin{cases}
1 & \text{if } a = a^* \\
0 & \text{otherwise}
\end{cases}
$$


实现时需要处理答案提取和比较。

模型的输出可能包含大量文本，我们需要提取最终答案。

常见的提取方法包括:

- 查找"Final Answer:"后的数字、
- 查找"####"标记后的数字、
- 使用正则表达式提取最后一个数字。


- 答案比较时需要处理数值精度(如 72.0 和 72 应该视为相同)、
- 单位转换(如 1000 和 1k)、
- 格式差异(如"72"和"seventy-two")。

使用示例:

```python
from hello_agents.tools import RLTrainingTool
import json
rl_tool = RLTrainingTool()

# 创建准确率奖励函数
reward_result = rl_tool.run({
    "action": "create_reward",
    "reward_type": "accuracy"
})
reward_data = json.loads(reward_result)

print(f"奖励类型: {reward_data['reward_type']}")
print(f"描述: {reward_data['description']}")

# 注意: RLTrainingTool的create_reward操作返回的是配置信息,
# 实际的奖励函数会在训练时自动创建和使用
```

输出:

```json
预测: 72, 真实: 72, 奖励: 1.0
预测: 72.0, 真实: 72, 奖励: 1.0
预测: 73, 真实: 72, 奖励: 0.0
```

准确率奖励的优点是简单直接，容易理解和实现，适合有明确正确答案的任务。缺点是奖励稀疏，只有答案完全正确才有奖励，无法区分"接近正确"和"完全错误"，可能导致训练初期缺乏有效反馈。
