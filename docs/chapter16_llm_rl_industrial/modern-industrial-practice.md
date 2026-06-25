# 9.9 现代工业实战 与 GLM-4.6、Llama 4 与 MuonClip

[9.8 节工业实践](./industrial-post-training) 讨论了 2024-2025 年的主流工业方案——MiniMax、Qwen、Kimi、Seed、DeepSeek。这一节我们补充几个 2025-2026 年的最新工业实践：

- **GLM-4.5 / GLM-4.6**（智谱）：中国开源推理模型新秀
- **Llama 4**（Meta）：开源旗舰的进化
- **Seed-Thinking**（字节 Seed）：reasoning model 工业配方
- **MuonClip + QK-clip**（Kimi K2）：训练稳定性的新工具

这些工作代表了 LLM RL 工业训练的**最新 SOTA**——把 [前面章节](./intro) 的算法用到极致。

## 9.9.1 智谱 GLM-4.5 / GLM-4.6

[GLM-4.5](https://github.com/zhipuai-llm/GLM-4.5)（智谱 AI, 2025.07 发布）和 [GLM-4.6](https://github.com/zhipuai-llm/GLM-4.6)（2025.10）是中国开源推理模型的重要进展。

### GLM 系列的特点

GLM（General Language Model）系列与 Qwen、DeepSeek 的差异：

- **Mixture of Experts (MoE)**：GLM-4.5 用 MoE 架构，355B 总参数 / 32B 激活
- **双模式**：thinking / non-thinking，与 Qwen3 类似
- **完全开源**：权重 + 训练方法 + 部分数据
- **代码能力**：特别强化了代码生成与 agentic 能力

### GLM-4.5 的训练流程

```text
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Base 预训练（MoE 架构）                          │
│   - 15T tokens 高质量数据                                 │
│   - MoE: 355B total / 32B active                         │
│   - RoPE scaling 支持长 context                          │
├──────────────────────────────────────────────────────────┤
│ Phase 2: 通用 SFT                                         │
│   - 多语言对话数据                                        │
│   - 工具调用格式训练                                      │
├──────────────────────────────────────────────────────────┤
│ Phase 3: 推理 RL                                          │
│   - 数学、代码、推理任务                                  │
│   - GRPO + 规则奖励                                      │
│   - Self-validation 集成                                  │
├──────────────────────────────────────────────────────────┤
│ Phase 4: 通用 RLHF                                        │
│   - 对话质量、安全性                                      │
│   - Helpfulness / Harmlessness 双目标                    │
├──────────────────────────────────────────────────────────┤
│ Phase 5: Thinking / Non-Thinking 统一                    │
│   - 混合数据 SFT                                          │
│   - 让模型学会模式切换                                    │
└──────────────────────────────────────────────────────────┘
```

这个流程与 [DeepSeek-R1 的训练流程](../chapter18_grpo/deepseek-dapo) 高度相似——都是 SFT + 推理 RL + 通用 RLHF 的多阶段范式。

### GLM-4.6 的改进

2025.10 的 GLM-4.6 升级：

- **更长的 thinking**：支持 100K+ token 的长 CoT
- **更强的 agentic**：内部集成了更多工具（搜索、代码执行、文件操作）
- **更好的多模态**：与 GLM-4.5V 视觉模型协同
- **更细的 thinking budget**：用户可以指定 budget（与 Qwen3 类似）

### GLM-4.6 的 benchmark 表现

| Benchmark     | GLM-4.5 | GLM-4.6 |
| ------------- | ------- | ------- |
| AIME 2025     | 75.3    | 83.6    |
| MATH-500      | 92.1    | 95.4    |
| LiveCodeBench | 56.2    | 62.7    |
| GPQA Diamond  | 68.5    | 72.4    |

GLM-4.6 在多项 benchmark 上达到开源 SOTA，与 Claude Opus 4.5 / GPT-5 接近。

### GLM 的工业意义

GLM 系列的工业意义：

1. **开源推理模型的多样性**——不止 Qwen、DeepSeek，GLM 是第三个有影响力的中国开源模型
2. **MoE 架构在推理 RL 上的验证**——证明了 [GSPO](../chapter18_grpo/grpo-family) 在 MoE 上的有效性
3. **代码 + 推理的整合**——GLM-4.6 特别强调 agentic 能力，与 Claude Code 等竞争

## 9.9.2 Meta Llama 4

[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)（Meta, 2025.04 发布）是 Meta 的开源旗舰。

### Llama 4 系列

Llama 4 包含三个变体：

- **Llama 4 Scout**：109B total / 17B active（MoE），10M context
- **Llama 4 Maverick**：400B total / 17B active，1M context
- **Llama 4 Behemoth**（未发布）：2T total / 288B active，Meta 内部训练

### Llama 4 的关键创新

**创新 1：原生多模态**

Llama 4 不是先训文本再加视觉，而是**从头开始多模态训练**——文本和图像 token 同等对待。这让 Llama 4 在视觉理解上优于"后接视觉"的方案。

**创新 2：Early Fusion**

不同于后期融合（先各自处理文本和图像再合并），Llama 4 用 **early fusion**——在模型早期就融合多模态信息。

**创新 3：MoE 架构**

Llama 4 全系采用 MoE——这是 Meta 第一次大规模 MoE。MoE 让模型在固定激活参数下，可以扩展总参数（提高能力，不增加推理成本）。

**创新 4：超长 Context**

Llama 4 Scout 支持 **10M token context**——这是当时开源模型中最长的。通过 iRoPE（interleaved RoPE）和 attention sparsity 实现。

### Llama 4 的训练方法

Meta 没有公开 Llama 4 的完整训练细节，但从论文和博客可以推断：

```text
┌─────────────────────────────────────────────────────────┐
│ Phase 1: 多模态预训练                                   │
│   - 文本 + 图像 + 视频联合训练                          │
│   - 22T tokens 级别（推测）                            │
│   - Early fusion 架构                                  │
├─────────────────────────────────────────────────────────┤
│ Phase 2: Mid-training（中等规模 SFT）                  │
│   - 通用指令跟随                                        │
│   - 工具调用格式                                        │
├─────────────────────────────────────────────────────────┤
│ Phase 3: 后训练 RL                                      │
│   - RLHF + RLVR 混合                                   │
│   - Helpfulness / Safety / Reasoning 多目标           │
└─────────────────────────────────────────────────────────┘
```

### Llama 4 的争议

Llama 4 发布后引发了一些争议：

**争议一：Benchmark 与实际表现差距**

Llama 4 Maverick 在多个 benchmark 上分数高，但用户实际使用时感觉不如 Claude 3.5 / GPT-5。Meta 后来承认 benchmark 评估与实际体验有差距。

**争议二：Maverick 的"特殊版本"**

LM Arena 上跑的 Maverick 是一个**经过专门优化的版本**——使用了 chat template 调整和 prompt engineering。开源的 Maverick 与 arena 版本有差异。

这个事件是 [Qwen3 数据污染](../chapter30_alignment_failures/modern-incidents) 类似的诚信问题——**benchmark 评估的脆弱性**。

### Llama 4 的工业意义

尽管有争议，Llama 4 仍然是开源 LLM 的重要进展：

1. **开源 MoE 的成熟**——证明 MoE 在开源生态的可行性
2. **多模态新范式**——early fusion 是后续工作的参考
3. **超长 context**——10M context 开启新应用场景

## 9.9.3 Seed-Thinking 与 字节的 Reasoning 配方

[Seed-Thinking 1.5](https://arxiv.org/abs/2505.07083)（字节 Seed, 2025.05）是字节对 reasoning model 工业训练的系统总结。

### Seed-Thinking 的核心贡献

Seed-Thinking 不是一个新算法，而是**工业配方的系统化**——把多个组件组合起来，达到 SOTA：

**组件 1：数据 curation**

```text
数学数据：
  - 高质量数学题（AIME、Putnam 历年题）
  - 自动生成题（用强 LLM 生成新题）
  - 难度分级（按 base model 的通过率）

代码数据：
  - Codeforces 题目（带测试用例）
  - SWE-bench / SWE-smith（带 PR 数据）
  - 函数生成（HumanEval 扩展）
```

**组件 2：GRPO + DAPO 改进**

Seed-Thinking 用了 [DAPO](../chapter18_grpo/deepseek-dapo) 的四项工程改造 + 一些新改进：

- **动态 KL**：训练初期 KL 强，后期减弱
- **Adaptive clip**：根据训练进度调整 clip range
- **Group size 调度**：早期大 group，后期小 group

**组件 3：Self-Verification**

让模型在生成答案后做自我验证：

```python
def self_verification_reward(response, ground_truth):
    answer = extract_answer(response)

    # 让模型重新读题、验证答案
    verification_prompt = f"检查这个答案是否正确：{answer}"
    verification = model.generate(verification_prompt)

    if "正确" in verification and answer == ground_truth:
        return 1.0  # 答对 + 验证通过
    elif "错误" in verification and answer != ground_truth:
        return 0.5  # 答错但识别出错误
    else:
        return 0.0  # 答错且没识别
```

这种 self-verification reward 让模型学会**反思**——不只是答对，还要能识别错误。

**组件 4：Curriculum Learning**

按难度排序训练数据，先训简单的，再训复杂的。课程学习让训练更稳定，避免训练初期 reward 信号太稀疏。

### Seed-Thinking 的成绩

Seed-Thinking 1.5 在多个 benchmark 上：

| Benchmark         | 分数  |
| ----------------- | ----- |
| AIME 2024         | 86.4% |
| MATH-500          | 96.2% |
| GPQA Diamond      | 75.1% |
| Codeforces Rating | 1822  |

这是字节 Seed 内部 reasoning model 的核心配方——后来用于豆包 Pro 推理版等产品。

## 9.9.4 Kimi K2 的 MuonClip + QK-clip

[Kimi K2](https://arxiv.org/abs/2507.14432)（Moonshot, 2025.07）的工业贡献之一是 **MuonClip + QK-clip**——训练稳定性的新工具。

### Muon optimizer

[Muon](https://arxiv.org/abs/2502.14682)（2025.02）是新提出的 optimizer——Muon (Momentum + Orthogonalization)。它结合了：

- **Momentum**（来自 Adam）
- **Orthogonalization**（对梯度做正交化）

正交化让更新方向更稳定——避免 Adam 在某些方向震荡。

### MuonClip 的核心

MuonClip 在 Muon 基础上加入了 **clip**：

```python
def muon_clip_update(grad, momentum, clip_threshold=1.0):
    # Muon 主流程
    momentum = beta * momentum + (1 - beta) * grad
    orthogonalized = orthogonalize(momentum)

    # Clip 防止爆炸
    norm = torch.norm(orthogonalized)
    if norm > clip_threshold:
        orthogonalized = orthogonalized * (clip_threshold / norm)

    return -lr * orthogonalized
```

Clip 在大规模训练中非常重要——避免单个 outlier 梯度破坏整个训练。

### QK-clip 与 注意力稳定化

QK-clip 是 Kimi K2 的另一个创新——**裁剪 attention 的 Q·K 乘积**：

```python
def attention_with_qk_clip(Q, K, V, clip_value=30.0):
    # 标准 attention
    scores = Q @ K.T / sqrt(d)

    # QK-clip: 防止 attention scores 过大
    scores = torch.clamp(scores, min=-clip_value, max=clip_value)

    # Softmax + 加权
    attn = softmax(scores)
    output = attn @ V

    return output
```

**为什么需要 QK-clip？**

在长 context 训练中，attention scores 可能因为 **attention sink**（某些 token 吸引过多注意力）而变得极大——破坏 softmax 分布，导致梯度爆炸。

QK-clip 通过限制 scores 的范围，避免这种问题。

### Kimi K2 的训练效果

用 MuonClip + QK-clip，Kimi K2 在大规模训练中：

- **训练稳定性**：从平均每 1T tokens 一次 loss spike，提升到每 10T tokens 一次
- **训练速度**：比 Adam 快约 15%
- **最终性能**：Kimi K2 在多项 benchmark 上达到开源 SOTA

### MuonClip 的工业意义

MuonClip 是训练**超大规模 LLM**的关键工具：

- **万亿参数级训练**：传统 Adam 在万亿参数上不稳定，MuonClip 解决
- **超长 context**：QK-clip 让 1M+ context 训练可行
- **开源生态**：Muon 已经在开源社区普及（OpenLM、PyTorch 都支持）

## 9.9.5 中国工业实践的总结

到 2026 年中，中国 LLM RL 工业实践的格局：

| 厂商              | 旗舰模型           | RL 算法         | 工业贡献      |
| ----------------- | ------------------ | --------------- | ------------- |
| **DeepSeek**      | R1, V3.2           | GRPO + 改进     | 透明度、开源  |
| **阿里 Qwen**     | Qwen3 系列         | GSPO            | MoE 训练稳定  |
| **字节 Seed**     | 豆包 Pro, Seedance | DAPO + VAPO     | 多线并行      |
| **Moonshot Kimi** | K2, K2.5           | GRPO + MuonClip | 训练稳定工具  |
| **智谱 GLM**      | GLM-4.6            | GSPO-style      | 开源 MoE      |
| **MiniMax**       | M1, M2             | CISPO           | 低精度训练    |
| **StepFun**       | Step3              | 内部方法        | 推理 + 多模态 |

可以看到：

1. **每家厂商都有自己的"招牌 RL 算法"**——GRPO、GSPO、DAPO、CISPO、VAPO、MuonClip
2. **算法创新主要来自中国**——这与美国闭源派（OpenAI、Anthropic）形成对比
3. **工业贡献互补**——不是替代，是从不同角度解决 RL 训练问题

## 9.9.6 未来的工业方向

### 万亿参数 + 超长 context

- 万亿参数 base model（DeepSeek V3 是 671B，下一代会到 1T+）
- 10M+ token context（Llama 4 Scout 已经支持）
- 训练稳定性是核心挑战——MuonClip 是方向

### 多模态原生 RL

- 不再是"文本 RL + 视觉 SFT"，而是"多模态联合 RL"
- Llama 4 的 early fusion 是开始
- 未来会有更多原生多模态 RL 算法

### Agentic RL 工业化

- Agent 训练成为主流（不只是 SWE，还有客服、研究、操作）
- Agent trajectory 数据是关键
- Agent RL infra 投入巨大

### 训练成本下降

- 2024 年训练一个 SOTA 模型需要 $100M+
- 2026 年可能下降到 $10M（更优算法 + 更便宜算力）
- 这让小团队也能参与 SOTA 研究

## 小结

现代 LLM RL 工业实践的核心趋势：

- **GLM-4.6 / Llama 4**：开源旗舰的进化，MoE + 多模态
- **Seed-Thinking**：reasoning model 的工业配方系统化
- **MuonClip + QK-clip**：训练稳定性的新工具
- **中国厂商主导算法创新**：每家都有自己的招牌方法

这些工作共同把 RL 在 LLM 训练中的地位**从可选变为核心**——没有 RL，就无法训出 SOTA 模型。

接下来：

- [第 10 章 Reasoning Models](../chapter19_reasoning/intro)——推理模型的详细讨论
- [第 11 章 PRM](../chapter20_prm_search/intro)——过程奖励的工业实践
- [第 12 章 RL-based SWE](../chapter23_rl_based_swe/intro)——代码 agent 的训练
