# 第 18 章 · 过程奖励模型与推理时搜索

上一章我们看到推理模型在硬任务上取得了突破——o1、DeepSeek-R1、Claude Opus 4.6 都能做复杂数学、代码、科研推理。但这些模型都依赖一个关键假设：**最终答案的对错可以作为 RL 的奖励信号**。

这个假设在简单任务上成立——一道数学题答对就是答对，答错就是答错。但随着任务变复杂，这个假设开始失效：

- **长 CoT 任务**：一个 10000 token 的推理链，可能前 8000 token 都对，最后 2000 token 错。模型只看到"答错了"，不知道哪里错了
- **代码生成**：一个程序编译失败，是哪一行错了？哪个逻辑错了？
- **多步 agent 任务**：10 步 trajectory 失败了，是第几步失败的？

这就是 **稀疏奖励问题（sparse reward problem）**——奖励信号只在序列结尾出现，中间步骤得不到反馈。**过程奖励模型（Process Reward Model, PRM）** 就是为这个问题而生：它给推理过程的每一步打分，把稀疏奖励变成密集奖励。

## 这一章要回答的问题

- **Outcome 奖励 vs Process 奖励**的本质区别是什么？
- **判别式 PRM**（OpenAI 的经典路线）怎么工作？标注成本为什么是瓶颈？
- **生成式 PRM**（ThinkPRM）为什么用更少标签就能超越判别式？
- **形式化 PRM**（AlphaProof、Lean4）怎么实现"零误判"验证？
- **推理时搜索**（MCTS、Tree of Thoughts、Beam Search）怎么用 PRM 引导？
- **并行协调推理**（PaCoRe）如何替代传统的深度优先推理？

## 章节地图

```text
11.1 Outcome 奖励 vs Process 奖励
     ├── 稀疏奖励问题
     ├── 信用分配的本质
     └── 为什么 PRM 在长 CoT 任务里不可替代
11.2 判别式 PRM（经典路线）
     ├── OpenAI "Let's Verify Step by Step"
     ├── PRM800K 数据集
     ├── PRM 作为 Re-ranking 模型
     └── 局限：标注成本高、泛化弱
11.3 生成式 PRM（新路线）
     ├── ThinkPRM：生成式 verifier
     ├── 标签少 100 倍的关键
     ├── Verifier Compute Scaling
     └── 生成式 vs 判别式对比
11.4 形式化 PRM（终极 verifier）
     ├── Lean4 / Coq：零误判验证
     ├── AlphaProof：IMO 银牌
     ├── AlphaGeometry 2：几何专用
     └── DeepSeek-Prover-V2：MiniF2F 88.9%
11.5 推理时搜索
     ├── Beam Search over Thoughts
     ├── MCTS over Thoughts
     ├── Tree of Thoughts
     └── AlphaCodium / rStar
11.6 并行协调推理（PaCoRe）
     ├── 16 路并行 rollout
     ├── outcome-based RL 训练
     ├── AIME 2025: 94.4
     └── 深度 vs 广度的权衡
11.7 GenRM 与 Verifier 模型
     ├── Generative Reward Model
     ├── LLM-as-Judge
     └── Self-Rewarding Language Models
```

## 与其他章的关系

这一章假定你已经读过：

- [第 13 章 RLHF 奖励模型](../chapter15_rlhf/reward-function-design)——Outcome Reward Model 的基础
- [第 16 章 GRPO 改进家族](../chapter18_grpo/grpo-family)——信用分配问题在 GRPO 中的体现
- [第 17 章推理模型](../chapter19_reasoning/intro)——为什么推理模型需要 PRM

本章后续会指向：

- [第 20 章 Agentic RL](../chapter22_agentic/intro)——多步 trajectory 的过程奖励
- [第 28 章奖励黑客](../chapter30_alignment_failures/intro)——PRM 的 reward hacking 问题

## 一个直觉性的开场

在进入正式内容前，先建立两个关键直觉：

**直觉一：PRM 是把"考试评分"变成"作业批改"**。传统 outcome reward 就像考试评分——只看最终答案对不对，对了 100 分，错了 0 分。PRM 就像老师批改作业——每一步推理都打分，对的步骤给正分，错的步骤给负分，半对的步骤给部分分。批改虽然更费时，但反馈更细致，学生（模型）能学到更多。

**直觉二：PRM 是 verifier，不是 policy**。一个常见的混淆是把 PRM 当作"另一种 reward model"。严格来说，PRM 是一个 **verifier**——它的工作不是"生成好的推理"，而是"判断推理好不好"。Verifier 和 policy 的训练目标不同：policy 要学会"做什么"，verifier 要学会"评价什么"。这一区分对理解后续的 GenRM、LLM-as-Judge 很重要。

带着这两个直觉，我们进入 11.1。
