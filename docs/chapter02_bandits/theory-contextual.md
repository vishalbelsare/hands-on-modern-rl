# 2.3 遗憾界、PAC 与上下文老虎机

> [2.2](./ucb-thompson) 讲了 UCB 和 Thompson 的算法实现。本节给出两个更深的视角：(1) **理论分析**——如何证明 MAB 算法的最优性？(2) **上下文扩展**——当最优动作依赖于上下文（如推荐系统中的用户特征）时，MAB 怎么扩展？这是通往 RLHF 的形式化桥梁。

## 遗憾界与 PAC 分析

```mermaid
flowchart LR
    A[Regret Bound<br/>累积遗憾] --> B[Sample Complexity<br/>PAC 框架]
    A --> C[Bayes Regret<br/>贝叶斯遗憾]
    B --> D[ε-greedy<br/>O(K/ε²·log 1/δ)]
    C --> E[Thompson<br/>O(√KT·log T)]
```

**两种评估视角：**

- **频率派（Regret Bound）**：最坏情况下，算法在 $T$ 轮内的累积遗憾上界。UCB、ε-贪心都属于此类。
- **贝叶斯（Bayes Regret）**：假设问题本身从一个先验分布采样，求期望遗憾。Thompson 采样的理论分析常走这条路。

**PAC 框架**（Probably Approximately Correct）：另一类评估。问"以 $1-\delta$ 概率，算法在 $T$ 轮内能找到 $\epsilon$-最优策略吗？" 答案给出了**样本复杂度** $T(\epsilon, \delta)$。

经典结论（Lai & Robbins 1985; Auer et al. 2002）：

| 算法 | 遗憾 | 样本复杂度 |
|------|------|----------|
| ε-贪心（固定 ε） | $\Theta(T)$ | $O(K/\epsilon^2 \log 1/\delta)$ |
| ε-贪心（衰减） | $O(\log T)$ | $O(K/\epsilon^2 \log 1/\delta)$ |
| UCB1 | $O(\log T)$ | $O(K/\epsilon^2 \log 1/\delta)$ |
| Thompson | $O(\log T)$ | $O(K/\epsilon^2 \log 1/\delta)$ |

Lai-Robbins 下界告诉我们：**没有任何策略能做到比 $\Theta(\log T)$ 更好的遗憾**（在渐近意义下）。UCB 和 Thompson 都达到了这个下界。

## 上下文老虎机与 LLM 对齐

普通 MAB 假设奖励分布是静态的——同一个摇臂永远给同样的分布。但现实问题中，**最优动作依赖于上下文**：

- 推荐系统：不同用户对不同内容的偏好不同
- 广告投放：同一广告对不同人群的点击率不同
- **RLHF**：同一个 prompt，不同回答的"好坏"依赖于 prompt 内容

这就是**上下文老虎机（Contextual Bandit）**问题：

- 每轮先观察**上下文** $x_t \in \mathcal{X}$
- 然后选动作 $A_t$
- 收到奖励 $R_t \sim P(R \mid x_t, A_t)$
- 目标：学一个策略 $\pi(a \mid x)$ 最大化 $\mathbb{E}[R \mid x, \pi(x)]$

### 线性模型 LinUCB 与 LinTS

最简单的上下文模型：假设奖励是上下文特征的线性函数：

$$\mathbb{E}[R \mid x, a] = \theta_a^\top x$$

为每个摇臂 $a$ 学一个 $\theta_a$。LinUCB（Li et al. 2010）用岭回归估计 $\theta_a$，加上不确定性奖金：

$$A_t = \arg\max_a \left[ \hat{\theta}_a^\top x_t + c \sqrt{x_t^\top (X_a^\top X_a + \lambda I)^{-1} x_t} \right]$$

LinUCB 是 Yahoo News 推荐系统的核心算法（2010 年代），显著提升了 CTR。

### 神经网络版 Neural Bandit

非线性奖励用神经网络近似 $\hat{r}(x, a; \theta)$。代表性工作：

- **NeuralUCB**（Zhou et al. 2020）：用神经网络 + 神经正切核（NTK）分析不确定性
- **SupBan**（Kassraie & Krause 2022）：子采样集成 + 神经网络

### 与 RLHF 的连接

RLHF 本质上是一个**上下文老虎机问题**的特例：

- 上下文 $x_t$ = 用户 prompt
- 动作 $a_t$ = 模型生成的回答
- 奖励 $R_t$ = Reward Model 给的回答打分

但比传统 CB 复杂得多：

- 动作空间是整个 token 序列（不是有限离散摇臂）
- 奖励来自学习到的 RM（而非真实分布）
- 单次决策影响后续 token（部分序列化）

第 16 章 RLHF 流水线、第 18 章 DPO、第 19 章 GRPO 都建立在这个抽象之上——理解了上下文老虎机的探索-利用本质，你就能理解为什么 GRPO 要采样多条 rollout、为什么 PPO 需要 importance sampling。

## 本章总结

多臂老虎机把 RL 的"探索-利用"难题剥离出来单独研究。三个核心收获：

1. **遗憾是衡量策略好坏的根本指标**——好的策略遗憾次线性增长。
2. **ε-贪心、UCB、Thompson 三大算法对应三种哲学**：随机探索、不确定性量化、贝叶斯后验采样。它们都达到 $\Theta(\log T)$ 渐近下界。
3. **上下文老虎机是 LLM 对齐的形式化桥梁**——RLHF、DPO、GRPO 都可以看作上下文老虎机的扩展。

下一章 [第 3 章 马尔可夫决策过程](../chapter03_mdp/mdp) 引入**状态转移**——真正的 RL 问题开始了。

## 延伸阅读

- Sutton & Barto《Reinforcement Learning: An Introduction》第 2 章是 MAB 的标准教材
- [Auer et al. 2002 "Finite-time Analysis of the Multiarmed Bandit Problem"](https://link.springer.com/article/10.1023/A:1013689704352) — UCB1 的原始论文
- [Russo et al. 2018 "A Tutorial on Thompson Sampling"](https://arxiv.org/abs/1707.02038) — Thompson 采样完整教程
- [Lattimore & Szepesvári《Bandit Algorithms》](https://banditalgs.com/) — MAB 的现代教材（在线免费）