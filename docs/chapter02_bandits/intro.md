# 第 2 章 · 多臂老虎机与探索-利用理论

> [第 1 章](../chapter01_cartpole/principles) 用一个公式定义了 RL 问题：找最优策略 $\pi^*$ 最大化累积回报。但有个细节被一笔带过——**智能体怎么知道哪个动作好？** 这是 RL 区别于监督学习的第一个根本难题：**探索 vs 利用**。本章用一个最简化的 RL 问题——多臂老虎机——把这个难题隔离出来研究。

## 2.1 多臂老虎机问题

多臂老虎机（Multi-Armed Bandit, MAB）是 RL 的"果蝇"——最简化的序列决策问题，却保留了"探索-利用"张力的全部本质。

### 问题定义

想象一台有 $K$ 个摇臂的赌场老虎机，每个摇臂 $a \in \{1, 2, \ldots, K\}$ 拉一下会以未知分布返回一个奖励。你的目标是在 $T$ 次拉动中最大化累积奖励。

形式化：每个摇臂对应一个未知奖励分布 $R_a$，其期望 $\mu_a = \mathbb{E}[R_a]$。智能体每轮 $t$ 选一个 $A_t \in \{1, \ldots, K\}$，观察奖励 $R_t \sim R_{A_t}$。目标：$\max \sum_{t=1}^T R_t$。

注意这里的简化：

- **没有状态转移**：每轮的选择不改变"环境状态"，奖励分布只依赖于摇臂。这把 MDP 的状态维度剥离掉，聚焦在"动作选择"本身。
- **奖励是即时的**：拉一下立刻看到结果，没有延迟奖励。
- **没有折扣**：所有轮次权重相同。

但即便这么简化，**最优动作 $a^* = \arg\max_a \mu_a$ 是未知的**——智能体必须通过尝试来估计每个 $\mu_a$。这就是"探索-利用"的根源。

### 遗憾

由于 $\mu_a$ 未知，智能体不可能每轮都选 $a^*$。我们用**遗憾（Regret）** 来衡量策略好坏：

$$\text{Regret}(T) = T \cdot \mu^* - \mathbb{E}\left[\sum_{t=1}^T R_t\right] = \sum_{t=1}^T \mathbb{E}[\mu^* - \mu_{A_t}]$$

其中 $\mu^* = \max_a \mu_a$。遗憾是"如果每轮都选最优，能多拿多少"的期望差。

**遗憾界（Regret Bound）** 是评估算法的根本指标。一个好策略的遗憾应该随 $T$ **次线性增长**（sublinear），即 $\text{Regret}(T) = o(T)$——这意味着随着 $T \to \infty$，智能体越来越接近最优，每轮的平均损失趋于 0。

| 增长率 | 含义 | 评价 |
|-------|------|------|
| $\Theta(T)$ | 线性 | 智能体没学到东西，纯随机 |
| $\Theta(\sqrt{T})$ | 次线性 | 标准好策略（UCB、Thompson） |
| $\Theta(\log T)$ | 对数 | 理论下界（Lai-Robins 1985） |

## 2.2 ε-贪心与衰减调度

最朴素的策略：大部分时间利用已知最优，偶尔随机探索。

### ε-贪心算法

```python
import numpy as np

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q = np.zeros(n_arms)         # 每个摇臂的估计期望
        self.n = np.zeros(n_arms)         # 每个摇臂被拉过的次数

    def select(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)   # 探索
        return np.argmax(self.q)                     # 利用

    def update(self, arm, reward):
        self.n[arm] += 1
        # 增量式更新均值：避免存储全部历史
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]
```

核心思想：

- **利用（概率 $1-\epsilon$）**：选当前估计最高的摇臂
- **探索（概率 $\epsilon$）**：随机选一个，包括"已知最差"的

为什么不能纯利用？因为初始估计 $q_a = 0$ 可能误导。如果第一次拉摇臂 1 得到 0，纯利用策略会永远认为它差——但也许它的真实期望是 0.7，只是运气不好。

ε-贪心的遗憾是 $\Theta(T)$ 在固定 $\epsilon$ 下，但**采样次数足够多后期望遗憾**会变成次线性。具体地：经过 $T$ 轮后，每个摇臂平均被探索 $\epsilon T / K$ 次，估计精度 $\sim 1/\sqrt{\epsilon T / K}$，所以总遗憾 $\sim \epsilon T \cdot \Delta + K / \epsilon$，最优 $\epsilon^* \sim \sqrt{K/T}$ 给出 $\Theta(\sqrt{KT})$ 遗憾。

### 衰减调度

固定 $\epsilon$ 永远浪费 $\epsilon$ 比例的探索。更聪明的方法：**早期多探索，后期多利用**。

```python
class EpsilonDecaying:
    def __init__(self, n_arms, epsilon_start=1.0, epsilon_end=0.01, decay=0.995):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay = decay
        # ... 其他同上

    def select(self):
        if np.random.random() < self.epsilon:
            arm = np.random.randint(self.n_arms)
        else:
            arm = np.argmax(self.q)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.decay)
        return arm
```

衰减调度（如 $\epsilon_t = 1/t$ 或指数衰减）能给出 $\Theta(\log T)$ 遗憾，逼近理论下界。这个思想在深度 RL 中也广泛使用——Atari DQN 的 ε 从 1.0 衰减到 0.1。

## 本节总结

多臂老虎机（MAB）是 RL 最简化的形式——无状态、即时奖励，但完整保留了"探索-利用"的张力。ε-贪心是最朴素的策略：以 ε 概率随机探索，1-ε 概率利用当前最优。固定 ε 的遗憾是 Θ(T)，衰减 ε 可达到 Θ(log T) 渐近下界。

下一节 [2.2 UCB 与 Thompson 采样](./ucb-thompson) 讲更聪明的探索策略——优先探索那些"可能很好但还不确定"的摇臂。
