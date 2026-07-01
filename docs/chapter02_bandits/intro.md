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

| 增长率             | 含义   | 评价                        |
| ------------------ | ------ | --------------------------- |
| $\Theta(T)$        | 线性   | 智能体没学到东西，纯随机    |
| $\Theta(\sqrt{T})$ | 次线性 | 标准好策略（UCB、Thompson） |
| $\Theta(\log T)$   | 对数   | 理论下界（Lai-Robins 1985） |

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

固定 $\epsilon$ 的好处是简单稳定：无论当前估计多么确定，智能体都会保留一小部分尝试机会。这能防止早期几次坏运气把某个好摇臂过早排除。
它的代价也很直接：即使已经基本知道哪个摇臂最好，智能体仍然会按固定比例随机探索，长期看会浪费一部分选择机会。

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

固定 $\epsilon$ 简单可靠，但会长期浪费一部分探索机会；实际训练中更常见的是让 $\epsilon$ 随时间下降，使智能体从"多尝试"逐步转向"相信已有经验"。这个思想在深度 RL 中也广泛使用——Atari DQN 的 ε 会从较大的初始值逐步衰减到较小值。

::: details 补充：ε-贪心的理论性质

固定 ε-贪心会持续保留随机探索，因此即使已经知道哪个动作最好，也仍然会以 ε 的概率选择其他动作。这样做能防止早期误判，但长期看会产生持续损失，所以固定 ε 的累积遗憾通常按线性速度增长。

如果让 ε 随时间下降，例如从较大的初始值逐步衰减到很小，智能体会在早期多探索、后期多利用。合适的衰减策略可以把长期遗憾降到次线性；在一些更严格的设定下，可以得到接近对数级别的遗憾界。

PAC 样本复杂度换了一个问题问法：它不再问前 $T$ 轮一共损失多少，而是问"需要多少次尝试，才能以至少 $1-\delta$ 的概率找到一个距离最优不超过 $\epsilon$ 的动作"。这个视角主要用于理论分析，本课程只需要知道它是在衡量"学到足够好策略所需的样本量"。

:::

## 2.3 更有针对性的探索

ε-贪心解决了"不能只相信当前最优"的问题，但它的探索方式很粗：只要进入探索分支，就会在所有动作里随机选一个。这样会把机会分给那些已经明显很差的摇臂。

更聪明的探索会问一个更细的问题：**哪些动作还值得试？**

### UCB：给不确定性加奖金

UCB（Upper Confidence Bound）的做法是给每个摇臂的估计奖励加上一个"不确定性奖金"：

$$A_t = \arg\max_a \left[ \hat{\mu}_a + c \sqrt{\frac{\ln t}{N_a}} \right]$$

这里 $\hat{\mu}_a$ 是摇臂 $a$ 的当前平均奖励，$N_a$ 是它已经被尝试的次数。一个摇臂被试得越少，$N_a$ 越小，不确定性奖金越大；被试得越多，奖金越小，选择就更依赖真实估计值。

UCB 的直觉很直接：如果一个动作当前表现不错，就继续利用；如果一个动作试得太少，还不能确定它差，就给它一些额外机会。探索不再平均分给所有动作，而是集中到"可能好但还不确定"的动作上。

::: details 补充：UCB 的遗憾界

UCB1 的遗憾可以证明达到对数级别。常见形式是：

$$\mathbb{E}[\text{Regret}(T)] \leq 8 \sum_{a: \mu_a < \mu^*} \frac{\ln T}{\Delta_a} + \left(1 + \frac{\pi^2}{3}\right) \sum_{a: \mu_a < \mu^*} \Delta_a$$

其中 $\Delta_a = \mu^* - \mu_a$ 是次优摇臂与最优摇臂的差距。第一遍学习不需要记住这个式子，只需要理解它表达的机制：UCB 会逐渐减少对明显次优动作的尝试，把探索集中在真正难以区分的动作上。

:::

### Thompson 采样：按"成为最优"的可能性选择

Thompson 采样换了一个视角。它不直接给不确定性加奖金，而是为每个摇臂维护一个后验分布，表示"这个摇臂的真实平均奖励可能是多少"。每一轮从所有后验分布里各采样一次，然后选择采样值最大的摇臂。

如果某个摇臂已经试了很多次，后验分布会很窄，采样值通常接近当前估计；如果某个摇臂试得很少，后验分布更宽，它就有机会采到较高的值并获得探索机会。这样，探索会自然流向那些仍有希望成为最优的动作。

在 0/1 奖励场景中，可以用 Beta 分布表示每个摇臂的后验：

$$\mu_a \sim \text{Beta}(\alpha_a, \beta_a)$$

拉到摇臂 $a$ 并得到奖励 1，就令 $\alpha_a \leftarrow \alpha_a + 1$；得到奖励 0，就令 $\beta_a \leftarrow \beta_a + 1$。这个更新规则很轻量，因此 Thompson 采样常用于推荐、广告和 A/B 测试系统。

::: details 补充：Thompson 采样的理论视角

Thompson 采样的理论分析常使用贝叶斯遗憾：先假设问题本身来自一个先验分布，再计算策略在这个先验下的期望损失。在常见随机老虎机设定下，它可以达到与 UCB 同阶的对数级别遗憾。

这个细节不是本章主线。主线只需要理解：Thompson 采样把"某个动作可能是最优"转化成选择概率，因此它比 ε-贪心更少浪费探索机会。

:::

### 上下文老虎机与 RLHF

普通多臂老虎机假设每个摇臂的奖励分布固定不变。但在真实系统里，动作好不好常常取决于当前输入。

推荐系统里，同一篇文章对不同用户的吸引力不同；广告系统里，同一条广告面对不同人群会有不同点击率；大语言模型里，同一段回答的质量也必须放在具体问题下判断。

这就得到**上下文老虎机（Contextual Bandit）**：

- 每轮先观察上下文 $x_t$
- 再选择动作 $A_t$
- 收到奖励 $R_t$
- 目标是学习策略 $\pi(a \mid x)$，让动作依赖于当前上下文

从这个角度看，RLHF 可以先被理解成一种上下文老虎机问题：prompt 是上下文，模型生成的回答是动作，奖励模型给出的分数是奖励。这个抽象还没有包含完整的 token 级状态转移，但它已经解释了一个关键事实：模型不能只学习"哪个回答整体更常见"，而要学习"在这个 prompt 下，哪类回答更合适"。

后续进入 MDP 之后，我们会把这个问题继续展开：当一个回答由多个 token 逐步生成时，动作不再是一次性选择，当前 token 会改变后续状态和可选动作。这时，老虎机问题就扩展成真正的序列决策问题。

## 本章总结

多臂老虎机（MAB）是 RL 最简化的形式——无状态、即时奖励，但完整保留了"探索-利用"的张力。ε-贪心用固定概率保留尝试机会，是理解探索的起点；UCB 把尝试机会分配给不确定性更高的动作；Thompson 采样用后验采样把"可能是最优"转化成选择概率。

多臂老虎机没有状态转移，奖励也会立刻出现。下一章 [马尔可夫决策过程](../chapter03_mdp/mdp) 会加入状态转移和长期回报，真正进入序列决策问题。

## 延伸阅读

- Sutton & Barto《Reinforcement Learning: An Introduction》第 2 章
- [Auer et al. 2002 "Finite-time Analysis of the Multiarmed Bandit Problem"](https://link.springer.com/article/10.1023/A:1013689704352)
- [Russo et al. 2018 "A Tutorial on Thompson Sampling"](https://arxiv.org/abs/1707.02038)
- [Lattimore & Szepesvári《Bandit Algorithms》](https://banditalgs.com/)
