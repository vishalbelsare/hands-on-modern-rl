# 2.2 UCB 与 Thompson 采样

> [2.1](./intro) 讲了 ε-贪心——简单但浪费探索预算（在已知好的摇臂上也探索）。本节给出两个更聪明的策略：**UCB** 用"不确定性奖金"优先探索可能好的摇臂，**Thompson 采样**用贝叶斯后验实现"概率匹配"。

## 上置信界（UCB）算法

ε-贪心有一个明显缺陷：**它随机探索所有摇臂，包括"已知很好"的**。这浪费了探索预算。更聪明的做法：**优先探索那些"可能很好但目前不确定"的摇臂**。

### 乐观面对不确定性

UCB（Upper Confidence Bound, Auer et al. 2002）的核心思想：**对每个摇臂的估计加上一个"不确定性奖金"**，选总和最大的。

$$A_t = \arg\max_a \left[ \hat{\mu}_a + c \sqrt{\frac{\ln t}{N_a}} \right]$$

其中：

- $\hat{\mu}_a$ 是摇臂 $a$ 的当前估计期望
- $N_a$ 是摇臂 $a$ 被拉过的次数
- $c$ 是探索常数（通常 $c = \sqrt{2}$）
- $\sqrt{\ln t / N_a}$ 是不确定性项

直觉：如果 $N_a$ 很小（很少探索），不确定性大，奖金高，鼓励再试。如果 $N_a$ 很大（已经试够），不确定性小，奖金低，只看估计值。

### 代码实现

```python
class UCB1:
    def __init__(self, n_arms, c=1.0):
        self.n_arms = n_arms
        self.c = c
        self.q = np.zeros(n_arms)
        self.n = np.zeros(n_arms)
        self.t = 0

    def select(self):
        self.t += 1
        # 第一次：每个摇臂都试一遍，避免除零
        for arm in range(self.n_arms):
            if self.n[arm] == 0:
                return arm
        ucb = self.q + self.c * np.sqrt(np.log(self.t) / self.n)
        return np.argmax(ucb)

    def update(self, arm, reward):
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]
```

### UCB 的遗憾界

UCB1 的遗憾严格证明是（Auer et al. 2002）：

$$\mathbb{E}[\text{Regret}(T)] \leq 8 \sum_{a: \mu_a < \mu^*} \frac{\ln T}{\Delta_a} + \left(1 + \frac{\pi^2}{3}\right) \sum_{a: \mu_a < \mu^*} \Delta_a$$

其中 $\Delta_a = \mu^* - \mu_a$ 是次优摇臂与最优的差距。这是 $\Theta(\log T)$ 增长——达到了 Lai-Robins 下界。**UCB 是渐近最优的**。

## Thompson 采样的贝叶斯视角

UCB 用频率派的不确定性（基于 Hoeffding 不等式）来量化"不确定"。Thompson 采样（Thompson 1934, 重新发现于 2011）用贝叶斯视角，思路更优雅。

### 贝叶斯思想

为每个摇臂 $a$ 维护一个**后验分布** $P(\mu_a \mid \text{history})$。每轮：

1. 从每个摇臂的后验采样一个 $\tilde{\mu}_a$
2. 选 $\arg\max_a \tilde{\mu}_a$（这次"假装"采到的估计是真的）
3. 观察奖励，用贝叶斯更新后验

**为什么这能平衡探索-利用？**

- 如果某摇臂的后验分布**很窄**（已经试很多次，估计准），采样的 $\tilde{\mu}_a$ 接近 $\hat{\mu}_a$，倾向于利用
- 如果某摇臂的后验分布**很宽**（试得少，估计不确定），采样的 $\tilde{\mu}_a$ 方差大，有概率采到高值，自动探索

这是"概率匹配"（Probability Matching）的实现：选每个摇臂的概率等于它是最优摇臂的后验概率。

### Bernoulli 奖励的 Beta 共轭先验

最常见的设定：奖励是 0/1（点击 vs 不点击），用 Beta 分布作为 $\mu_a$ 的先验：

$$\mu_a \sim \text{Beta}(\alpha_a, \beta_a)$$

后验更新非常简单：

- 拉到 $a$，得 1：$\alpha_a \leftarrow \alpha_a + 1$
- 拉到 $a$，得 0：$\beta_a \leftarrow \beta_a + 1$

```python
class ThompsonSampling:
    def __init__(self, n_arms, alpha=1, beta=1):
        self.n_arms = n_arms
        self.alpha = np.full(n_arms, alpha, dtype=float)
        self.beta = np.full(n_arms, beta, dtype=float)

    def select(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, reward):
        # reward 假设是 0/1
        if reward > 0.5:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

### 实证表现

Thompson 采样在大量实验中表现**优于或持平 UCB**，且对超参数更鲁棒。工业界的推荐系统、A/B 测试、广告投放广泛使用它。Google 2012 年起在 AdWords 中用 Thompson 采样替换了 ε-贪心，点击率显著提升。

理论遗憾也是 $\Theta(\log T)$，与 UCB 同阶。

## 本节总结

UCB 和 Thompson 采样都达到了 $\Theta(\log T)$ 渐近下界，是工业界推荐的两种 MAB 算法。UCB 用频率派的不确定性（基于 Hoeffding 不等式），Thompson 用贝叶斯后验。两者性能相近，但 Thompson 对超参数更鲁棒，在推荐系统、A/B 测试中应用更广。

下一节 [2.3 遗憾界、PAC 与上下文老虎机](./theory-contextual) 把视角从算法扩展到理论分析——如何证明一个 MAB 算法是"最优"的？以及最重要的扩展——上下文老虎机，它是 RLHF 的形式化桥梁。
