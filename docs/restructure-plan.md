# 全书重构计划

> 生成日期：2026-04-19
> 核心变动：Ch3 重写为通识导览，Ch5 拆分为策略梯度 + Actor-Critic 两章，后续顺延

---

## 一、设计原则

1. **Ch3 是地图，不是深潜。** 读者看完知道 RL 全景，不深入任何一条路线的计算细节
2. **Q 和 J 是并列的两条路线。** V 是基础工具，不是第三条路线
3. **V 的局限驱动 Q 和 J 的出现。** "V 不告诉你选哪个动作" → 这个问题催生两条路线
4. **V 在全书中渐进出场：** Ch3 概念 → Ch6 Critic 深入 → Ch7 GAE 应用
5. **每章末尾的局限自然引出下一章**，不跳跃

---

## 二、叙事主线（贯穿 Ch3-7）

```
Ch3: 看到全景地图
  "RL 要解决什么？MDP 是什么？V 是什么？怎么算 V？
   V 有局限 → 两条路线：Q（打分再选）和 J（直接优化）"

Ch4: 走路线一（Q）
  "Q-Learning 表格 → DQN 神经网络 → 但只能处理离散动作"

Ch5: 走路线二（J）
  "REINFORCE 能工作 → 但方差太大 → 需要更好的评价标准"

Ch6: 两条路线融合（Actor-Critic）
  "V 回来当评委（Critic）→ Actor + Critic 架构 → 但训练不稳定"

Ch7: 工程化落地（PPO）
  "裁剪 + GAE → 稳定的 Actor-Critic → 连接到 LLM 对齐"
```

贯穿暗线：
```
MC → Ch3 速览 → Ch5 REINFORCE（用 G_t，完整轨迹，方差大）
TD → Ch3 速览 → Ch4 Q-Learning（走一步更新）
                → Ch6 Critic 训练（TD Error）
                → Ch7 GAE（多个 TD Error 加权）
DP → Ch3 速览 → Ch6 作为理论基准（知道模型时的最优）
```

---

## 三、新章节结构与内容映射

### 第 1-2 章（不变）

| 章 | 标题 | 文件 | 状态 |
|---|---|---|---|
| 1 | RL 初印象（CartPole） | `chapter01_cartpole/` | 不变 |
| 2 | 现代 RL 初体验（DPO） | `chapter02_dpo/` | 不变 |

---

### 第 3 章：RL 通识 — 理论地图与全局视野（重写）

> 定位：看完知道 RL 地图，不深入任何路线的细节。
> V 是基础工具，Q 和 J 是从 V 的局限中长出的两条路。

| 新小节 | 标题 | 内容 | 来源 | 状态 |
|---|---|---|---|---|
| 3.0 | 章节导览 | 核心问题链 + 导航表 | 重写 intro.md | **新建** |
| 3.1 | 两台老虎机 | 探索vs利用、期望回报、Agent-Env循环 | 现有 bandit.md | **改写语气** |
| 3.2 | MDP + G_t + 策略 π | 五元组、折扣回报、策略（确定性/随机性） | 现有 formalism.md（拆分） | **改写** |
| 3.3 | V(s) 与贝尔曼方程 | V 定义、贝尔曼方程、DP/MC/TD 速览、TD Error | 现有 bellman-equation.md + classic-methods.md（前半） | **合并改写** |
| 3.4 | 路线一：Q(s,a) | Q 定义、V/Q 关系、argmax Q = 最优动作、迷宫牌子类比 | 现有 formalism.md（Q 部分） | **改写** |
| 3.5 | 路线二：J(θ) | J(θ) 定义、θ*=argmax J、反复走迷宫类比 | **新写** | **新建** |
| 3.6 | 全景地图 | 两条路线对比、探索vs利用回扣、Actor-Critic 伏笔、表→网络预告 | 现有 classic-methods.md（后半） | **改写** |

#### 3.1 两台老虎机（改写）

**保留：** 三种策略对比、期望回报、Agent-Environment 交互循环、代码

**改写：**
- 语气：去掉"恭喜你"、"花10秒钟想一下"，改为教材陈述性语气
- 结尾更明确地引出 MDP："老虎机是单状态的，真实 RL 是多步序列决策，需要更完整的框架"

**来源：** `bandit.md` 改写

---

#### 3.2 MDP 五元组 + G_t + 策略 π（改写）

**内容：**
- MDP 五元组 (S, A, P, R, γ) — 老虎机/CartPole/LLM 三例对照
- 折扣累积回报 G_t — 公式 + CartPole 具体数字例子
- 策略 π — 正式定义，确定性 vs 随机性，随机性策略天然兼顾探索

**不包含：** V(s)、Q(s,a)（移到 3.3 和 3.4）

**来源：** `formalism.md` 前半部分（MDP + G_t）改写，加入策略 π 的正式定义

---

#### 3.3 V(s) 与贝尔曼方程（合并改写）

**叙事逻辑：**
```
"这个局面值多少分？" → V(s) 定义
  → V 怎么算？→ 贝尔曼方程（只看一步）
    → 三种算 V 的方法：DP / MC / TD（每个 2-3 段速览）
      → TD Error：预测与现实的落差
        → V 的局限：告诉你局面好不好，但不告诉你选哪个动作
```

**内容：**
- V(s) 定义 + 直觉（棋局评估器类比）
- 贝尔曼方程：V(s) = R + γΣP·V(s')（宝藏地图手画保持，改写语气）
- DP：知道模型 → 直接迭代贝尔曼方程（精确但不现实）
- MC：不知道模型 → 跑完一整局（无偏但方差大）
- TD：走一步就更新（有偏但实用）— 一句话带过 REINFORCE 是 MC 策略版
- TD Error：δ = r + γV(s') - V(s)
- 关键洞察：DP→MC→TD 每代解决前一代的局限

**不包含：** 贝尔曼最优方程深入推导、Q 函数、GridWorld 实验（留给后续章节）

**来源：**
- `bellman-equation.md`（V 定义 + 贝尔曼 + TD Error）
- `classic-methods.md` 前半（DP/MC/TD 速览）

**合并为一个文件**，删掉 `formalism.md` 中 V/Q 部分的重复内容

---

#### 3.4 路线一：Q(s,a)（改写）

**叙事逻辑：**
```
V 告诉你局面好不好，但不告诉你选哪个动作
  → Q(s,a) 加了一个动作条件 → "做这个动作值多少分"
    → 比较所有动作的 Q → argmax → 最优策略
```

**内容：**
- Q(s,a) 定义：固定第一步动作后的期望回报
- V 和 Q 的关系：V = Σπ(a|s)·Q(s,a)
- 最优策略：π*(s) = argmax_a Q(s,a)
- 迷宫牌子类比：每条路有牌子写分数，选最高的
- 代表算法预告：Q-Learning → DQN（Ch4 展开）

**不包含：** 贝尔曼 for Q（留给 Ch4）、Q-Learning 细节

**来源：** `formalism.md` 中 Q 部分 + V/Q 关系部分，改写为独立小节

---

#### 3.5 路线二：J(θ)（新建）

**叙事逻辑：**
```
Q 的做法是先打分再选——但如果动作空间是连续的呢？
  → 换一个思路：跳过打分，直接学"该做什么"
    → 策略 π_θ 由参数 θ 定义
      → J(θ) = E[G_t]：这个策略平均拿多少分
        → θ* = argmax J(θ)
```

**内容：**
- 动机：连续动作空间 / 概率分布 → argmax 不好使
- J(θ) 定义：E_{π_θ}[G_t]
- θ* = argmax J(θ)
- 迷宫类比：不挂牌子，反复走迷宫加强/削弱选择信心
- 代表算法预告：REINFORCE → PPO（Ch5 展开）

**不包含：** 策略梯度定理、REINFORCE 代码（留给 Ch5）

**来源：** 全新编写

---

#### 3.6 全景地图（改写）

**内容：**
- 两条路线对比表：
  - 路线一：打分准、样本效率高、但不擅长探索、只能离散动作
  - 路线二：擅长探索、支持连续动作、但打分不准、方差大
- 探索 vs 利用回扣：路线一的短板（ε-贪婪是人工补丁）和路线二的短板（高方差）
- 暗示：两条路拼起来？→ Actor-Critic（Ch6 伏笔）
- 表格 → 神经网络：Deep RL 一句话预告
- V 的后续角色预告：Critic = V(s) 的网络实现
- 章节导航：后续各章对应地图上的哪条路

**来源：** `classic-methods.md` 后半（全景部分）改写

---

### 第 3 章文件变更汇总

| 操作 | 文件 | 说明 |
|---|---|---|
| **改写** | `intro.md` | 新章节导览 |
| **改写** | `bandit.md` | 语气修正 |
| **改写** | `formalism.md` → 重命名为 `mdp.md` | 只保留 MDP + G_t + π，去掉 V/Q |
| **合并改写** | `bellman-equation.md` + `classic-methods.md` 前半 → `value-v.md` | V + 贝尔曼 + DP/MC/TD 速览 |
| **新建** | `value-q.md` | Q(s,a) + argmax Q = 路线一 |
| **新建** | `policy-objective.md` | J(θ) + argmax J = 路线二 |
| **改写** | `classic-methods.md` 后半 → `panorama.md` | 全景地图 + 对比 + 伏笔 |
| **删除** | `classic-methods.md` 原文件 | 拆散到 value-v.md 和 panorama.md |

---

### 第 4 章：Q-Learning 到 DQN（调整）

> 定位：路线一的深入展开。基本保持现有内容，微调衔接。

| 小节 | 内容 | 来源 | 状态 |
|---|---|---|---|
| 4.0 | 章节导览 | 现有 intro.md | **改写衔接** |
| 4.1 | Q-Learning → GridWorld 实验 | 现有 from-q-to-dqn.md | **调整**：Ch3 已讲过 Q 概念，这里直接切入算法 |
| 4.2 | DQN 三大组件 | 现有 dqn-components.md | 不变 |
| 4.3 | 动手：CartPole DQN | 现有 cartpole-dqn.md | 不变 |
| 4.4 | 训练分析 | 现有 training-analysis.md | 不变 |
| 4.5 | DQN 家族 | 现有 dqn-family.md | 不变 |
| 4.6 | Atari DQN | 现有 atari-dqn.md | 不变 |
| 4.7 | Retro Pokemon | 现有 retro-pokemon.md | 不变 |
| 4.8 | VizDoom | 现有 vizdoom-dqn.md | 不变 |

**变更：**
- intro.md 改写：从"Ch3 路线一的深入"切入，不再重复 Q-Learning 基础概念
- from-q-to-dqn.md 调整：GridWorld 实验保留，但去掉 Q(s,a) 定义的重复介绍
- 其他文件不动

---

### 第 5 章：策略梯度 — 直接优化策略（拆分）

> 定位：路线二的深入展开。从原 Ch5 拆出，去掉 Actor-Critic 部分。

| 小节 | 内容 | 来源 | 状态 |
|---|---|---|---|
| 5.0 | 章节导览 | 重写 intro.md | **改写** |
| 5.1 | 摇骰子赌博机 | 现有 dice-game.md | 不变 |
| 5.2 | 策略梯度定理与 REINFORCE | 现有 policy-gradient.md | **微调**：强调 REINFORCE = MC 策略版，回扣 Ch3 |
| 5.3 | 基线实验 | 现有 baseline-experiment.md | **微调**：结尾引出 V(s) 作为最佳基线 → 暗示 Critic |
| 5.4 | 章节小结 | **新写** | **新建** |

**变更：**
- intro.md 重写：从"Ch3 路线二的深入"切入
- policy-gradient.md 微调：加入与 Ch3 MC 概念的回扣
- baseline-experiment.md 微调：结尾明确引出"V(s) 是最佳基线 → 需要一个网络估计 V → 这就是 Critic"
- 去掉 actor-critic.md 和 alphago.md（移到新 Ch6）

---

### 第 6 章：Actor-Critic — 两条路线的融合（新章节）

> 定位：V(s) 在这里获得深入处理。Critic 的训练 = V 的估计 = DP/MC/TD 的实战应用。

| 小节 | 内容 | 来源 | 状态 |
|---|---|---|---|
| 6.0 | 章节导览 | **新写** | **新建** |
| 6.1 | 优势函数与 V(s) 的角色 | **新写** | **新建** |
| 6.2 | 怎么训练 Critic | **新写**（展开 Ch3 的 DP/MC/TD） | **新建** |
| 6.3 | Actor-Critic 架构 | 现有 actor-critic.md | **改写** |
| 6.4 | 动手：Actor-Critic 实验 | **新写**（CartPole 对比实验） | **新建** |
| 6.5 | 动手：AlphaGo 简易复现 | 现有 alphago.md | **调整衔接** |
| 6.6 | 章节小结 | **新写** | **新建** |

#### 6.1 优势函数与 V(s) 的角色（新建）

**内容：**
- 从 Ch5 结尾承接：REINFORCE 方差太大，需要更好的评价标准
- 优势函数 A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s) = TD Error
- 为什么 A 比 G_t 好：减掉了"本来就能拿到的分"，只保留"因为做了这个动作多拿的分"
- V(s) 作为 Critic 的直觉：评委给动作打分

#### 6.2 怎么训练 Critic（新建）

**内容：**
- 承接 Ch3：Ch3 速览了 DP/MC/TD，现在展开讲"怎么用这些方法训练 V 网络"
- DP：知道模型时 → 理论最优（价值迭代/策略迭代概念）
- MC：用完整轨迹估计 V → 无偏但方差大
- TD：用 r + γV(s') 更新 V → 有偏但方差小 → 实际首选
- 神经网络作为 V 的函数逼近器：Critic 网络结构
- TD Error 作为 Critic 的训练损失

**关键设计：** DP/MC/TD 在这里展开有明确动机——"Critic 怎么训练"。

#### 6.3 Actor-Critic 架构（改写）

**来源：** 现有 `actor-critic.md` 改写
**调整：** 去掉与策略梯度的重复介绍，直接聚焦 Actor + Critic 的协作机制

#### 6.4 动手：Actor-Critic 实验（新建）

**内容：**
- CartPole 上对比：纯 REINFORCE vs Actor-Critic
- 亲眼看到 Critic 降低了方差
- 训练曲线对比图

---

### 第 7 章：PPO — 稳定训练的艺术（原 Ch6 不变）

> 定位：Actor-Critic 的工程化落地。

| 小节 | 内容 | 来源 | 状态 |
|---|---|---|---|
| 7.0 | 章节导览 | 现有 intro.md | **微调衔接** |
| 7.1 | PPO 训练 LunarLander | 现有 ppo-lunar-lander.md | 不变 |
| 7.2 | PPO 数学推导 | 现有 ppo-math.md | 不变 |
| 7.3 | 信任域与裁剪 | 现有 trust-region-clipping.md | 不变 |
| 7.4 | GAE、奖励模型与 LLM 对齐 | 现有 gae-reward-model.md | 不变 |

**文件搬迁：** `chapter06_ppo/` → `chapter07_ppo/`（目录重命名）

---

### 第 8-14 章（顺延 +1）

| 新编号 | 标题 | 原目录 | 新目录 | 状态 |
|---|---|---|---|---|
| 8 | DPO / KTO / SimPO | `chapter07_alignment/` | `chapter08_alignment/` | 重命名 |
| 9 | GRPO / DAPO / RLVR | `chapter08_grpo_rlvr/` | `chapter09_grpo_rlvr/` | 重命名 |
| 10 | SAC / TD3 连续控制 | `chapter09_continuous_control/` | `chapter10_continuous_control/` | 重命名 |
| 11 | RLHF 完整流水线 | `chapter10_rlhf/` | `chapter11_rlhf/` | 重命名 |
| 12 | VLM 强化学习 | `chapter11_vlm_rl/` | `chapter12_vlm_rl/` | 重命名 |
| 13 | Agentic RL | `chapter12_agentic_rl/` | `chapter13_agentic_rl/` | 重命名 |
| 14 | 未来趋势 | `chapter13_future_trends/` | `chapter14_future_trends/` | 重命名 |

**Ch8-14 内容不变**，只做目录重命名 + 内部链接更新。

---

## 四、执行顺序

### Phase 1：Ch3 重写（最高优先级）

1. **新建 `value-q.md`** — Q(s,a) + argmax Q = 路线一
2. **新建 `policy-objective.md`** — J(θ) + argmax J = 路线二
3. **改写 `intro.md`** — 新章节导览
4. **改写 `bandit.md`** — 语气修正
5. **改写 `formalism.md`** → 重命名为 `mdp.md` — 只保留 MDP + G_t + π
6. **合并改写** `bellman-equation.md` + `classic-methods.md` 前半 → `value-v.md`
7. **改写** `classic-methods.md` 后半 → `panorama.md`
8. **删除** `classic-methods.md` 原文件
9. **更新** `config.mjs` 中 Ch3 的 sidebar

### Phase 2：Ch4 微调

10. **改写** Ch4 `intro.md` — 衔接调整
11. **调整** `from-q-to-dqn.md` — 去掉重复的 Q 概念介绍

### Phase 3：Ch5 拆分

12. **改写** Ch5 `intro.md` — 去掉 Actor-Critic 相关内容
13. **微调** `policy-gradient.md` — 加入与 Ch3 MC 的回扣
14. **微调** `baseline-experiment.md` — 结尾引出 Critic
15. **新建** 章节小结

### Phase 4：Ch6 新建（Actor-Critic）

16. **新建** `chapter06_actor_critic/` 目录
17. **新建** `intro.md` — 章节导览
18. **新建** `advantage-critic.md` — 优势函数 + V 角色 + Critic 训练方法
19. **搬入改写** `actor-critic.md` — 从原 Ch5 搬来，改写衔接
20. **新建** `actor-critic-experiment.md` — CartPole 对比实验
21. **搬入调整** `alphago.md` — 从原 Ch5 搬来
22. **新建** 章节小结

### Phase 5：Ch7+ 重命名

23. `chapter06_ppo/` → `chapter07_ppo/`（改写 intro 衔接）
24. `chapter07_alignment/` → `chapter08_alignment/`
25. `chapter08_grpo_rlvr/` → `chapter09_grpo_rlvr/`
26. `chapter09_continuous_control/` → `chapter10_continuous_control/`
27. `chapter10_rlhf/` → `chapter11_rlhf/`
28. `chapter11_vlm_rl/` → `chapter12_vlm_rl/`
29. `chapter12_agentic_rl/` → `chapter13_agentic_rl/`
30. `chapter13_future_trends/` → `chapter14_future_trends/`
31. **更新** `config.mjs` 全局 sidebar 配置
32. **更新** 所有内部链接（跨章节引用）

---

## 五、V(s) 在全书中出现的节奏

| 位置 | V 的角色 | 深度 |
|---|---|---|
| Ch3.3 | "什么是 V" — 概念 + 贝尔曼 + DP/MC/TD 速览 | 浅（知道是什么） |
| Ch5.3 | "V 可以当基线" — 一句话提及 | 提及（知道能用） |
| Ch6.1-6.2 | "V 是 Critic" — 优势函数 + 训练 Critic = 训练 V | 深（知道怎么训） |
| Ch7.4 | "V 用于 GAE" — 多个 TD Error 的加权和 | 应用（知道怎么用） |

---

## 六、待讨论 / 后续决策

1. **Ch3.5（J(θ)）的篇幅**：作为通识章节的一节，J(θ) 只需要概念定义 + 直觉类比，不需要推导。篇幅控制在 1-2 页。
2. **Ch4 是否需要精简**：当前 Ch4 有 8 个文件（含 Retro Pokemon、VizDoom），是否全部保留？
3. **Ch6.4 实验设计**：CartPole 上 REINFORCE vs Actor-Critic 的对比实验，用什么框架？纯 PyTorch 还是 SB3？
4. **Phase 5 重命名的影响**：目录重命名 + 内部链接更新工作量较大，是否可以脚本自动化？
