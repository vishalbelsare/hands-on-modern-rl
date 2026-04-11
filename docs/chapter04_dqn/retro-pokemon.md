# 4.8 动手：用 stable-retro 玩宝可梦

CartPole 的状态是 4 个数字，Atari 的状态是 84×84 像素，ViZDoom 的状态是 3D 第一人称画面。现在，我们要面对一种全新的挑战——**一个需要长期规划、信息不完全、决策链长达数千步的游戏：宝可梦**。

## 为什么宝可梦比 Atari 难得多？

Atari 的 Pong 可以在几百个 episode 内学会——因为每一局只有几十步，奖励信号（得分）即时且明确。但宝可梦完全不同：

|            | Pong（Atari）        | 宝可梦红（Game Boy）              |
| ---------- | -------------------- | --------------------------------- |
| 单局时长   | ~20 秒               | 数小时（通关）                    |
| 决策步数   | ~100 步              | 数万步                            |
| 奖励信号   | 每帧都有（得分变化） | 极度稀疏（打赢道馆馆主才有）      |
| 状态空间   | 球的位置 + 球拍位置  | HP、等级、技能、背包、地图位置... |
| 部分可观测 | 否                   | 是（对方宝可梦的技能未知）        |
| 长期规划   | 不需要               | 必须（练级 → 进化 → 挑战道馆）    |

宝可梦是一个**延迟奖励、部分可观测、超长决策链**的环境。它是检验 DQN 系列算法极限的绝佳试金石。

## stable-retro：把复古游戏变成 RL 环境

[stable-retro](https://github.com/Farama-Foundation/stable-retro) 由 Farama 基金会维护（和 Gymnasium、PettingZoo 同一个团队），是 OpenAI gym-retro 的继任者。它通过模拟器把经典的复古游戏（Game Boy、世嘉、SNES 等）包装成 Gymnasium 兼容的 RL 环境。

支持的平台包括：

- Nintendo Game Boy / Game Boy Color
- Sega Master System / Genesis / Game Gear
- Super Nintendo (SNES)
- Atari 2600 / 7800
- NEC TurboGrafx-16

安装方式：

```bash
pip install stable-retro
```

stable-retro 内置了一些游戏（主要是世嘉 Genesis 的 Sonic 系列），其他游戏（如宝可梦）需要你自己提供 ROM 文件并集成：

```python
import retro

# 查看内置游戏
print(retro.list_games())

# 创建 Sonic 环境
env = retro.make("SonicTheHedgehog-Genesis", state="GreenHillZone.Act1")
print(f"观测空间: {env.observation_space}")  # Box(0, 255, (224, 320, 3), uint8)
print(f"动作空间: {env.action_space}")        # MultiBinary(12)
```

注意动作空间是 `MultiBinary(12)`——12 个按钮的独立开关（上/下/左/右/A/B/Start/Select 等），和 Atari 的 `Discrete` 不同。

## 加载宝可梦红

加载宝可梦需要 ROM 文件和自定义集成。以下是完整的集成流程：

### 第一步：准备 ROM 文件

你需要合法获取宝可梦红的 Game Boy ROM 文件（`.gb` 格式）。将 ROM 放入 stable-retro 的数据目录：

```bash
python -c "import retro; print(retro.data.Integrations.CUSTOM_ONLY_PATH)"
# 输出类似: ~/.local/lib/python3.10/site-packages/retro/data/custom
```

```bash
mkdir -p ~/.local/lib/python3.10/site-packages/retro/data/custom/PokemonRed
cp PokemonRed.gb ~/.local/lib/python3.10/site-packages/retro/data/custom/PokemonRed/rom.gb
```

### 第二步：创建集成文件

集成文件告诉 stable-retro 如何与游戏交互——从哪些内存地址读取状态、如何定义奖励等：

```bash
mkdir -p ~/.local/lib/python3.10/site-packages/retro/data/custom/PokemonRed
```

创建 `data.json`，定义内存地址映射：

```json
{
  "info": {
    "player_x": { "address": 0xd362, "type": "|u1" },
    "player_y": { "address": 0xd361, "type": "|u1" },
    "player_map": { "address": 0xd35e, "type": "|u1" },
    "pokemon_hp": { "address": 0xd16c, "type": ">u2" },
    "pokemon_level": { "address": 0xd18c, "type": "|u1" },
    "badges": { "address": 0xd356, "type": "|u1" },
    "money": { "address": 0xd347, "type": ">u3" }
  },
  "reward": {
    "entities": [
      { "type": "memory", "address": 0xd356, "id": "badges" },
      { "type": "memory", "address": 0xd18c, "id": "level" }
    ]
  }
}
```

> 这些内存地址来自 Game Boy 模拟器社区对宝可梦红的逆向工程。[参考来源](https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map)

### 第三步：定义动作空间

宝可梦红的操作只需要几个按键，不需要所有 8 个 Game Boy 按钮。创建一个简化的动作映射：

```python
import retro
import numpy as np

# 自定义动作组合：只保留有效操作
SIMPLE_MOVES = [
    [0, 0, 0, 0, 0, 0, 0, 0],  # 不动
    [1, 0, 0, 0, 0, 0, 0, 0],  # 上 (B)
    [0, 1, 0, 0, 0, 0, 0, 0],  # 下
    [0, 0, 1, 0, 0, 0, 0, 0],  # 左
    [0, 0, 0, 1, 0, 0, 0, 0],  # 右
    [0, 0, 0, 0, 1, 0, 0, 0],  # A（确认/对话/攻击）
    [0, 0, 0, 0, 0, 1, 0, 0],  # B（取消/返回）
    [0, 0, 0, 0, 0, 0, 1, 0],  # Start（菜单）
]

env = retro.make(
    game="PokemonRed",
    obs_type=retro.Observations.IMAGE,  # 屏幕像素
    controls=SIMPLE_MOVES,
)
```

## 奖励设计：稀疏奖励的核心难题

宝可梦最大的 RL 挑战是**奖励极其稀疏**。整个游戏可能要玩几个小时才能获得一个徽章（Badge）。如果直接把"获得徽章数"作为奖励，DQN 根本学不到东西——因为它在数万步的随机操作中几乎不可能碰巧拿到徽章。

常见的奖励设计策略：

```python
class PokemonRewardWrapper(retro.RetroReward):
    """自定义奖励函数：组合多个辅助奖励"""
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.prev_level = 0
        self.prev_x = 0
        self.prev_y = 0
        self.prev_badges = 0
        self.prev_map = 0

    def reward(self, info):
        reward = 0

        # 主要奖励：获得徽章（+100）
        badges = info.get('badges', 0)
        if badges > self.prev_badges:
            reward += 100 * (badges - self.prev_badges)
        self.prev_badges = badges

        # 辅助奖励 1：等级提升（+1）
        level = info.get('pokemon_level', 0)
        if level > self.prev_level:
            reward += (level - self.prev_level)
        self.prev_level = level

        # 辅助奖励 2：探索新地图（+0.5）
        map_id = info.get('player_map', 0)
        if map_id != self.prev_map:
            reward += 0.5
        self.prev_map = map_id

        # 惩罚：死亡（HP 归零）（-10）
        hp = info.get('pokemon_hp', 0)
        if hp == 0:
            reward -= 10

        return reward
```

这种**分层奖励设计**（主奖励 + 辅助奖励）是解决稀疏奖励问题的标准手法。

## 训练框架

由于宝可梦的特殊性（超长 episode、稀疏奖励），标准的 DQN 往往不够用。实践中常用的组合：

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

env = retro.make("PokemonRed", controls=SIMPLE_MOVES)
env = PokemonRewardWrapper(env)

# PPO 通常比 DQN 更适合这类长序列任务
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=32,
    n_epochs=4,
    clip_range=0.1,
    verbose=1,
)

checkpoint = CheckpointCallback(
    save_freq=100_000,
    save_path="./models/pokemon_red",
)

model.learn(total_timesteps=10_000_000, callback=checkpoint)
```

::: warning
训练宝可梦 RL 是一个**长期项目**。即使有辅助奖励，从随机操作到学会移动和战斗，通常需要数百万步。这与 Pong 几百个 episode 就能收敛完全不同——它更像是一个研究课题而非课堂练习。
:::

## 从像素 RL 的视角回顾

让我们回顾一下像素 RL 的三个环境：

```
CartPole    → 4 维向量，MLP，几分钟训练
Atari Pong  → 84×84 像素，CNN，几小时训练
宝可梦红    → 160×144 像素，CNN + 稀疏奖励，数天训练
```

**同样的算法框架，环境的复杂度决定了问题的难度**。这正是强化学习与其他机器学习方法的核心区别：你的算法可能很优雅，但环境决定了一切。

在接下来的章节中，我们将离开像素世界，进入另一个重要方向——直接优化策略的策略梯度方法。
