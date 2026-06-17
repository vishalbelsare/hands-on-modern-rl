# C.5 Softmax 与 Cross-Entropy

面试热身题。写 DPO / PPO 之前，面试官可能先让你手写一个数值稳定的 softmax 和交叉熵。

---

## 数值稳定的 Softmax

**核心问题**：把任意实数 logits 映射成概率分布（和为 1），同时避免 $\exp(1000)=\text{inf}$ 这类数值溢出。

**核心变量**：

- `x` / `logits`：模型输出的实数向量
- `x_shifted`：减去 $\max(x)$ 后的平移向量，用于数值稳定
- `exp_x`：平移后的指数
- `axis`：归一化方向，LLM 里通常是最后一维（词表）

### 一句话记忆

> **先减最大值，再 exp、再求和、再相除——防止数字爆炸。**

### 伪代码

```
# 第 1 步：所有数减去最大值（最大的变成 0，避免 exp 溢出）
x_shifted = x - max(x)

# 第 2 步：每个数取指数
exp_x = exp(x_shifted)

# 第 3 步：除以总和，归一化成概率（加起来等于 1）
softmax = exp_x / sum(exp_x)
```

### 记忆方法

为什么不直接 `exp(x) / sum(exp(x))`？因为 `exp(1000)` = inf。减去 max 后最大值变成 `exp(0)=1`，其余都小于 1，不会溢出。

### Python 实现

```python
import numpy as np

def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
```

### PyTorch 实现

```python
import torch
import torch.nn.functional as F

# 工程中直接用
probs = F.softmax(logits, dim=-1)

# 手写版（面试用）
def manual_softmax(x, dim=-1):
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x_shifted)
    return e_x / e_x.sum(dim=dim, keepdim=True)
```

---

## Log-Sum-Exp 技巧

**核心问题**：LLM 训练需要的是 log 概率，不是概率本身。若先 softmax 再 log，先除再 log 会损失精度，且小概率位置会下溢为 0 后 log 变 -inf。log-sum-exp 公式把减 max 和 log 合并成一步，得到数值稳定的 log 概率。

**核心变量**：

- `x_shifted`：减去 $\max(x)$ 后的向量
- `log_sum_exp`：$\log\sum_i \exp(x_i - \max)$，logistic normalizer 的对数
- 输出：`log_softmax(x) = x_shifted - log_sum_exp`

### 一句话记忆

> **算 log(softmax) 别分两步——直接 log-sum-exp，先把 max 提出来。**

面试追问：`log(softmax(x))` 怎么算才不会溢出？答：不要先 softmax 再 log，直接用 log-softmax。

### Python 实现

```python
def log_softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    return x_shifted - np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))
```

### PyTorch 实现

```python
# 内置，数值稳定
log_probs = F.log_softmax(logits, dim=-1)

# 手写
def manual_log_softmax(x, dim=-1):
    max_val = x.max(dim=dim, keepdim=True).values
    return x - max_val - torch.log(torch.sum(torch.exp(x - max_val), dim=dim, keepdim=True))
```

---

## Cross-Entropy Loss

**核心问题**：分类/SFT 任务需要一个标量损失来衡量"预测分布"和"真实标签"的差距。Cross-Entropy 把它压缩成"目标位置 log 概率的负数"——预测越准，loss 越小。

**核心变量**：

- `logits`：模型输出，形状 `[N, C]`，N 是样本数，C 是类别数
- `targets`：真实类别索引，形状 `[N]`
- `ignore_index`：跳过的位置（如 padding/prompt），默认 `-100`
- `log_probs`：log_softmax 后的对数概率，用于取出目标位置

### 一句话记忆

> **预测对了多少看概率：取目标位置的概率，加负号取 log。**

### 伪代码

```
# 第 1 步：logits 过 log_softmax，得到每个类的 log 概率
log_probs = log_softmax(logits)

# 第 2 步：取出"正确答案"那个位置的 log 概率，取负取平均
loss = -log_probs[target].mean()
```

### 记忆方法

交叉熵 = 一个预测对了多少的度量。预测越准，$p_{y}$ 越大，$-\log p_{y}$ 越小。loss 小 = 好。

### Python 实现

```python
def cross_entropy(logits, targets, ignore_index=-100):
    """
    logits: [N, C]
    targets: [N]  (整数类别)
    """
    log_probs = log_softmax(logits, axis=-1)
    total, count = 0.0, 0
    for i in range(len(targets)):
        if targets[i] == ignore_index:
            continue
        total += -log_probs[i, targets[i]]
        count += 1
    return total / max(count, 1)
```

### PyTorch 实现

```python
def manual_cross_entropy(logits, targets, ignore_index=-100):
    """
    logits: [B, C]
    targets: [B]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # gather 取出 target 位置的 log 概率
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    # mask 掉 ignore_index
    mask = targets != ignore_index
    return -target_log_probs[mask].mean()
```

---

## 易错点

| 易错                       | 说明                                                                 |
| -------------------------- | -------------------------------------------------------------------- |
| softmax 忘了减 max         | 面试手写第一步就扣分                                                 |
| 先 softmax 再 log          | 数值不稳定，用 `log_softmax` 一步到位                                |
| Cross-Entropy 从 logits 算 | 不要先 softmax 再 log 再 CE，直接 `F.cross_entropy(logits, targets)` |
| `ignore_index`             | 面试追问 SFT loss 时会问，padding token 怎么处理                     |
| temperature                | `logits / temperature` 再 softmax，T 越大分布越平                    |
