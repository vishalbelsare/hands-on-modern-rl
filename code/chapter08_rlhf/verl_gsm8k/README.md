# 8.7 veRL + GSM8K 适配代码

本目录对应教程 [8.7 动手：用 veRL 在 GSM8K 上跑 PPO 训练](../../../docs/chapter08_rlhf/verl-ppo-gsm8k.md)。

本仓库不复制 veRL 源码。这里仅提供课程用的 GSM8K reward 函数和启动脚本；实际训练入口仍来自外部 veRL 仓库。

## 外部依赖

- veRL 官方仓库：<https://github.com/volcengine/verl>
- veRL PPO 训练入口：`python3 -m verl.trainer.main_ppo`
- veRL GSM8K 数据预处理：`examples/data_preprocess/gsm8k.py`

## 使用方式

先按教程安装 veRL，并准备 GSM8K 数据：

```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
```

然后在 veRL 环境中使用本目录脚本：

```bash
cd /path/to/hands-on-modern-rl/code/chapter08_rlhf/verl_gsm8k
chmod +x run_qwen2_5_0_5b_ppo_single_gpu.sh
./run_qwen2_5_0_5b_ppo_single_gpu.sh
```

如果要切换到进阶 reward：

```bash
./run_qwen2_5_0_5b_ppo_single_gpu.sh \
  custom_reward_function.path="$(pwd)/gsm8k_reward_advanced.py" \
  custom_reward_function.name=compute_score
```

## 文件对应关系

| 文件                                 | 作用                            |
| ------------------------------------ | ------------------------------- |
| `gsm8k_reward.py`                    | 基础 0/1 accuracy reward        |
| `gsm8k_reward_advanced.py`           | accuracy + format 的组合 reward |
| `run_qwen2_5_0_5b_ppo_single_gpu.sh` | 单卡 0.5B PPO 启动脚本          |
| `run_qwen2_5_0_5b_ppo_8gpu.sh`       | 单机 8 卡 PPO 启动脚本          |
