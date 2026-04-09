# 多步滞后训练（Multi-Step Off-Policy）使用说明

## 📋 概述

本文档说明如何使用**方案2：降低权重同步频率**来实现多步滞后训练。

### 修改内容

通过降低 rollout 模型的权重同步频率，使 rollout 模型使用"过期"的权重生成数据，从而实现多步 off-policy 训练。

---

## 🎯 核心原理

### 标准 one-step-off（sync_frequency=1）
```
Step 1: 同步 v1 → rollout, 生成 Batch A, 训练 → v2
Step 2: 同步 v2 → rollout, 生成 Batch B, 训练 → v3
Step 3: 同步 v3 → rollout, 生成 Batch C, 训练 → v4
```

### 多步滞后（sync_frequency=3）
```
Step 1: 同步 v1 → rollout, 生成 Batch A, 训练 → v2  ✅ 同步
Step 2: 跳过同步,          生成 Batch B (用v1), 训练 → v3  ⏭️ 跳过
Step 3: 跳过同步,          生成 Batch C (用v1), 训练 → v4  ⏭️ 跳过
Step 4: 同步 v4 → rollout, 生成 Batch D, 训练 → v5  ✅ 同步
```

**关键点**：
- Rollout 模型在 Step 2-3 使用 v1 的权重（滞后 1-3 步）
- 训练模型一直是最新的（v2, v3, v4）
- 减少了权重同步开销，可能提升训练速度

---

## 🔧 修改的文件

### 1. `/workspace/qingnan/verl/recipe/one_step_off_policy/ray_trainer.py`

**修改位置 1**: 第 154-157 行（`__init__` 方法）
```python
# 添加同步频率控制（方案2：降低同步频率）
self.sync_frequency = config.actor_rollout_ref.rollout.get("sync_frequency", 1)
self.sync_counter = 0
print(f"[Multi-Step Off-Policy] Sync frequency set to: {self.sync_frequency} (sync every {self.sync_frequency} step(s))")
```

**修改位置 2**: 第 349-353 行（`_async_gen_next_batch` 方法）
```python
# sync weights from actor to rollout (with frequency control)
# 方案2：降低同步频率 - 只在特定步数同步权重
if self.sync_counter % self.sync_frequency == 0:
    self.sync_rollout_weights()
self.sync_counter += 1
```

### 2. `/workspace/qingnan/verl/recipe/one_step_off_policy/config/one_step_off_ppo_trainer.yaml`

**修改位置**: 第 16-23 行
```yaml
# 方案2：多步滞后配置 - 降低权重同步频率
actor_rollout_ref:
  rollout:
    # 权重同步频率（默认1表示每步都同步）
    # sync_frequency: 2  表示每2步同步一次权重（rollout模型会滞后1-2步）
    # sync_frequency: 3  表示每3步同步一次权重（rollout模型会滞后1-3步）
    # sync_frequency: 5  表示每5步同步一次权重（rollout模型会滞后1-5步）
    sync_frequency: 1
```

---

## 📖 使用方法

### 方法 1：修改配置文件（推荐）

编辑 `/workspace/qingnan/verl/recipe/one_step_off_policy/config/one_step_off_ppo_trainer.yaml`：

```yaml
actor_rollout_ref:
  rollout:
    sync_frequency: 3  # 改为 3，表示每 3 步同步一次
```

然后正常启动训练：

```bash
cd /workspace/qingnan/verl

# FSDP2 示例
bash recipe/one_step_off_policy/dapo_7b_math_fsdp2_4_12.sh

# 或 Megatron 示例
bash recipe/one_step_off_policy/dapo_7b_math_megatron_4_12.sh
```

### 方法 2：命令行覆盖参数

不修改配置文件，直接在启动脚本中覆盖：

```bash
python3 -m recipe.one_step_off_policy.main_ppo \
    --config-path=config \
    --config-name='one_step_off_ppo_trainer.yaml' \
    actor_rollout_ref.rollout.sync_frequency=3  # 命令行参数
```

或修改现有启动脚本，添加这一行：

```bash
# 在 dapo_7b_math_fsdp2_4_12.sh 中
python3 -m recipe.one_step_off_policy.main_ppo \
    ... \
    actor_rollout_ref.rollout.sync_frequency=3
```

---

## 🧪 实验建议

### 保守方案（推荐首次尝试）

```yaml
sync_frequency: 2  # 每 2 步同步一次
```

**预期效果**：
- ✅ 训练稳定性：高（接近标准模式）
- ✅ 加速效果：10-15%
- ⚠️ Off-policy 程度：轻度

**监控指标**：
- `training/rollout_probs_diff_max`：应该 < 0.1
- `training/rollout_probs_diff_mean`：应该 < 0.05
- PPO clip ratio：应该 < 0.3

### 平衡方案

```yaml
sync_frequency: 3  # 每 3 步同步一次
```

**预期效果**：
- ✅ 训练稳定性：中等
- ✅ 加速效果：20-30%
- ⚠️ Off-policy 程度：中度

**监控指标**：
- `training/rollout_probs_diff_max`：应该 < 0.2
- `training/rollout_probs_diff_mean`：应该 < 0.1
- PPO clip ratio：应该 < 0.5

### 激进方案（需谨慎）

```yaml
sync_frequency: 5  # 每 5 步同步一次
```

**预期效果**：
- ⚠️ 训练稳定性：低（可能不稳定）
- ✅ 加速效果：30-50%
- ⚠️ Off-policy 程度：重度

**监控指标**：
- `training/rollout_probs_diff_max`：可能 > 0.3
- 训练曲线可能波动较大
- 如果训练发散，请降低 sync_frequency

---

## 📊 监控训练状态

### 关键日志输出

启动时会看到：
```
[Multi-Step Off-Policy] Sync frequency set to: 3 (sync every 3 step(s))
```

### 推荐监控的指标

在训练日志或 wandb/tensorboard 中关注：

1. **权重差异指标**（重要！）
   - `training/rollout_probs_diff_max`：rollout 和 actor 模型的最大概率差异
   - `training/rollout_probs_diff_mean`：平均概率差异
   - `training/rollout_probs_diff_std`：标准差

2. **PPO 训练指标**
   - `actor/ppo_clip_ratio`：裁剪率（过高说明 off-policy 太严重）
   - `actor/approx_kl`：近似 KL 散度（监控策略变化）
   - `actor/entropy`：熵（确保探索性）

3. **性能指标**
   - `timing/sync_rollout_weights`：权重同步时间（应该减少）
   - `throughput/tflops`：训练吞吐量（应该提升）

### 判断训练是否正常

**正常现象**：
- ✅ `rollout_probs_diff_max` 随 sync_frequency 增大而增大
- ✅ `sync_rollout_weights` 时间显著减少
- ✅ 总训练时间减少
- ✅ 奖励曲线平滑上升

**异常现象（需要降低 sync_frequency）**：
- ❌ `rollout_probs_diff_max > 0.5`
- ❌ `ppo_clip_ratio > 0.7`（大量样本被裁剪）
- ❌ 奖励曲线剧烈波动或下降
- ❌ `approx_kl` 突然飙升

---

## ⚠️ 注意事项

### 1. 显存占用

方案2 **不会增加显存占用**，因为：
- 仍然是单批次训练
- 只是跳过了某些步骤的权重同步
- 显存占用与标准 one-step-off 相同

### 2. Off-Policy 程度

```
实际滞后步数 = 0 到 (sync_frequency - 1) 步

例如 sync_frequency=3：
- Step 1: 滞后 0 步（刚同步）
- Step 2: 滞后 1 步
- Step 3: 滞后 2 步
- Step 4: 滞后 0 步（再次同步）
```

### 3. 与标准 GRPO 的区别

| 特性 | 标准 GRPO | one-step-off (sync_freq=1) | multi-step-off (sync_freq=3) |
|------|----------|---------------------------|------------------------------|
| 模型更新 | 每步 | 每步 | 每步 |
| 权重同步 | 每步（自动） | 每步（手动） | 每 3 步 |
| 显存占用 | 低 | 低 | 低 |
| Off-policy | 无 | 1 步 | 1-3 步 |
| 代码位置 | `verl/trainer/` | `recipe/one_step_off_policy/` | `recipe/one_step_off_policy/` |

### 4. 何时不适用

不建议使用高 sync_frequency 的场景：
- ❌ 模型较小（< 1B 参数）：权重同步开销本来就小
- ❌ 学习率很大：策略变化快，off-policy 误差大
- ❌ 任务需要精确对齐（如微调对话模型）

---

## 🔄 回退到标准模式

如果多步滞后导致训练不稳定，可以随时回退：

```yaml
actor_rollout_ref:
  rollout:
    sync_frequency: 1  # 恢复为每步同步
```

或者完全回退到标准 GRPO（使用混合引擎）：

```bash
# 使用标准训练脚本
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml'
```

---

## 📚 参考资料

- **原始文档**: `/workspace/qingnan/verl/docs/advance/one_step_off.md`
- **标准配置**: `/workspace/qingnan/verl/verl/trainer/config/ppo_trainer.yaml`
- **示例脚本**:
  - `/workspace/qingnan/verl/recipe/one_step_off_policy/dapo_7b_math_fsdp2_4_12.sh`
  - `/workspace/qingnan/verl/recipe/one_step_off_policy/dapo_7b_math_megatron_4_12.sh`

---

## 🐛 故障排查

### 问题 1：日志没有显示 sync frequency 信息

**原因**：配置文件未生效

**解决**：
```bash
# 确认配置文件路径
python3 -m recipe.one_step_off_policy.main_ppo --help

# 使用命令行参数强制覆盖
python3 -m recipe.one_step_off_policy.main_ppo \
    actor_rollout_ref.rollout.sync_frequency=3
```

### 问题 2：训练变慢而非变快

**原因**：可能是其他瓶颈（如数据加载、奖励计算）

**解决**：
```bash
# 查看各阶段耗时
grep "sync_rollout_weights" training.log
grep "wait_prev_gen" training.log

# 如果 sync_rollout_weights 本来就很快（< 1s），提升 sync_frequency 收益有限
```

### 问题 3：训练发散

**原因**：sync_frequency 太大，off-policy 误差积累

**解决**：
1. 降低 sync_frequency（如从 5 → 3 → 2）
2. 降低学习率
3. 增加 PPO clip epsilon
4. 使用更小的 batch size

---

## 📝 总结

### 代码修改汇总

| 文件 | 行号 | 修改内容 |
|------|------|---------|
| `ray_trainer.py` | 154-157 | 添加 `sync_frequency` 配置读取 |
| `ray_trainer.py` | 349-353 | 条件同步权重逻辑 |
| `one_step_off_ppo_trainer.yaml` | 16-23 | 添加 `sync_frequency` 参数 |

### 快速开始

```bash
# 1. 修改配置文件
vim /workspace/qingnan/verl/recipe/one_step_off_policy/config/one_step_off_ppo_trainer.yaml
# 将 sync_frequency 改为 2 或 3

# 2. 启动训练
cd /workspace/qingnan/verl
bash recipe/one_step_off_policy/dapo_7b_math_fsdp2_4_12.sh

# 3. 监控日志
tail -f logs/training.log | grep -E "Multi-Step|rollout_probs_diff"
```

祝实验顺利！🚀
