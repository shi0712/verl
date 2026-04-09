# 自适应同步频率（Adaptive Sync Frequency）使用说明

## 📋 概述

本文档说明如何使用**自适应同步频率**功能，该功能根据训练稳定性（PPO clip ratio）动态调整 rollout 模型的权重同步频率，在保证训练稳定的前提下最大化训练速度。

---

## 🎯 核心原理

### 传统固定同步频率的问题

**固定 sync_frequency=3**：
```
Step 1-3:   训练稳定，clip_ratio=0.08  ← 可以更 off-policy
Step 4-6:   训练稳定，clip_ratio=0.06  ← 可以更 off-policy
Step 7-9:   训练不稳定，clip_ratio=0.55 ← 应该更 on-policy，但无法调整
```

### 自适应同步频率的优势

**自适应模式**：
```
Step 1-3:   clip_ratio=0.08 → sync_freq: 3→4  ✅ 自动延长
Step 4-7:   clip_ratio=0.06 → sync_freq: 4→5  ✅ 继续延长
Step 8-12:  clip_ratio=0.05 → sync_freq: 5→6  ✅ 继续延长
Step 13-18: clip_ratio=0.45 → sync_freq: 6→5  ✅ 检测到不稳定，自动缩短
```

**核心指标**：`actor/pg_clipfrac`（PPO clip ratio）
- **物理意义**：被 PPO 裁剪的样本比例，反映策略变化速度
- **判断标准**：
  - `< 0.1`：训练很稳定，可以增大 sync_frequency（更 off-policy）
  - `0.1 - 0.4`：正常范围，保持 sync_frequency
  - `> 0.4`：训练不稳定，减小 sync_frequency（更 on-policy）

---

## 🔧 修改的文件

### 1. 新增文件

**`/workspace/qingnan/verl/recipe/one_step_off_policy/adaptive_sync.py`**
- `AdaptiveSyncFrequency` 类实现
- 负责记录 clip_ratio 历史、计算平均值、调整 sync_frequency

### 2. 修改的文件

#### `/workspace/qingnan/verl/recipe/one_step_off_policy/ray_trainer.py`

**修改位置 1**: 第 154-176 行（`__init__` 方法）
```python
# 初始化同步频率控制（支持固定和自适应两种模式）
self.enable_adaptive_sync = config.actor_rollout_ref.rollout.get("adaptive_sync_frequency", False)

if self.enable_adaptive_sync:
    # 自适应模式
    from recipe.one_step_off_policy.adaptive_sync import AdaptiveSyncFrequency

    self.adaptive_sync = AdaptiveSyncFrequency(
        init_sync_freq=config.actor_rollout_ref.rollout.get("init_sync_frequency", 3),
        min_sync_freq=config.actor_rollout_ref.rollout.get("min_sync_frequency", 1),
        max_sync_freq=config.actor_rollout_ref.rollout.get("max_sync_frequency", 10),
        stable_threshold=config.actor_rollout_ref.rollout.get("stable_threshold", 0.1),
        unstable_threshold=config.actor_rollout_ref.rollout.get("unstable_threshold", 0.4),
    )
    self.sync_frequency = self.adaptive_sync.get_sync_frequency()
```

**修改位置 2**: 第 639-643 行（`fit()` 方法，记录 clip_ratio）
```python
# 如果启用自适应同步频率，记录 clip_ratio
if self.adaptive_sync is not None:
    clip_ratio = actor_output_metrics.get("actor/pg_clipfrac", None)
    if clip_ratio is not None:
        self.adaptive_sync.record_clip_ratio(clip_ratio)
```

**修改位置 3**: 第 368-377 行（`_async_gen_next_batch()` 方法，调整 sync_frequency）
```python
# sync weights from actor to rollout (with adaptive or fixed frequency control)
if self.sync_counter % self.sync_frequency == 0:
    # 如果启用自适应，在同步前计算新的 sync_frequency
    if self.adaptive_sync is not None:
        new_sync_freq = self.adaptive_sync.compute_and_adjust()
        self.sync_frequency = new_sync_freq

    # 执行权重同步
    self.sync_rollout_weights()
```

#### `/workspace/qingnan/verl/recipe/one_step_off_policy/config/one_step_off_ppo_trainer.yaml`

**修改位置**: 第 19-46 行
```yaml
actor_rollout_ref:
  rollout:
    # 是否启用自适应同步频率
    adaptive_sync_frequency: false  # 改为 true 启用自适应

    # 固定模式参数
    sync_frequency: 1

    # 自适应模式参数
    init_sync_frequency: 3
    min_sync_frequency: 1
    max_sync_frequency: 10
    stable_threshold: 0.1
    unstable_threshold: 0.4
```

---

## 📖 使用方法

### 方法 1：修改配置文件（推荐）

编辑 `/workspace/qingnan/verl/recipe/one_step_off_policy/config/one_step_off_ppo_trainer.yaml`：

```yaml
actor_rollout_ref:
  rollout:
    # 启用自适应模式
    adaptive_sync_frequency: true

    # 自适应参数（可选，使用默认值也可以）
    init_sync_frequency: 3      # 初始值
    min_sync_frequency: 1       # 最小值
    max_sync_frequency: 10      # 最大值
    stable_threshold: 0.1       # 稳定阈值
    unstable_threshold: 0.4     # 不稳定阈值
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
    actor_rollout_ref.rollout.adaptive_sync_frequency=true \
    actor_rollout_ref.rollout.init_sync_frequency=3 \
    actor_rollout_ref.rollout.max_sync_frequency=10
```

---

## 🧪 参数配置建议

### 保守方案（追求稳定）

```yaml
actor_rollout_ref:
  rollout:
    adaptive_sync_frequency: true
    init_sync_frequency: 2      # 从较小的值开始
    min_sync_frequency: 1
    max_sync_frequency: 5       # 限制最大值
    stable_threshold: 0.08      # 更严格的稳定标准
    unstable_threshold: 0.3     # 更低的不稳定阈值
```

**特点**：
- 同步频率范围小 [1, 5]
- 更容易触发"减小 sync_frequency"
- 适合首次尝试或关键任务

### 平衡方案（推荐）⭐

```yaml
actor_rollout_ref:
  rollout:
    adaptive_sync_frequency: true
    init_sync_frequency: 3
    min_sync_frequency: 1
    max_sync_frequency: 10
    stable_threshold: 0.1       # PPO 标准阈值
    unstable_threshold: 0.4
```

**特点**：
- 同步频率范围适中 [1, 10]
- 使用标准 PPO 的 clip ratio 阈值
- 适合大多数场景

### 激进方案（追求极致加速）

```yaml
actor_rollout_ref:
  rollout:
    adaptive_sync_frequency: true
    init_sync_frequency: 5      # 从较大的值开始
    min_sync_frequency: 2       # 提高最小值
    max_sync_frequency: 20      # 允许更大的间隔
    stable_threshold: 0.15      # 更宽松的稳定标准
    unstable_threshold: 0.6     # 更高的不稳定阈值
```

**特点**：
- 同步频率范围大 [2, 20]
- 不容易触发"减小 sync_frequency"
- 适合显存充足、追求速度的场景

---

## 📊 验证自适应是否生效

### 启动时的日志

应该看到：

```
[AdaptiveSyncFrequency] Initialized with:
  - init_sync_freq: 3
  - min_sync_freq: 1
  - max_sync_freq: 10
  - stable_threshold: 0.1
  - unstable_threshold: 0.4
[Multi-Step Off-Policy] Adaptive mode enabled, init_sync_frequency=3
```

### 训练过程中的日志

每次同步时会输出：

```
[AdaptiveSyncFrequency] avg_clip_ratio=0.085, sync_frequency: 3 → 4 (stable)
[AdaptiveSyncFrequency] avg_clip_ratio=0.065, sync_frequency: 4 → 5 (stable)
[AdaptiveSyncFrequency] avg_clip_ratio=0.250, sync_frequency=5 (normal range)
[AdaptiveSyncFrequency] avg_clip_ratio=0.450, sync_frequency: 5 → 4 (unstable)
```

### 关键监控指标

在 wandb/tensorboard 中关注：

1. **actor/pg_clipfrac**（每步）
   - 原始的 clip ratio 值
   - 应该在 [0, 1] 范围内

2. **adaptive/sync_frequency**（需要手动记录）
   - sync_frequency 的变化轨迹
   - 观察是否随训练阶段变化

3. **timing/sync_rollout_weights**
   - 权重同步耗时
   - 应该随 sync_frequency 增大而减少

4. **throughput/tflops**
   - 训练吞吐量
   - 应该随 sync_frequency 增大而提升

---

## 🔍 算法执行流程示例

### 时间线演示

```
初始化: sync_frequency = 3, sync_counter = 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→

Step 1:
  - _async_gen_next_batch(): sync_counter=0, 0 % 3 == 0 ✅ 同步
  - compute_and_adjust(): 无历史数据，保持 sync_freq=3
  - sync_rollout_weights()
  - sync_counter++  → 1
  - generate & train
  - actor 更新后: clip_ratio=0.12 → 记录到 history

Step 2:
  - _async_gen_next_batch(): sync_counter=1, 1 % 3 != 0 ⏭️ 跳过同步
  - sync_counter++  → 2
  - generate & train（用 Step 1 同步的权重）
  - actor 更新后: clip_ratio=0.15 → 记录到 history

Step 3:
  - _async_gen_next_batch(): sync_counter=2, 2 % 3 != 0 ⏭️ 跳过同步
  - sync_counter++  → 3
  - generate & train（用 Step 1 同步的权重）
  - actor 更新后: clip_ratio=0.18 → 记录到 history

Step 4:
  - _async_gen_next_batch(): sync_counter=3, 3 % 3 == 0 ✅ 同步
  - compute_and_adjust():
      * history = [0.12, 0.15, 0.18]
      * avg = 0.15
      * 0.1 < 0.15 < 0.4 → normal range
      * sync_freq 保持 = 3
      * 清空 history
  - sync_rollout_weights()
  - sync_counter++  → 4
  - generate & train
  - actor 更新后: clip_ratio=0.08 → 记录到新 history

Step 5:
  - _async_gen_next_batch(): sync_counter=4, 4 % 3 != 0 ⏭️ 跳过同步
  - sync_counter++  → 5
  - generate & train
  - actor 更新后: clip_ratio=0.06 → 记录到 history

Step 6:
  - _async_gen_next_batch(): sync_counter=5, 5 % 3 != 0 ⏭️ 跳过同步
  - sync_counter++  → 6
  - generate & train
  - actor 更新后: clip_ratio=0.07 → 记录到 history

Step 7:
  - _async_gen_next_batch(): sync_counter=6, 6 % 3 == 0 ✅ 同步
  - compute_and_adjust():
      * history = [0.08, 0.06, 0.07]
      * avg = 0.07
      * 0.07 < 0.1 → stable!
      * sync_freq = min(3+1, 10) = 4  ✅ 调整
      * 清空 history
  - sync_rollout_weights()
  - sync_counter++  → 7
  - generate & train
  - actor 更新后: clip_ratio=0.09 → 记录到新 history

Step 8-10:
  - sync_counter % 4 != 0，跳过同步
  - 记录 clip_ratio

Step 11:
  - sync_counter=10, 10 % 4 != 0 ⏭️ 跳过

Step 12:
  - sync_counter=11, 11 % 4 != 0 ⏭️ 跳过

（以此类推，每 4 步同步一次）
```

**关键点**：
1. sync_frequency 的调整发生在**即将同步的时刻**
2. 新的 sync_frequency **立即生效**
3. 历史数据在调整后**清空**，重新开始记录

---

## ⚠️ 注意事项

### 1. 初始阶段 clip_ratio 不可用

**问题**：前 `init_sync_frequency` 步没有 clip_ratio 历史

**解决**：
- 代码已处理：如果 history 为空，保持 sync_frequency 不变
- 建议 `init_sync_frequency=3`，这样第一次调整时已有 2-3 个样本

### 2. 不要过于频繁调整

**当前设计**：每次同步时都会调整一次

**优点**：
- 响应快，能及时发现问题
- 调整幅度小（每次 ±1），不会剧烈变化

**如果观察到频繁振荡**：
- 检查 `stable_threshold` 和 `unstable_threshold` 是否合理
- 可以适当拉大两个阈值的间隔（如 0.08 和 0.45）

### 3. 显存占用不变

自适应模式**不会增加显存占用**，因为：
- 仍然是单批次训练
- 只是动态调整同步频率
- 显存占用与固定模式相同

### 4. 与固定模式对比

| 特性 | 固定模式 | 自适应模式 |
|------|---------|-----------|
| 配置复杂度 | 简单（1个参数） | 中等（5个参数） |
| 训练稳定性 | 取决于 sync_frequency 选择 | 自动调整，更鲁棒 |
| 加速效果 | 固定 | 动态优化，理论上更好 |
| 调试难度 | 简单 | 需要监控 clip_ratio |
| 适用场景 | 参数已知 | 探索性实验 |

---

## 🔄 如何在固定模式和自适应模式之间切换

### 切换到自适应模式

```yaml
actor_rollout_ref:
  rollout:
    adaptive_sync_frequency: true  # 改为 true
    init_sync_frequency: 3
```

### 切换回固定模式

```yaml
actor_rollout_ref:
  rollout:
    adaptive_sync_frequency: false  # 改为 false
    sync_frequency: 3               # 使用固定值
```

---

## 🐛 故障排查

### 问题 1：日志没有显示 AdaptiveSyncFrequency 信息

**原因**：配置文件未生效

**解决**：
```bash
# 检查配置
python3 -m recipe.one_step_off_policy.main_ppo --help

# 使用命令行参数强制启用
python3 -m recipe.one_step_off_policy.main_ppo \
    actor_rollout_ref.rollout.adaptive_sync_frequency=true
```

### 问题 2：sync_frequency 一直保持初始值

**原因**：可能 clip_ratio 一直在正常范围 [0.1, 0.4]

**解决**：
- 这是正常现象，说明训练稳定
- 可以适当降低 `stable_threshold`（如 0.08）来更容易触发调整

### 问题 3：sync_frequency 到达上限/下限后不再变化

**原因**：已达到 `max_sync_frequency` 或 `min_sync_frequency`

**解决**：
- 检查日志，确认是否达到边界
- 如果需要，调整 `max_sync_frequency` 或 `min_sync_frequency`

### 问题 4：训练变慢而非变快

**原因**：其他瓶颈（数据加载、奖励计算等）

**解决**：
```bash
# 查看各阶段耗时
grep "sync_rollout_weights" training.log
grep "wait_prev_gen" training.log
grep "reward" training.log

# 如果 sync_rollout_weights 本来就很快，提升空间有限
```

---

## 📚 相关文档

- **固定模式文档**: `/workspace/qingnan/verl/recipe/one_step_off_policy/MULTI_STEP_OFF_POLICY_USAGE.md`
- **原始 one-step-off 文档**: `/workspace/qingnan/verl/docs/advance/one_step_off.md`
- **标准 GRPO 配置**: `/workspace/qingnan/verl/verl/trainer/config/ppo_trainer.yaml`

---

## 🎯 快速开始

```bash
# 1. 修改配置文件启用自适应
vim /workspace/qingnan/verl/recipe/one_step_off_policy/config/one_step_off_ppo_trainer.yaml
# 将 adaptive_sync_frequency 改为 true

# 2. 启动训练
cd /workspace/qingnan/verl
bash recipe/one_step_off_policy/dapo_7b_math_fsdp2_4_12.sh

# 3. 监控日志
tail -f logs/training.log | grep -E "AdaptiveSyncFrequency|pg_clipfrac"
```

---

## 📝 总结

### 代码修改汇总

| 文件 | 行号 | 修改内容 |
|------|------|---------|
| `adaptive_sync.py` | 新增文件 | AdaptiveSyncFrequency 类实现 |
| `ray_trainer.py` | 154-176 | 初始化自适应控制器 |
| `ray_trainer.py` | 639-643 | 记录 clip_ratio |
| `ray_trainer.py` | 368-377 | 调整 sync_frequency |
| `one_step_off_ppo_trainer.yaml` | 19-46 | 添加自适应参数配置 |

### 推荐配置

```yaml
# 首次尝试使用这个配置
actor_rollout_ref:
  rollout:
    adaptive_sync_frequency: true
    init_sync_frequency: 3
    min_sync_frequency: 1
    max_sync_frequency: 10
    stable_threshold: 0.1
    unstable_threshold: 0.4
```

祝实验顺利！🚀
