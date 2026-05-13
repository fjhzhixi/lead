# Planning Decoder / Action Head 分析

> 适用条件：`use_planning_decoder=true`（后训练阶段）

---

## 1. 规划解码器架构

规划解码器位于 `lead/tfv6/planning_decoder.py`，由两个子模块组成。

### PlanningContextEncoder（Key/Value 来源）

融合以下输入，输出 `[B, N, 256]` 的 token 序列：

| Token 类型 | 编码方式 | 说明 |
|---|---|---|
| BEV features | `Conv2d → 256` + 正弦位置编码 | 空间栅格特征 |
| velocity | `Linear(1 → 256)` | 归一化车速 |
| command | `Linear(command_dim → 256)` | 6类离散导航命令（CARLA） |
| target point | `Linear(2 → 256)` | 当前/前/后目标点（CARLA） |
| radar queries | `Linear(256 → 256)` | 20个雷达 token（CARLA） |
| acceleration | `Linear(1 → 256)` | 加速度（非 CARLA 数据集） |

所有状态 token 额外叠加一个**可学习位置嵌入** (`status_pos_embedding`)。

### PlanningDecoder（Transformer 交叉注意力）

```
PlanningDecoder
├── PlanningContextEncoder         (提供 K/V)
├── Learned queries                nn.Parameter [1, num_queries, 256]
├── TransformerDecoder             6层 × 8头，d_model=256，GELU + LayerNorm
└── 输出头（按配置条件启用）:
    ├── route_decoder:         Linear(256 → 2)                      [predict_spatial_path=True]
    ├── wp_decoder:            Linear(256 → 2)                      [predict_temporal_spatial_waypoints=True]
    ├── heading_decoder:       Linear(256 → 1)                      [NavSim 数据]
    └── target_speed_decoder:  Linear(256→256) → ReLU → Linear(256→8)  [predict_target_speed=True]
```

**Query 布局**（按顺序拼接）：

| 切片 | 数量 | 预测目标 |
|---|---|---|
| `[0 : num_route_points_prediction]` | 10 | 空间路径检查点 |
| `[10 : 10 + num_way_points_prediction]` | 8（CARLA）| 时空轨迹点 |
| `[-1]` | 1 | 目标速度分布 |

---

## 2. 输出格式

| 输出张量 | Shape | 说明 |
|---|---|---|
| `pred_route` | `[B, 10, 2]` | 空间路径检查点（x, y），累积偏移解码 |
| `pred_future_waypoints` | `[B, 8, 2]` | 时空轨迹点，4 Hz × 2 s = 8 个点 |
| `pred_target_speed_distribution` | `[B, 8]` | 8 个速度 bin 的 logits |
| `pred_target_speed_scalar` | `[B]` | 解码后的标量速度（m/s），推理/评估用 |
| `pred_headings` | `[B, 8]` | 累积航向角（NavSim 专用） |

**解码方式**：route 和 waypoints 均预测**增量差值**，再通过 `torch.cumsum` 累加为绝对轨迹坐标。

**目标速度 bin**（CARLA）：`[0.0, 4.0, 8.0, 10.0, 13.89, 16.0, 17.78, 20.0]` m/s，采用**two-hot 软标签编码**；brake 标志激活时全部概率压到 bin 0。

---

## 3. 损失函数与权重

### 规划损失（`PlanningDecoder.compute_loss`）

| 损失名 | 损失函数 | 说明 |
|---|---|---|
| `loss_spatio_temporal_waypoints` | `F.l1_loss`（mean） | 预测轨迹 vs GT `future_waypoints[:, :8]`；NavSim 额外加 heading L1 |
| `loss_target_speed` | `F.cross_entropy` | two-hot GT vs 原始 logits |
| `loss_spatial_route` | `F.l1_loss` | ADE（全 10 点）+ FDE（终点） |

三个规划损失的**原始权重各为 `1.0`**。

### 权重归一化

所有损失（规划 + 辅助感知）在每个 epoch 开始时**归一化到总和为 1.0**，因此每个损失的实际占比取决于激活的任务数量。

**辅助感知损失**（权重均为 `1.0`）：

- `loss_semantic`、`loss_depth`（`1e-5`）、`loss_bev_semantic`
- `loss_center_net_heatmap`、`loss_center_net_wh`、`loss_center_net_offset`
- `loss_center_net_yaw_class`、`loss_center_net_yaw_res`、`loss_center_net_velocity`
- `radar_loss`

---

## 4. 预训练 vs 后训练差异

| 方面 | 预训练（`use_planning_decoder=False`） | 后训练（`use_planning_decoder=True`） |
|---|---|---|
| PlanningDecoder 实例化 | 否 | 是，query 随机初始化 |
| 规划损失 | 全部权重为 `0.0` | 各为 `1.0`（参与归一化） |
| `is_pretraining` | `True` | `False` |
| `skip_first / skip_last` | 各 1 帧 | 各 8 帧（`num_way_points_prediction`） |
| 数据集桶 | `FullPretrainBucketCollection` | `FullPosttrainBucketCollection` |
| 权重加载 | 从零训练 | `strict=False` 加载预训练权重 |
| 辅助感知任务 | 激活 | 同样激活，联合训练 |

**关键加载行为**：后训练时以 `load_file=pretrain/model_XXXX.pth` 加载骨干网络和感知头权重（`strict=False`），新增的 `PlanningDecoder` 从零开始训练，骨干继续联合微调。

---

## 5. 监督标签的数据读取流程

### 原始存储（每帧 `metas/<frame:04d>.pkl`）

由 expert 采集时写入，关键规划字段：

| 字段 | 类型/Shape | 说明 |
|---|---|---|
| `future_positions` | `float32 (T+1, 3)` | **已在 ego 系**下的未来位置，通过 `T_world→ego @ pos_world` 计算 |
| `future_yaws` | `float32 (T+1,)` | 相对当前 yaw 的增量角 |
| `target_speed` | `float` m/s | Expert 规划的目标速度（含障碍/限速/IDM 推理后） |
| `brake` | `bool` | CARLA `VehicleControl.brake` |
| `route` | `list[(x,y)]` | 采集时已转到 ego 系的路径点 |
| `next_target_points_<d>` | `list[list[float]]` | 世界系导航点（用于 target_point，不直接作损失） |

---

### Dataset 读取 & 变换（Disk → Batch Tensor）

#### `future_waypoints` — 用于 `loss_spatio_temporal_waypoints`

```python
# CARLA: waypoints_spacing=5, num_way_points_prediction=8
# 对应时间步: 0.25s, 0.5s, ..., 2.0s（20 Hz 采样）
indices = [5, 10, 15, 20, 25, 30, 35, 40]
future_waypoints = future_positions[indices, :2]   # (8, 2), ego 系, 单位 m

# sensor perturbation 数据增强（y向平移 + yaw旋转）
future_waypoints = perturbate_waypoints(future_waypoints, y_perturb, yaw_perturb)
```
→ batch key: `data["future_waypoints"]`，shape `(B, 8, 2)`，**不归一化**

#### `route` — 用于 `loss_spatial_route`

```python
route = meta["route"][:num_route_points_smoothing]   # 已是 ego 系, ≤20 点
route = perturbate_route(route, ...)                  # 同步扰动

if config.smooth_route:
    route = smooth_path(config, route, target_first_distance=2.5)  # 三次样条平滑

route = route[:num_route_points_prediction]           # 取前 10 点, (10, 2)
```
→ batch key: `data["route"]`，shape `(B, 10, 2)`，**不归一化**

#### `target_speed` + `brake` — 用于 `loss_target_speed`

```python
data["target_speed"] = float(meta["target_speed"])   # 标量, m/s
data["brake"]        = meta["brake"]                 # bool

# 损失计算时动态做 two-hot 编码:
# brake=True  → 全部概率压到 bin 0 (0 m/s)
# brake=False → 线性插值到相邻两个 bin
# bins = [0.0, 4.0, 8.0, 10.0, 13.89, 16.0, 17.78, 20.0] m/s
target_speed_dist = encode_two_hot(target_speed, bins, brake)   # (B, 8)
```
→ batch keys: `data["target_speed"]` `(B,)`，`data["brake"]` `(B,)`

#### `target_point` — 输入到 Context Encoder（不参与 loss）

```python
# 世界系 → ego 系坐标变换
ego_point = R(yaw).T @ (world_point - ego_pos)
ego_point = perturbate_target_point(ego_point, ...)
```
→ batch keys: `data["target_point"]`、`data["target_point_next"]`、`data["target_point_previous"]`，shape `(B, 2)`

---

### Batch Key → 规划损失映射

| Batch Key | Shape | 规划损失 | 损失函数 |
|---|---|---|---|
| `"future_waypoints"` | `(B, 8, 2)` | `loss_spatio_temporal_waypoints` | `F.l1_loss(pred, label)`（ADE；NavSim 额外加 heading L1） |
| `"target_speed"` + `"brake"` | `(B,)` | `loss_target_speed` | `F.cross_entropy(logits, two_hot_label)` |
| `"route"` | `(B, 10, 2)` | `loss_spatial_route` | `F.l1_loss`（ADE 全 10 点）+ FDE（终点单独加权） |

---

### `skip_first / skip_last` 帧对齐

后训练时 `skip_last = num_way_points_prediction = 8`，确保帧 `f` 后面至少有 `8 × 5 = 40` 帧（`future_positions[40]` 必然存在，覆盖 2 秒预测域）。`skip_first = 8` 保证足够的历史帧供 past_positions/speeds 使用。

**模型侧对应**：`PlanningDecoder` 通过 `torch.cumsum(wp_decoder(queries), dim=1)` 将预测的**增量差值**累加为绝对 ego 系坐标，再与上述 `future_waypoints` 标签做 L1 损失。

---

## 6. 参考文件

- `lead/tfv6/planning_decoder.py` — 模型定义
- `lead/training/config_training.py` — 相关配置项（`use_planning_decoder`、`predict_target_speed`、`predict_spatial_path` 等）
- `scripts/posttrain_ddp.sh` — 后训练启动脚本
