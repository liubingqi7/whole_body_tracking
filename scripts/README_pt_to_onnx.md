# PyTorch to ONNX 转换脚本使用说明

## 功能描述

`pt_to_onnx.py` 脚本用于将训练好的 PyTorch 模型转换为 ONNX 格式，便于部署和推理。

## 使用方法

### 基本用法

```bash
python scripts/pt_to_onnx.py \
    --task=Tracking-Flat-G1-v0 \
    --wandb_path=lbq/humanoid/ebloc4v5 \
    --registry_name=lbq-org/wandb-registry-motions/motion_name \
    --output_dir=./exported_models \
    --onnx_filename=policy.onnx
```

### 参数说明

- `--task`: 任务名称，例如 `Tracking-Flat-G1-v0`
- `--wandb_path`: WandB 运行路径，格式为 `{organization}/{project}/{run_id}`
- `--registry_name`: WandB registry 名称，用于加载 motion 文件，格式为 `{organization}-org/wandb-registry-motions/{motion_name}`
- `--output_dir`: 输出目录，默认为 `./exported_models`
- `--model_filename`: WandB 中的模型文件名，默认为 `model.pt`
- `--onnx_filename`: 输出的 ONNX 文件名，默认为 `policy.onnx`

### 示例

```bash
# 从 WandB 下载模型并转换为 ONNX
python scripts/pt_to_onnx.py \
    --task=Tracking-Flat-G1-v0 \
    --wandb_path=my-org/motion-tracking/abc12345 \
    --registry_name=my-org/wandb-registry-motions/walk_motion \
    --output_dir=./models \
    --onnx_filename=g1_policy.onnx

# 指定特定的模型文件
python scripts/pt_to_onnx.py \
    --task=Tracking-Flat-G1-v0 \
    --wandb_path=my-org/motion-tracking/abc12345/model_1000.pt \
    --registry_name=my-org/wandb-registry-motions/walk_motion \
    --output_dir=./models \
    --onnx_filename=g1_policy_1000.onnx
```

## 输出文件

脚本会在指定的输出目录中生成以下文件：

1. `policy.onnx` (或指定的文件名): ONNX 格式的模型文件
2. 模型包含以下输出：
   - `actions`: 动作输出
   - `joint_pos`: 关节位置
   - `joint_vel`: 关节速度
   - `body_pos_w`: 身体位置（世界坐标系）
   - `body_quat_w`: 身体四元数（世界坐标系）
   - `body_lin_vel_w`: 身体线速度（世界坐标系）
   - `body_ang_vel_w`: 身体角速度（世界坐标系）

## 注意事项

1. 确保已安装 Isaac Lab 和相关依赖
2. 需要有效的 WandB 访问权限
3. 模型文件会自动下载到临时目录，转换完成后会清理
4. ONNX 模型包含完整的元数据信息，便于部署时使用