#!/bin/bash

# 示例：将 PyTorch 模型转换为 ONNX 格式
# 请根据你的实际情况修改参数

# 基本用法示例
python scripts/pt_to_onnx.py \
    --task=Tracking-Flat-G1-v0 \
    --wandb_path=lbq/humanoid/ebloc4v5 \
    --model_filename=model_29500.pt \
    --output_dir=./exported_models \
    --onnx_filename=policy.onnx \
    --registry_name mscproject/wandb-registry-motions/walk1_subject1 \
    --headless

# 指定特定模型文件的示例
# python scripts/pt_to_onnx.py \
#     --task=Tracking-Flat-G1-v0 \
#     --wandb_path=your-org/your-project/your-run-id/model_1000.pt \
#     --output_dir=./exported_models \
#     --onnx_filename=policy_1000.onnx

echo "ONNX 模型转换完成！"