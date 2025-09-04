"""Script to convert PyTorch model to ONNX format."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import pathlib

from isaaclab.app import AppLauncher

# add argparse arguments

parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--wandb_path", type=str, required=True, help="WandB run path to load the model from.")
parser.add_argument("--registry_name", type=str, required=True, help="WandB registry name for motion file.")
parser.add_argument("--output_dir", type=str, default="./exported_models", help="Output directory for ONNX model.")
parser.add_argument("--model_filename", type=str, default="model.pt", help="Model filename in WandB run.")
parser.add_argument("--onnx_filename", type=str, default="policy.onnx", help="Output ONNX filename.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import wandb

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Convert PyTorch model to ONNX format."""
    
    # load the motion file from the wandb registry
    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"
    import pathlib

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    print(f"[INFO]: Motion file loaded from registry: {env_cfg.commands.motion.motion_file}")
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Load model from WandB
    print(f"[INFO]: Loading model from WandB path: {args_cli.wandb_path}")
    
    api = wandb.Api()
    run_path = args_cli.wandb_path
    
    # Handle different WandB path formats
    if "model" in args_cli.wandb_path:
        run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        model_filename = args_cli.wandb_path.split("/")[-1]
    else:
        model_filename = args_cli.model_filename
    
    wandb_run = api.run(run_path)
    
    # Download the model file
    wandb_file = wandb_run.file(model_filename)
    temp_dir = "./temp_model"
    os.makedirs(temp_dir, exist_ok=True)
    model_path = os.path.join(temp_dir, model_filename)
    wandb_file.download(temp_dir, replace=True)
    
    print(f"[INFO]: Model downloaded to: {model_path}")
    
    # Load the trained model
    from rsl_rl.runners import OnPolicyRunner
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(model_path)
    
    # Create output directory
    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Export policy to ONNX
    print(f"[INFO]: Exporting model to ONNX format...")
    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=output_dir,
        filename=args_cli.onnx_filename,
    )
    
    # Attach metadata to ONNX model
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path, output_dir, args_cli.onnx_filename)
    
    print(f"[INFO]: ONNX model exported to: {os.path.join(output_dir, args_cli.onnx_filename)}")
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
