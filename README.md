# Dexbotic Benchmark

A unified robot benchmarking framework that supports automated evaluation of CALVIN, LIBERO, Simpler, RoboTwin 2.0, and ManiSkill2 environments.

## Overview

Dexbotic Benchmark provides a comprehensive evaluation framework for robotic learning algorithms across multiple environments:

- **CALVIN**: A large-scale dataset and benchmark for learning long-horizon manipulation tasks
- **LIBERO**: A benchmark for learning robotic manipulation from human demonstrations
- **Simpler**: A framework for evaluating and reproducing real-world robot manipulation policies (e.g., RT-1, RT-1-X, Octo) in simulation under common setups (e.g., Google Robot, WidowX+Bridge)
- **RoboTwin 2.0**: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation
- **ManiSkill2**: A benchmark for generalizable manipulation skill learning with diverse tasks and robot embodiments
- **VLN-CE**: A benchmark for Vision-and-Language Navigation in Continuous Environments

## Quick Start

### Prerequisites

**System Requirements:**
- A machine equipped with an NVIDIA GPU (single GPU recommended; tested on 2080Ti, A100, H100, and 4090)
- Docker with GPU support

```bash
# Clone the repository
git clone https://github.com/Dexmal/dexbotic-benchmark.git
cd dexbotic-benchmark

# Initialize submodules
git submodule update --init --recursive
```

### 🐳 Docker (Recommended) 

For users who prefer containerized deployment, you can use Docker to run the evaluation environments:


```bash
docker pull dexmal/dexbotic_benchmark
```

### Run with Docker

**Important Note:** The Docker image serves as a client that requires a separate dexbotic model server to be running. Make sure you have the dexbotic model server started before running the evaluation commands.

```bash
# Run CALVIN evaluation
docker run --gpus all --network host -v $(pwd):/workspace \
  dexbotic-benchmark \
  bash /workspace/scripts/env_sh/calvin.sh /workspace/evaluation/configs/calvin/example_cavin.yaml

# Run LIBERO evaluation
docker run --gpus all --network host -v $(pwd):/workspace \
  dexmal/dexbotic_benchmark \
  bash /workspace/scripts/env_sh/libero.sh /workspace/evaluation/configs/libero/example_libero.yaml

# Run Simpler evaluation
docker run --gpus all --network host -v $(pwd):/workspace\
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  dexmal/dexbotic_benchmark \
  bash scripts/env_sh/simpler.sh evaluation/configs/simpler/example_simpler.yaml

# Run RoboTwin evaluation
# Note: You need to download the RoboTwin assets and mount them to the container (ref: https://robotwin-platform.github.io/doc/usage/robotwin-install.html#4-download-assets-robotwin-od-texture-library-and-embodiments)
docker run --gpus all --network host \
  -v [path/to/assets]:[path/to/assets] \
  -v [path/to/assets]:/app/assets \
  -v [path/to/assets]:/app/RoboTwin/assets \
  -v $(pwd)/evaluation:/app/evaluation \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/result_test:/app/result_test \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  dexmal/dexbotic_benchmark \
  bash scripts/env_sh/robotwin2.sh evaluation/configs/robotwin2/example_robotwin2.yaml

# Run ManiSkill2 evaluation
docker run --gpus all --network host -v $(pwd):/workspace \
  dexbotic-benchmark \
  python evaluation/run_maniskill2_evaluation.py --config evaluation/configs/maniskill2/example_maniskill2.yaml

# Run VLN-CE evaluation (R2R)
docker run --gpus all \
  --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
  -v "$(pwd)":/workspace \
  -v /your/datasets/path/datasets/:/workspace/datasets \
  -w /workspace \
  dexbotic-benchmark \
  bash scripts/env_sh/vlnce.sh \
  evaluation/configs/vlnce/r2r_baselines/navila_eval.yaml
```
Note: For LIBERO evaluation, use `example_pi0_libero.yaml` for PI0/PI05 and
`example_dm0_libero.yaml` for DM0. Switch scenarios by setting `benchmark` to
`libero_spatial`, `libero_goal`, `libero_object`, or `libero_10`. For CogAct,
use the scenario-specific configs directly: `libero_spatial.yaml`,
`libero_goal.yaml`, `libero_object.yaml`, or `libero_10.yaml`.


Note: RoboTwin2.0 has 50 sub-tasks, and each sub-task has two levels of difficulty. According to the official setting of RoboTwin2.0, each subtask needs to be evaluated separately. You can modify the `task_name` and `task_config` parameters in the configuration file to select different subtasks and difficulty levels for evaluation. ref: https://robotwin-platform.github.io/leaderboard

### Viewing Results

After running the Docker commands, evaluation results will be saved in the location specified by the `output_dir` parameter in your configuration file. For example:

- **Results Location**: Check the `output_dir` field in your configuration file (e.g., `evaluation/configs/calvin/example_cavin.yaml`)
- **Default Output**: Results are typically saved in `./result_test/` directory by default
- **Log Files**: Console output contains detailed evaluation progress and result information
- **Configuration Files**: Evaluation configuration files are located in `evaluation/configs/` directory

## Local Installation

For detailed local installation instructions, please refer to the comprehensive guide in [docs/local_install.md](docs/local_install.md). 

## Contributing

We welcome contributions to improve the Dexbotic Benchmark framework. Please feel free to submit issues and pull requests.

## License

This project is licensed under the terms specified in the LICENSE file.
