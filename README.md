# Dexbotic Benchmark

A unified robot benchmarking framework that supports automated evaluation of CALVIN, LIBERO, VLA-Arena, RoboDojo, Simpler, RoboTwin 2.0, ManiSkill2, and VLN-CE environments.

## Overview

Dexbotic Benchmark provides a comprehensive evaluation framework for robotic learning algorithms across multiple environments:

- **CALVIN**: A large-scale dataset and benchmark for learning long-horizon manipulation tasks
- **LIBERO**: A benchmark for learning robotic manipulation from human demonstrations
- **VLA-Arena**: A benchmark for evaluating VLA safety, distractor robustness, extrapolation, and long-horizon behavior across L0-L2 difficulty levels
- **RoboDojo**: An Isaac Sim benchmark for generalist bimanual robot manipulation
- **Simpler**: A framework for evaluating and reproducing real-world robot manipulation policies (e.g., RT-1, RT-1-X, Octo) in simulation under common setups (e.g., Google Robot, WidowX+Bridge)
- **RoboTwin 2.0**: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation
- **ManiSkill2**: A benchmark for generalizable manipulation skill learning with diverse tasks and robot embodiments
- **VLN-CE**: A benchmark for Vision-and-Language Navigation in Continuous Environments

## Quick Start

### Prerequisites

**System Requirements:**
- A machine equipped with an NVIDIA GPU (single GPU recommended; tested on 2080Ti, A100, H100, and 4090)
- Docker with GPU support for containerized benchmarks
- Conda and an NVIDIA driver for local VLA-Arena evaluation

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
docker pull docker.io/dexmal/dexbotic_benchmark:latest
```

### Run with Docker

**Important Note:** The Docker image serves as a client that requires a separate dexbotic model server to be running. Make sure you have the dexbotic model server started before running the evaluation commands.

```bash
# Run CALVIN evaluation
docker run --gpus all --network host -v $(pwd):/workspace \
  docker.io/dexmal/dexbotic_benchmark:latest \
  bash /workspace/scripts/env_sh/calvin.sh /workspace/evaluation/configs/calvin/example_cavin.yaml

# Run LIBERO evaluation
docker run --gpus all --network host -v $(pwd):/workspace \
  docker.io/dexmal/dexbotic_benchmark:latest \
  bash /workspace/scripts/env_sh/libero.sh /workspace/evaluation/configs/libero/example_libero.yaml

# Run RoboDojo evaluation
git submodule update --init --recursive robodojo
# Follow the official RoboDojo asset flow: download with Git LFS, then generate
# the absolute paths used by CuRobo. On Ubuntu, install Git LFS once:
sudo apt-get install -y git-lfs
git lfs install
bash robodojo/scripts/init_assets.sh
(
  cd robodojo
  python utils/update_embodiment_config_path.py
)
ROBODOJO_ASSETS="$(cd robodojo && pwd -P)/Assets"

docker run --rm --gpus all --network host --ipc host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$(pwd):/workspace" \
  -v "${ROBODOJO_ASSETS}:/workspace/robodojo/Assets:ro" \
  -v "${ROBODOJO_ASSETS}:${ROBODOJO_ASSETS}:ro" \
  -w /workspace \
  dexmal/dexbotic_benchmark:latest \
  bash /workspace/scripts/env_sh/robodojo.sh /workspace/evaluation/configs/robodojo/example_robodojo.yaml

# Run Simpler evaluation
docker run --gpus all --network host -v $(pwd):/workspace\
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  docker.io/dexmal/dexbotic_benchmark:latest \
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
  docker.io/dexmal/dexbotic_benchmark:latest \
  bash scripts/env_sh/robotwin2.sh evaluation/configs/robotwin2/example_robotwin2.yaml

# Run ManiSkill2 evaluation
docker run --gpus all --network host -v $(pwd):/workspace \
  docker.io/dexmal/dexbotic_benchmark:latest \
  python evaluation/run_maniskill2_evaluation.py --config evaluation/configs/maniskill2/example_maniskill2.yaml

# Run VLN-CE evaluation (R2R)
docker run --gpus all \
  --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
  -v "$(pwd)":/workspace \
  -v /your/datasets/path/datasets/:/workspace/datasets \
  -w /workspace \
  docker.io/dexmal/dexbotic_benchmark:latest \
  bash scripts/env_sh/vlnce.sh \
  evaluation/configs/vlnce/r2r_baselines/navila_eval.yaml

# Run VLA-Arena evaluation
docker run --gpus all --network host -v $(pwd):/workspace -w /workspace \
  docker.io/dexmal/dexbotic_benchmark:latest \
  python3 evaluation/run_arena_evaluation.py \
  --config evaluation/configs/arena/example_arena.yaml

# Run VLA-Arena L0/L1/L2 concurrently
docker run --gpus all --network host -v $(pwd):/workspace -w /workspace \
  docker.io/dexmal/dexbotic_benchmark:latest \
  python3 evaluation/run_arena_evaluation.py \
  --config evaluation/configs/arena/example_arena.yaml \
  --set task_level all \
  --set parallelized true
```
Note: For LIBERO evaluation, use `example_pi0_libero.yaml` for PI0/PI05 and
`example_dm0_libero.yaml` for DM0. Switch scenarios by setting `benchmark` to
`libero_spatial`, `libero_goal`, `libero_object`, or `libero_10`. For CogAct,
use the scenario-specific configs directly: `libero_spatial.yaml`,
`libero_goal.yaml`, `libero_object.yaml`, or `libero_10.yaml`.

Note: The Arena evaluator uses the same external Dexbotic model-server boundary as LIBERO. Select suites with `task_suite_name`, difficulty with `task_level`, and repeatable evaluation seeds with `seeds`. Parallel workers share the configured model service, and server-global sampling RNG prevents exact serial/parallel trajectory reproduction. For native local setup and troubleshooting, see the [VLA-Arena installation and evaluation guide](docs/local_install.md#vla-arena-environment).

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
