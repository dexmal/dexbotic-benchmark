
# Local Installation Guide

This document provides detailed instructions for setting up and running the Dexbotic Benchmark locally.

## Prerequisites

```bash
bash scripts/env_setup.sh
```

## Simpler Environment

### Setup

```bash
# Create conda environment for Simpler
conda create -n simpler_env python=3.10 -y
conda activate simpler_env

# Install Simpler ManiSkill2
cd simpler 
cd ManiSkill2_real2sim && pip install -e .
cd .. && pip install -e .

# Install additional dependencies
pip install matplotlib mediapy omegaconf hydra-core numpy==1.24.4
cd ..
```

### Running Evaluation

```bash
# Using shell script (recommended)
bash scripts/env_sh/simpler.sh [path/to/config]

# Or run directly with Python
python3 evaluation/run_simpler_evaluation.py --config [path/to/config]

# Override configuration parameters
python evaluation/run_simpler_evaluation.py \
  --config [path/to/config] \
  --set base_url http://localhost:7891 \
  --set output_dir [path/to/output]
```

## LIBERO Environment

### Setup

```bash
# Create conda environment for LIBERO
conda create -n libero_env python=3.8 -y
conda activate libero_env

# Install LIBERO
cd libero
pip uninstall setuptools -y
pip install setuptools==57.5.0
pip install -r requirements.txt && pip install -e .

# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
cd ..
```

### Running Evaluation

```bash
# Using shell script (recommended)
bash scripts/env_sh/libero.sh [path/to/config]

# Or run directly with Python
python3 evaluation/run_libero_evaluation.py --config [path/to/config]

# Override configuration parameters
python evaluation/run_libero_evaluation.py \
  --config [path/to/config] \
  --set base_url http://localhost:7891 \
  --set output_dir [path/to/output]
```

## VLA-Arena Environment

VLA-Arena runs locally in a dedicated conda environment and calls the same
external Dexbotic model service used by LIBERO. The setup mirrors the runtime
baseline of `docker.io/dexmal/dexbotic_benchmark:latest`:
Ubuntu 22.04, CUDA 12.1, and NVIDIA EGL rendering. VLA-Arena additionally
requires Python 3.11, NumPy 1.26.4, and robosuite 1.5.1.

### Setup

```bash
# Install the system libraries used by the reference environment.
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  pkg-config \
  ffmpeg \
  libegl1 \
  libegl1-mesa \
  libegl1-mesa-dev \
  libgl1 \
  libglvnd-dev \
  libosmesa6-dev \
  libsm6 \
  libxext6 \
  libxrender-dev

# Initialize the VLA-Arena submodule and its packaged benchmark assets.
git submodule update --init --recursive arena

# Create the local Arena environment.
conda create -n arena python=3.11 pip -y
conda activate arena
python -m pip install --upgrade pip setuptools wheel

# Match the CUDA 12.1 PyTorch runtime used by the reference environment.
python -m pip install torch==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121

# Install VLA-Arena and the Dexbotic evaluator dependencies.
python -m pip install -e arena
python -m pip install \
  omegaconf \
  opencv-python-headless \
  requests \
  tqdm \
  Pillow \
  PyYAML \
  imageio-ffmpeg

# Persist the headless MuJoCo rendering configuration in this environment.
conda env config vars set -n arena \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  EGL_PLATFORM=device
conda deactivate
conda activate arena
```

The machine needs a working NVIDIA driver for MuJoCo EGL rendering. CUDA is not
used for model inference in this environment; the evaluator sends observations
to the external service configured by `base_url`.

### Running Evaluation

```bash
# Run with the example configuration.
conda activate arena
python3 evaluation/run_arena_evaluation.py \
  --config evaluation/configs/arena/example_arena.yaml

# Override configuration parameters. --set may be repeated.
python3 evaluation/run_arena_evaluation.py \
  --config evaluation/configs/arena/example_arena.yaml \
  --set task_suite_name extrapolation_unseen_objects \
  --set task_level 2 \
  --set base_url http://localhost:7891 \
  --set output_dir results/arena_unseen_objects_l2

# Run L0, L1, and L2 in three evaluator processes against one model service.
python3 evaluation/run_arena_evaluation.py \
  --config evaluation/configs/arena/safety_dynamic_obstacles.yaml \
  --set task_level all \
  --set parallelized true \
  --set base_url http://localhost:7891 \
  --set output_dir results/arena_safety_dynamic_parallel
```

The main Arena options are:

- `task_suite_name`: one suite name, a YAML list of suite names, or `all`. The
  `all` value evaluates the eleven Arena suites and excludes the upstream
  LIBERO compatibility suites.
- `task_level`: `0`, `1`, `2`, or `all` for all three difficulty levels.
- `parallelized`: when `true`, requires `task_level: all` and evaluates L0, L1,
  and L2 concurrently in three spawned processes. Per-level artifacts are kept
  under `output_dir/parallel_levels`, while merged results retain the standard
  output layout. It defaults to `false`.
- `num_trials_per_task`: number of episodes per task and seed.
- `seeds`: YAML list of evaluation seeds. For a CLI override, use a
  comma-separated value such as `--set seeds 7,42,1000`.
- `base_url`: URL of the external Dexbotic model server.
- `output_dir`: directory in which evaluation artifacts are written.

The example configuration evaluates all eleven suites at L0-L2 with seeds 7,
42, and 1000.

Each run writes `results.json`, `leaderboard.json`, the resolved `config.yaml`,
logs, and selected rollout videos under `output_dir`. Multi-seed runs also
write one `seed_<seed>` directory per seed and a timestamped
`batch_summary_<timestamp>.json` containing per-seed and aggregate SR/CC
statistics. Cumulative cost applies to safety suites.

### Parallel Level Execution

With `parallelized: true`, the evaluator creates one spawned process for each
of L0, L1, and L2. Each process owns its simulator, Python random state,
evaluator instance, logger, and per-level output directory. The workers do not
modify shared simulator state. Their only shared external dependency is the
model service selected by `base_url`.

All three workers send requests to that same service concurrently. Therefore,
parallel evaluation requires a server that:

- supports multiple evaluator clients without mixing requests or responses;
- has memory/history disabled, because the current evaluator request schema
  does not provide a distinct session ID for each level worker;
- uses the intended model code, checkpoint, robot transform, normalization
  statistics, and request defaults; and
- can sustain three request streams within `request_timeout`.

Parallel execution preserves the benchmark task and environment semantics, but
it does not guarantee byte-identical model trajectories. A stochastic backend
that draws from one process-global RNG consumes samples in request-arrival
order, which changes under concurrency. For exact serial/parallel alignment,
the client and server would need request-local deterministic sampling. The
current Arena configuration does not expose such an option, so use serial
evaluation when strict trajectory reproducibility is required.

### Arena Troubleshooting

- A platform status of `Killed` with exit code 143 means the evaluator received
  `SIGTERM`. Inspect platform events or the submitting controller; this is not
  by itself evidence of a Python or simulator failure. Partial per-level
  outputs remain under `output_dir/parallel_levels`.
- If requests succeed but every rollout reaches 300 steps with zero success,
  rerun the same workload with `parallelized: false`. If the serial run also
  fails, check the model checkpoint, server transform and normalization, and
  request-field handling before investigating evaluator concurrency.
- If serial succeeds but parallel differs, check server-side global sampling
  RNG, default-session memory, request timeouts, and capacity. Exact alignment
  requires request-local sampling seeds.
- `use_pruned_init: true` selects packaged initial states and is an explicit
  experiment, not a general remedy for a model-service mismatch. The default
  `false` follows seeded `env.reset()` behavior.

## RoboDojo Environment

### Setup

Initialize the RoboDojo submodule, download its assets, and install its local
runtime from the repository root. The asset steps below follow the
[official RoboDojo installation guide](https://robodojo-benchmark.com/doc/usage/install-and-download/).

```bash
git submodule update --init --recursive robodojo
bash robodojo/scripts/install.sh -i

# The official asset downloader requires Git LFS. On Ubuntu:
sudo apt-get install git-lfs
git lfs install

# Run RoboDojo's official download and path-generation steps from its root.
(
  cd robodojo
  bash scripts/init_assets.sh
  python utils/update_embodiment_config_path.py
)
```

### Running Evaluation

```bash
# Evaluate the task selected by `task_name` in the YAML
bash scripts/env_sh/robodojo.sh \
  evaluation/configs/robodojo/example_robodojo.yaml

# One-task smoke run without changing the YAML
python evaluation/run_robodojo_evaluation.py \
  --config evaluation/configs/robodojo/example_robodojo.yaml \
  --set task_name stack_bowls \
  --set seeds 0 \
  --set eval_num 1
```

The default adapter uses the legacy multipart `/process_frame` endpoint with a
14D joint state/action, action horizon 15, temperature 1.0, and a 120-second
request timeout. WebSocket reset events clear only local adapter state; they
are not forwarded to the HTTP inference service.

By default, the evaluator runs each seed in a separate Isaac SimulationApp
process. This bounds native PhysX/RTX resource accumulation while the local
policy bridge and remote inference service stay warm. Set
`simulation_app_scope: lane` only for tasks whose persistent-process memory
behavior has been validated. Every episode still constructs and closes a fresh
`EvalEnv` and USD stage.

cuRobo planner reuse is opt-in and is disabled by default. Enable it for a
persistent lane with:

```bash
export ROBODOJO_CACHE_CUROBO_PLANNERS=1
```

The cache is process-local and keyed by planner YAML, joints, timestep, and
table height; unset the variable to preserve the original planner lifecycle.

## CALVIN Environment

### Setup

```bash
# Create conda environment for CALVIN
conda create -n calvin_env python=3.8 -y
conda activate calvin_env

# Install CALVIN
cd calvin
pip uninstall setuptools -y
pip install setuptools==57.5.0
bash install.sh

# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
cd ..
```

### Running Evaluation

```bash
# Using shell script (recommended)
bash scripts/env_sh/calvin.sh [path/to/config]

# Or run directly with Python
python3 evaluation/run_calvin_evaluation.py --config [path/to/config]

# Override configuration parameters
python evaluation/run_calvin_evaluation.py \
  --config [path/to/config] \
  --set base_url http://localhost:7891 \
  --set output_dir [path/to/output]
```

## RoboTwin2 Environment

### Setup

```bash
# Create conda environment for RoboTwin2
conda create -n RoboTwin python=3.10 -y 
conda activate RoboTwin

# Install RoboTwin2
cd RoboTwin
export TORCH_CUDA_ARCH_LIST='7.5;8.0;8.9;9.0' && \
bash script/_install.sh
bash script/_download_assets.sh
pip install omegaconf
cd ..
```

### Running Evaluation

```bash
# Using shell script (recommended)
bash scripts/env_sh/robotwin2.sh [path/to/config]

# Or run directly with Python
python3 evaluation/run_robotwin2_evaluation.py --config [path/to/config] 

# Override configuration parameters
python evaluation/run_robotwin2_evaluation.py --config [path/to/config] \
    --set base_url http://localhost:7891 \
    --set output_dir [output_dir]
```

## ManiSkill2 Environment

### Setup

```bash
# Create conda environment for ManiSkill2
conda create -n maniskill2_env python=3.8 -y
conda activate maniskill2_env

# Install ManiSkill2
cd maniskill2/ManiSkill
pip install -e .

# Install additional dependencies
pip install gymnasium==0.29.1

# Install ManiSkill2-Learn
cd ../ManiSkill2-Learn
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --index-url https://download.pytorch.org/whl/cu113
pip install -U fvcore==0.1.5.post20221221
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
pip install ninja omegaconf
pip install -e .
cd ../..

# Build Warp library for soft-body environments
cd maniskill2/ManiSkill
export PYTHONPATH=$PWD/warp_maniskill:$PYTHONPATH
python -m warp_maniskill.build_lib
cd ../..

# Set asset directory environment variable
export MS2_ASSET_DIR=$(pwd)/maniskill2/ManiSkill/data
```

### Optional: Download Assets

Some ManiSkill2 environments require additional assets. You can download them as needed:

```bash
# Download assets for specific environments
conda activate maniskill2_env
python -m mani_skill2.utils.download_asset PickCube-v0 --non-interactive
python -m mani_skill2.utils.download_asset StackCube-v0 --non-interactive
python -m mani_skill2.utils.download_asset PickSingleYCB-v0 --non-interactive
python -m mani_skill2.utils.download_asset PickSingleEGAD-v0 --non-interactive
python -m mani_skill2.utils.download_asset PickClutterYCB-v0 --non-interactive

# Or download all assets
python -m mani_skill2.utils.download_asset all --non-interactive
```

### Running Evaluation

```bash
# Run with Python script
python evaluation/run_maniskill2_evaluation.py --config evaluation/configs/maniskill2/example_maniskill2.yaml

# Override configuration parameters
python evaluation/run_maniskill2_evaluation.py \
  --config evaluation/configs/maniskill2/example_maniskill2.yaml \
  --set env_name StackCube-v0 \
  --set num_episodes 20 \
  --set render true

# Run with VLA agent (requires VLA service)
python evaluation/run_maniskill2_evaluation.py \
  --config evaluation/configs/maniskill2/example_maniskill2.yaml \
  --set base_url http://localhost:7891
```

### Available Environments

- `PickCube-v0` - Pick and place cube task
- `StackCube-v0` - Stack cubes task
- `PickSingleYCB-v0` - Pick single YCB object
- `PickSingleEGAD-v0` - Pick single EGAD object
- `PickClutterYCB-v0` - Pick from cluttered YCB objects

## VLN-CE Environment

### Setup

```bash
# Create conda environment for VLN-CE
conda create -n vlnce python=3.8 -y
conda activate vlnce

# Install habitat-sim v0.17
wget https://api.anaconda.org/download/aihabitat/habitat-sim/0.1.7/linux-64/habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2
conda install -y habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2

# Install habitat-lab v0.17
cd habitat-lab
python -m pip install -r requirements.txt
python -m pip install "moviepy>=1.0.1" tb-nightly
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --no-deps
cd ..

# Install vln-ce
cd VLN-CE
grep -v -E "torch|torchvision|tensorflow" requirements.txt | pip install -r /dev/stdin
cd ..

# Add missing dependencies
pip install gitpython matplotlib flask omegaconf
pip install numpy==1.23.0
pip install torch==1.12.1 torchvision==0.13.1
pip install webdataset==0.1.103

# Fix gym version
pip install "setuptools<60"
python -m pip install "pip<24.1"
pip install "gym<=0.21.0"

# Remove the installation package
rm habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2
```

### Running Evaluation

```bash
# Using shell script (recommended)
bash scripts/env_sh/vlnce.sh [path/to/config]

# Or run directly with Python
python evaluation/run_vlnce_evaluation.py --config [path/to/config]

# Example: Run R2R evaluation
python evaluation/run_vlnce_evaluation.py --config evaluation/configs/vlnce/r2r_baselines/navila_eval.yaml
```
