
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