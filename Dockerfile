FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install dependencies
RUN sed -i -e "s/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list && \
    sed -i -e "s/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && apt-get update -y && \
    apt-get install --assume-yes --fix-missing build-essential && \
    apt-get install -y curl  && \
    apt-get install -y apt-utils \
    tzdata \
    git wget curl vim unzip ffmpeg \
    build-essential cmake pkg-config \
    llvm meson \
    libegl1-mesa \
    libegl1-mesa-dev \
    libegl1 \
    libgl1 \
    mesa-utils \
    libsm6 libxext6 libxrender-dev \
    libosmesa6-dev \
    python3.10 \
    python3-pip
RUN sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list && apt-get update && apt-get build-dep -y mesa && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

# Install Conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_25.5.1-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    /opt/conda/bin/conda config --set show_channel_urls yes && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
ENV PATH=/opt/conda/bin:$PATH

# Install Vulkan libraries
RUN apt-get update && apt-get install -y libvulkan1 mesa-vulkan-drivers vulkan-tools libglvnd-dev
RUN mkdir -p /usr/share/vulkan/icd.d \
             /usr/share/glvnd/egl_vendor.d \
             /etc/vulkan/implicit_layer.d && \
    printf '%s\n' \
    '{' \
    '    "file_format_version" : "1.0.0",' \
    '    "ICD": {' \
    '        "library_path": "libGLX_nvidia.so.0",' \
    '        "api_version" : "1.2.155"' \
    '    }' \
    '}' > /usr/share/vulkan/icd.d/nvidia_icd.json && \
    printf '%s\n' \
    '{' \
    '    "file_format_version" : "1.0.0",' \
    '    "ICD" : {' \
    '        "library_path" : "libEGL_nvidia.so.0"' \
    '    }' \
    '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
    printf '%s\n' \
    '{' \
    '    "file_format_version" : "1.0.0",' \
    '    "layer": {' \
    '        "name": "VK_LAYER_NV_optimus",' \
    '        "type": "INSTANCE",' \
    '        "library_path": "libGLX_nvidia.so.0",' \
    '        "api_version" : "1.2.155",' \
    '        "implementation_version" : "1",' \
    '        "description" : "NVIDIA Optimus layer",' \
    '        "functions": {' \
    '            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",' \
    '            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"' \
    '        },' \
    '        "enable_environment": {' \
    '            "__NV_PRIME_RENDER_OFFLOAD": "1"' \
    '        },' \
    '        "disable_environment": {' \
    '            "DISABLE_LAYER_NV_OPTIMUS_1": ""' \
    '        }' \
    '    }' \
    '}' > /etc/vulkan/implicit_layer.d/nvidia_layers.json

# Set up application directory
WORKDIR /app
COPY simpler /app/simpler
COPY calvin /app/calvin
COPY libero /app/libero
COPY RoboTwin /app/RoboTwin
COPY maniskill2 /app/maniskill2
COPY habitat-lab /app/habitat-lab
COPY VLN-CE /app/VLN-CE

# Install simpler environment
RUN /opt/conda/bin/conda create -n simpler_env python=3.10 -y && \
    /bin/bash -c "source activate simpler_env && \
        cd simpler && \
        cd ManiSkill2_real2sim && pip install -e . && \
        cd .. && pip install -e . && \
        pip install matplotlib mediapy omegaconf hydra-core && pip install numpy==1.24.4 && \
        cd .."

# Install calvin environment
RUN /opt/conda/bin/conda create -n calvin_env python=3.8 -y && \
    /bin/bash -c "source activate calvin_env && \
        cd calvin && \
        pip uninstall setuptools -y && \
        pip install setuptools==57.5.0 && \
        bash install.sh && \
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple/ && \
        cd .."

# Install libero environment
RUN /opt/conda/bin/conda create -n libero_env python=3.8 -y && \
    /bin/bash -c "source activate libero_env && \
        cd libero && \
        pip uninstall setuptools -y && \
        pip install setuptools==57.5.0 && \
        pip install -r requirements.txt && \
        pip install -e . && \
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple/ && \
        cd .."

# Install RoboTwin environment
RUN /opt/conda/bin/conda create -n RoboTwin python=3.10 -y && \
/bin/bash -c "source activate RoboTwin && \
    cd RoboTwin && \
    export TORCH_CUDA_ARCH_LIST='7.5;8.0;8.9;9.0' && \
    bash script/_install.sh && \
    pip install omegaconf && \
    cd .."

ENV MS2_ASSET_DIR=/app/maniskill2/ManiSkill/data

# Install ManiSkill2 environment
RUN /opt/conda/bin/conda create -n maniskill2_env python=3.8 -y && \
/bin/bash -c "source activate maniskill2_env && \
    cd maniskill2/ManiSkill && \
    pip install -e . && \
    pip install gymnasium==0.29.1 && \
    cd ../ManiSkill2-Learn && \
    pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --index-url https://download.pytorch.org/whl/cu113 && \
    pip install -U fvcore==0.1.5.post20221221 && \
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html && \
    pip install ninja omegaconf && \
    pip install -e . && \
    cd ../.."

# Build Warp library for soft-body environments
RUN /bin/bash -c "source activate maniskill2_env && \
    cd maniskill2/ManiSkill && \
    export PYTHONPATH=\$PWD/warp_maniskill:\$PYTHONPATH && \
    python -m warp_maniskill.build_lib && \
    cd ../.."

# Install VLN-CE environment
RUN /opt/conda/bin/conda create -n vlnce python=3.8 -y && \
    /bin/bash -c "source activate vlnce && \
        cd /app && \
        wget https://api.anaconda.org/download/aihabitat/habitat-sim/0.1.7/linux-64/habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2 && \
        conda install -y habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2 && \
        rm habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2 && \
        cd /app/habitat-lab && \
        python -m pip install -r requirements.txt && \
        python -m pip install 'moviepy>=1.0.1' tb-nightly -i https://mirrors.aliyun.com/pypi/simple/ && \
        python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt && \
        python setup.py develop --no-deps && \
        cd /app && \
        cd /app/VLN-CE && \
        grep -v -E 'torch|torchvision|tensorflow' requirements.txt | pip install -r /dev/stdin && \
        cd /app && \
        python -m pip install 'pip<24.1' && \
        pip install setuptools==59.8.0 wheel==0.37.1 && \
        pip install gym==0.21.0 && \
        pip install gitpython matplotlib flask omegaconf && \
        pip install numpy==1.23.0 && \
        pip install torch==1.12.1 torchvision==0.13.1 && \
        pip install webdataset==0.1.103"

RUN /opt/conda/bin/conda init bash
CMD ["bash"]
