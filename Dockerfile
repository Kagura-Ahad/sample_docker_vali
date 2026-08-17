# --- Base: matches what VALI needs (Ubuntu 22.04 + CUDA 12.2 devel headers) ---
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PATH=/opt/conda/bin:$PATH

# --- System packages (from your apt-get install steps) ---
RUN apt-get update && apt-get install -y \
    git python3 python3-pip cmake build-essential pkg-config \
    libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libavfilter-dev \
    libgl1-mesa-glx wget nano \
    && rm -rf /var/lib/apt/lists/*

# --- Miniconda (you installed this after finding system python3-pip insufficient) ---
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda init bash

# --- Accept Anaconda ToS for default channels (required since late 2025,
#     otherwise `conda create`/`conda install` fail non-interactively) ---
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- vali_env: python 3.11, matches your working setup ---
RUN /opt/conda/bin/conda create -n vali_env python=3.11 -y

# --- Python packages, installed INTO vali_env specifically ---
# Kept as a separate layer + requirements.txt so adding packages later
# only invalidates this one layer, not the conda/miniconda layers above.
COPY requirements.txt /tmp/requirements.txt
RUN /opt/conda/bin/conda run -n vali_env pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision \
    && /opt/conda/bin/conda run -n vali_env pip install -r /tmp/requirements.txt

# --- ffmpeg via conda-forge as a fallback if the system libav*-dev libs
#     aren't visible inside the conda env at runtime (you hit this once) ---
RUN /opt/conda/bin/conda install -n vali_env -c conda-forge ffmpeg -y

# --- Make vali_env the DEFAULT python everywhere: interactive shells,
#     non-interactive `bash -c "..."` calls (how Claude Code runs commands),
#     and VS Code tasks. Putting it first on PATH means no activation
#     step is needed at all — `python`, `pip`, `python3` all resolve to
#     vali_env's binaries unconditionally. ---
ENV PATH=/opt/conda/envs/vali_env/bin:$PATH \
    CONDA_DEFAULT_ENV=vali_env

# Keep .bashrc activation too, purely so the prompt shows "(vali_env)"
# for your own visual confirmation in interactive terminals — it's
# cosmetic now, not load-bearing.
RUN echo "conda activate vali_env" >> /root/.bashrc
SHELL ["/bin/bash", "-c"]

WORKDIR /workspace
CMD ["/bin/bash"]