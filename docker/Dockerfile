ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDA_ALIAS="101"
ARG CUDNN="7"
ARG MMCV="1.3.1"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install mmediting
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmediting.git /mmediting
WORKDIR /mmediting
ENV FORCE_CUDA="1"
RUN pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA_ALIAS}/torch${PYTORCH}/index.html
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .
