FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

ENV CUDA_HOME=/usr/local/cuda
RUN apt-get update && apt-get -y upgrade && apt-get -y install libglib2.0-0 libsm6 libxrender-dev libxext6
RUN conda install -y cudatoolkit cudnn
COPY . /workspace/mmsr
WORKDIR /workspace/mmsr
RUN cd ./codes && python3 -m pip install --no-cache-dir -r requirements.txt
# set TORCH_CUDA_ARCH_LIST properly. (6, 1) -> 6.1
ARG TORCH_CUDA_ARCH_LIST=6.1 
RUN cd ./codes/models/archs/dcn && python3 setup.py develop
CMD /bin/bash
