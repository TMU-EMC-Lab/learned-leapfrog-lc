# Training and evaluation environment used on the H100 server (torch 2.1.2, CUDA 11.8, numpy 1.26.4).
# Code and data are not copied into the image: mount the repository at /work, for example
#   docker build -t learned-leapfrog-lc docker/
#   docker run --gpus all --rm -v "$PWD":/work learned-leapfrog-lc python sweep_ic_generalization.py ...   (the full command is in README.md)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

RUN pip install --no-cache-dir numpy==1.26.4 tqdm==4.66.5 "matplotlib>=3.7,<3.10"

ENV MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1
WORKDIR /work
