# Stage 1: Builder/Compiler
FROM python:3.10-slim as builder
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc
COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir --user -r /requirements.txt

# Stage 2: Runtime
FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu22.04

RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.10 python3-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 3 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local/lib/python3.10/site-packages /usr/local/lib/python3.10/dist-packages
COPY ./src /src
CMD ["python3","./src/scripts/gpu_test.py"]
