FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS cuda_env
# Base image with CUDA support

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support
# Install PyTorch with CUDA support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install opencv-python


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


# FROM python:3.10-slim-buster

# WORKDIR /app

# # Copy requirements file and install dependencies
# COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt

# Copy application files
COPY . .

# # Copy PyTorch libraries from the CUDA environment
# COPY --from=cuda_env /app/lib/python3.10/site-packages/torch* ./lib/python3.10/site-packages/

# Set environment variables (optional)
# ENV TASK=jpeg_car
# ENV MODEL_NAME=Jpeg_dynamic.pth
# ENV INPUT_FOLDER="/app/input_folder"
# ENV OUTPUT_FOLDER="/app/output_folder"
# ENV MODEL_PATH="/app/pretrained_models"
ENV RANK=0

# COPY entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh

# Define default command to run the application
# CMD ["python3", "car.py"]

# CMD ["ls", "/usr/lib/python3.10/site-packages"]