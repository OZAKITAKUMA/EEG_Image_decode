FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

WORKDIR /home/moepy/ozakitakuma/EEG_Image_decode

RUN apt-get update && apt-get install -y \
    git \
    openssh-client \
    ca-certificates \
    wget \
    curl \
    unzip \
    vim \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

CMD ["bash"]