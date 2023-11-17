FROM nvcr.io/nvidia/pytorch:22.08-py3


RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1 \
    libgomp1 \
    libegl1 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt ./
RUN pip install -r requirements.txt

RUN useradd -ms /bin/bash reim-nerf
USER reim-nerf