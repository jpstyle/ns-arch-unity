FROM python:3.10.8

SHELL ["/bin/bash", "-c"]

# Import requirement file first and install required packages
WORKDIR /tmp
COPY python/requirements.txt .
COPY python/itl/memory/requirements.txt itl/memory/
COPY python/itl/vision/requirements.txt itl/vision/
COPY python/itl/lang/requirements.txt itl/lang/
COPY python/itl/symbolic_reasoning/requirements.txt itl/symbolic_reasoning/
COPY python/itl/practical_reasoning/requirements.txt itl/practical_reasoning/
COPY python/itl/lpmln/requirements.txt itl/lpmln/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade protobuf==3.20
RUN pip install --no-cache-dir gdown

# Some graphics related libraries needed
RUN apt update && apt install -y libgl1
RUN apt install -y vulkan-tools
RUN apt install -y mesa-utils

# Let's add in vim, less and rsync
RUN apt install -y vim
RUN apt install -y less
RUN apt install -y rsync

# Install virtual display buffer X server and X11 server utils
RUN apt install -y xvfb
RUN apt install -y x11-xserver-utils

# Create a non-root user
RUN useradd --create-home nonroot

# Import the repository content
RUN mkdir -p /home/nonroot/ns-arch-unity
RUN mkdir -p /mnt/data_volume
WORKDIR /home/nonroot/ns-arch-unity
COPY . .
RUN chown -R nonroot /home/nonroot
RUN chown -R nonroot /mnt/data_volume

# Login as non-root
USER nonroot

# Environment variables
ENV DISPLAY :99
ENV NVIDIA_DRIVER_CAPABILITIES all

ENTRYPOINT ["/bin/bash", "tools/container_internal_scripts/eval_with_xvfb.sh"]
