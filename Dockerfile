# Base Image
FROM openvino/ubuntu20_dev:latest AS base

USER root

# Install requirements
ENV TIMEZONE=Asia/Taipeia \
    DEBIAN_FRONTEND=noninteractive

# Setting TimeZone
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get -yq update \
    && apt-get -yq install tzdata \
    && ln -fs /usr/share/zoneinfo/${TIMEZONE} /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

# Install requirements
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
    libcanberra-gtk-module libcanberra-gtk3-module

# Install OpenCV Library
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libopencv-dev

# Install OpenBLAS Library
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libopenblas-dev

# Move to target workspace
WORKDIR /workspace

# Define Command
CMD [ "bash" ]