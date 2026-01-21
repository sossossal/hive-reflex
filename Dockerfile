# Use an official scalable base image
FROM ubuntu:22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    verilator \
    gtkwave \
    cmake \
    autoconf \
    gperf \
    bison \
    flex \
    libfl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install RISC-V Toolchain (pre-compiled xPack)
RUN wget https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v12.2.0-3/xpack-riscv-none-elf-gcc-12.2.0-3-linux-x64.tar.gz \
    && tar -xzf xpack-riscv-none-elf-gcc-12.2.0-3-linux-x64.tar.gz -C /opt \
    && rm xpack-riscv-none-elf-gcc-12.2.0-3-linux-x64.tar.gz
ENV PATH="/opt/xpack-riscv-none-elf-gcc-12.2.0-3/bin:${PATH}"

# Install Python requirements
COPY requirements.txt /tmp/requirements.txt
# Create a dummy requirements.txt if not exists to avoid copy error, 
# but best practice is to have it. I will generate it next.
RUN if [ -f /tmp/requirements.txt ]; then pip3 install -r /tmp/requirements.txt; else pip3 install numpy cocotb pytest; fi

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Default command
CMD ["/bin/bash"]
