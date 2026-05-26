# Define an argument for the GCC version
ARG GCC_VERSION=13.3.0

# Builder stage: Installing and compiling GCC, with debugging and clarity
# For GPU set image to nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04
FROM ubuntu:18.04 AS builder
ARG GCC_VERSION

# skip interactive request of time zone
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Step 1: Install dependencies for GCC building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        curl \
        ca-certificates \
        git \
        cmake \
        ninja-build \
        bzip2 \
        xz-utils \
        pkg-config \
        libssl-dev \
        zlib1g-dev \
        libffi-dev

# Step 2: install a newer cmake from sources (3.26 is the minimal version for DARTS)
RUN apt purge -y cmake && \
    wget https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3.tar.gz && \
    tar -xvf cmake-3.29.3.tar.gz && \
    cd cmake-3.29.3 && ./bootstrap --prefix=/opt/cmake --parallel=$(nproc) && make -j$(nproc) && make install && \
    /opt/cmake/bin/cmake --version && \
    cd / && rm -rf cmake-3.29.3 cmake-3.29.3.tar.gz

# Step 3: Install conda-forge
# Set Bash as the default shell (needed for the "source" command)
SHELL ["/bin/bash", "-c"]
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/conda && \
    rm -f Miniforge3-*.sh

# Step 4: Download and unpack GCC source
WORKDIR /tmp
RUN wget https://ftp.nluug.nl/languages/gcc/releases/gcc-${GCC_VERSION}/gcc-${GCC_VERSION}.tar.gz && \
  tar -xzf gcc-${GCC_VERSION}.tar.gz

# Step 3: Build GCC
WORKDIR /tmp/gcc-${GCC_VERSION}

RUN ./contrib/download_prerequisites && \
  mkdir build && cd build && \
  # Configure only for the C++ language
  ../configure --disable-multilib --enable-languages=c++ --prefix=/opt/gcc-${GCC_VERSION} --disable-debug && \
  make -j$(nproc) > build.log 2>&1

RUN cd build && make install-strip && \
    # Clean up build artifacts to reduce builder stage size (less for Kaniko to snapshot)
    cd /tmp && rm -rf gcc-${GCC_VERSION} gcc-${GCC_VERSION}.tar.gz


# Step 6: assembling clean environment for runtime use with GCC
FROM ubuntu:18.04

ARG GCC_VERSION

# skip interactive request of time zone
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Step 4: Copy GCC binaries and libraries; ignore build dependencies
COPY --from=builder /opt/gcc-${GCC_VERSION} /opt/gcc-${GCC_VERSION}
# Copy miniconda binaries
COPY --from=builder /opt/conda /opt/conda
# Copy cmake binaries
COPY --from=builder /opt/cmake /opt/cmake

# Step 5: Install dependencies, mold linker, and clean up in a single layer
#         (fewer layers = fewer Kaniko snapshots = lower peak memory during image build)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        wget \
        git \
        make \
        smbclient \
        valgrind \
        libglu1-mesa-dev \
        libxcursor1 \
        libxinerama1 && \
    # Install mold linker (much faster linking for template-heavy C++ code)
    wget -q https://github.com/rui314/mold/releases/download/v2.40.4/mold-2.40.4-x86_64-linux.tar.gz && \
    tar -xzf mold-2.40.4-x86_64-linux.tar.gz --strip-components=1 -C /usr/local && \
    rm mold-2.40.4-x86_64-linux.tar.gz && \
    mold --version && \
    # Clean up apt cache to reduce layer size
    rm -rf /var/lib/apt/lists/*

# Set up GCC environment variables for proper paths in execution
ENV CROSS_ROOT=/opt/gcc-${GCC_VERSION}/bin
ENV PATH="/opt/gcc-${GCC_VERSION}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/gcc-${GCC_VERSION}/lib64:${LD_LIBRARY_PATH}"
ENV CC=/opt/gcc-${GCC_VERSION}/bin/gcc
ENV CXX=/opt/gcc-${GCC_VERSION}/bin/g++
# ENV LD=/opt/gcc-${GCC_VERSION}/bin/ld

# Add cmake to path
ENV PATH="/opt/cmake/bin:${PATH}"

COPY Toolchain.cmake ${CROSS_ROOT}/../lib/
ENV CMAKE_TOOLCHAIN_FILE=${CROSS_ROOT}/../lib/Toolchain.cmake
