# --- Base: NVIDIA CUDA + Ubuntu 20.04 ---
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO noetic
ENV CATKIN_WS /root/catkin_ws
ENV GIRA_WS /root/gira_ws

# --- Add ROS Noetic apt repository ---
RUN apt-get update && apt-get install -y --no-install-recommends curl lsb-release && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    rm -rf /var/lib/apt/lists/*

# --- Install ROS1 Noetic ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl lsb-release \
    python3-pip python3-vcstool \
    python3-rosdep python3-catkin-tools \
    ros-noetic-ros-base ros-noetic-catkin \
    apt-utils x11-apps && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update

# Rewrite SSH git URLs to HTTPS so submodules clone without an SSH key
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"


RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

RUN pip3 install colcon-common-extensions

RUN mkdir -p ${CATKIN_WS}/src && mkdir -p ${GIRA_WS}

# --- build GTSAM 4.2 from source ---
WORKDIR /tmp
RUN git clone --branch 4.2 https://github.com/borglab/gtsam.git && \
    mkdir -p gtsam/build && cd gtsam/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DGTSAM_BUILD_WITH_MARCH_NATIVE=ON -DGTSAM_USE_SYSTEM_EIGEN=FALSE -DGTSAM_WITH_TBB=ON && \
    make -j$(nproc) && make install

WORKDIR ${CATKIN_WS}
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make"

# --- GIRA3D system dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-venv python3.8-dev \
    libboost-all-dev libblas-dev && \
    rm -rf /var/lib/apt/lists/*

# --- Eigen 3.4.0 from source (Ubuntu 20.04 ships 3.3.7, gira3d-reconstruction needs 3.4) ---
RUN git clone --branch 3.4.0 https://gitlab.com/libeigen/eigen.git /tmp/eigen && \
    mkdir /tmp/eigen/build && cd /tmp/eigen/build && \
    cmake .. && make install && ldconfig && \
    rm -rf /tmp/eigen

# --- GIRA3D Registration ---
WORKDIR ${GIRA_WS}
RUN git clone --recursive https://github.com/gira3d/gira3d-registration.git && \
    cd gira3d-registration && \
    python3.8 -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install numpy colcon-common-extensions && \
    bash -c "source .venv/bin/activate && CFLAGS='-fPIC' CXXFLAGS='-fPIC' ./build.sh" && \
    .venv/bin/pip install tqdm matplotlib open3d

# --- GIRA3D Reconstruction: clone + init submodules ---
RUN git clone https://github.com/gira3d/gira3d-reconstruction.git ${GIRA_WS}/gira3d-reconstruction && \
    cd ${GIRA_WS}/gira3d-reconstruction && \
    git checkout master && \
    git submodule update --recursive --init

# Patch Open3D CMakeLists files to control CUDA architectures
COPY patch_open3d_colcon.py patch_open3d.py /tmp/
RUN python3 /tmp/patch_open3d_colcon.py && python3 /tmp/patch_open3d.py

# Install Python deps and build
RUN cd ${GIRA_WS}/gira3d-reconstruction && \
    python3.8 -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install colcon-common-extensions wheel && \
    .venv/bin/pip install numpy scikit-learn opencv-python scikit-image future matplotlib termcolor && \
    bash -c "source .venv/bin/activate && CMAKE_POLICY_VERSION_MINIMUM=3.5 ./build"

# --- X11 Forwarding ---
ENV DISPLAY=:0
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libegl1-mesa && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source ${CATKIN_WS}/devel/setup.bash" >> /root/.bashrc && \
    echo "source ${GIRA_WS}/gira3d-reconstruction/workon" >> /root/.bashrc

CMD ["bash"]