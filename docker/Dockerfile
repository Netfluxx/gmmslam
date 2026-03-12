# --- Base: NVIDIA CUDA + Ubuntu 20.04 ---

# rebuild no cache: docker compose build --no-cache
# rebuild with cache: docker compose build  OR docker compose build slam && docker compose down slam && docker compose up -d slam

# run: docker compose run --rm disal_slam OR docker compose up -d disal_slam && docker exec -it disal_slam bash

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
    ros-noetic-rviz \
    apt-utils x11-apps && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update

# Rewrite SSH git URLs to HTTPS so submodules clone without an SSH key
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"

RUN pip3 install colcon-common-extensions scipy open3d

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

# --- Eigen 3.4.0 from source (gira3d-reconstruction needs 3.4) ---
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
COPY docker/patch_open3d_colcon.py docker/patch_open3d.py /tmp/
RUN python3 /tmp/patch_open3d_colcon.py && python3 /tmp/patch_open3d.py

# Install Python deps and build
RUN cd ${GIRA_WS}/gira3d-reconstruction && \
    python3.8 -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install colcon-common-extensions wheel && \
    .venv/bin/pip install 'numpy>=1.19,<2.0' 'scipy>=1.5' scikit-learn opencv-python scikit-image future matplotlib termcolor 'open3d>=0.13' && \
    bash -c "source .venv/bin/activate && CMAKE_POLICY_VERSION_MINIMUM=3.5 ./build"

# --- X11 / display libs ---
ENV DISPLAY=:0
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libegl1-mesa \
    libx11-6 libxext6 libxrender1 libsm6 libglib2.0-0 \
    libsndio7.0 libasound2 libpulse0 \
    python3.8-tk && \
    rm -rf /var/lib/apt/lists/*

# libsndio.so.7 symlink expected by Webots
RUN set -eux; \
    if [ -f /usr/lib/x86_64-linux-gnu/libsndio.so.7.0 ]; then \
      ln -sf /usr/lib/x86_64-linux-gnu/libsndio.so.7.0 /usr/lib/x86_64-linux-gnu/libsndio.so.7; \
    elif [ -f /lib/x86_64-linux-gnu/libsndio.so.7.0 ]; then \
      ln -sf /lib/x86_64-linux-gnu/libsndio.so.7.0 /lib/x86_64-linux-gnu/libsndio.so.7; \
    else \
      echo "ERROR: libsndio.so.7.0 not found"; exit 1; \
    fi; \
    ldconfig

# --- Webots (Cyberbotics apt repo) ---
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg && \
    curl -fsSL https://cyberbotics.com/Cyberbotics.asc | gpg --dearmor -o /usr/share/keyrings/cyberbotics.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cyberbotics.gpg] https://cyberbotics.com/debian/ binary-amd64/" > /etc/apt/sources.list.d/cyberbotics.list && \
    apt-get update && apt-get install -y --no-install-recommends webots && \
    rm -rf /var/lib/apt/lists/*

# --- ROS1 Noetic Webots bridge ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-webots-ros && \
    rm -rf /var/lib/apt/lists/*

# --- Webots environment variables ---
ENV WEBOTS_HOME=/usr/local/webots
ENV PYTHONPATH=${PYTHONPATH}:${WEBOTS_HOME}/lib/controller/python
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${WEBOTS_HOME}/lib/webots:${WEBOTS_HOME}/lib/controller
ENV ROS_PACKAGE_PATH=${WEBOTS_HOME}/projects/default/controllers/ros:${ROS_PACKAGE_PATH}
ENV ROS_PACKAGE_PATH=/opt/gmmslam:${ROS_PACKAGE_PATH}

# --- GMM-SLAM Python node ---
# python/ is bind-mounted at runtime via compose.yaml → /opt/gmmslam/python
# Only config files (not edited at runtime) are baked into the image.
COPY package.xml /opt/gmmslam/package.xml
COPY config/ /opt/gmmslam/config/
RUN mkdir -p /opt/gmmslam/scripts && \
    printf '#!/usr/bin/env bash\nexec python3 /opt/gmmslam/python/gmmslam.py "$@"\n' > /opt/gmmslam/scripts/gmmslam_node.py && \
    printf '#!/usr/bin/env bash\nexec python3 /opt/gmmslam/python/d2d_registration_node.py "$@"\n' > /opt/gmmslam/scripts/d2d_registration_node.py && \
    chmod +x /opt/gmmslam/scripts/gmmslam_node.py /opt/gmmslam/scripts/d2d_registration_node.py

# --- RViz visualization ---
RUN mkdir -p /opt/gmmslam/rviz
COPY config/gmmslam.rviz /opt/gmmslam/rviz/gmmslam.rviz

# Install ROS python bindings into the reconstruction venv so we can use its
# python3 (which has sogmm_py) while still importing rospy.
RUN ${GIRA_WS}/gira3d-reconstruction/.venv/bin/pip install rospkg catkin-pkg plotly

# Source the reconstruction workon (activates venv + adds sogmm_py).
# Manually add registration wet/install paths + examples/python to PYTHONPATH (don't activate its venv).
RUN printf '#!/usr/bin/env bash\n'\
'source /opt/ros/%s/setup.bash\n'\
'cd %s/gira3d-reconstruction && source ./workon\n'\
'export ROS_PACKAGE_PATH="/opt/gmmslam:${ROS_PACKAGE_PATH}"\n'\
'export PYTHONPATH="%s/gira3d-registration/wet/install/gmm_d2d_registration_py:%s/gira3d-registration/wet/install/gmm:%s/gira3d-registration/wet/install/gmm_d2d_registration:%s/gira3d-registration/wet/src/gmm_d2d_registration_examples/python:${PYTHONPATH}:/opt/gmmslam/python"\n'\
'rosrun rviz rviz -d ~/rviz_setup_gmmslam/clean_rviz_gmmslam.rviz >/tmp/gmmslam_rviz.log 2>&1 &\n'\
'exec roslaunch /opt/gmmslam/python/launch/gmmslam.launch "$@"\n' \
      "${ROS_DISTRO}" "${GIRA_WS}" "${GIRA_WS}" "${GIRA_WS}" "${GIRA_WS}" "${GIRA_WS}" > /usr/local/bin/run_gmmslam && \
    chmod +x /usr/local/bin/run_gmmslam

# Test simple GMM registration script
RUN printf '#!/usr/bin/env bash\n'\
'source /opt/ros/%s/setup.bash\n'\
'cd %s/gira3d-reconstruction && source ./workon\n'\
'export PYTHONPATH="%s/gira3d-registration/wet/install/gmm_d2d_registration_py:%s/gira3d-registration/wet/install/gmm:%s/gira3d-registration/wet/install/gmm_d2d_registration:%s/gira3d-registration/wet/src/gmm_d2d_registration_examples/python:${PYTHONPATH}:/opt/gmmslam/python"\n'\
'exec python3 /opt/gmmslam/python/test_simple_gmm_reg.py "$@"\n' \
      "${ROS_DISTRO}" "${GIRA_WS}" "${GIRA_WS}" "${GIRA_WS}" "${GIRA_WS}" "${GIRA_WS}" > /usr/local/bin/run_gmm_test && \
    chmod +x /usr/local/bin/run_gmm_test

# RViz launch script
RUN printf '#!/usr/bin/env bash\n'\
'source /opt/ros/%s/setup.bash\n'\
'exec rosrun rviz rviz -d /opt/gmmslam/rviz/gmmslam.rviz "$@"\n' \
      "${ROS_DISTRO}" > /usr/local/bin/run_rviz && \
    chmod +x /usr/local/bin/run_rviz

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc && \
    echo "source ${CATKIN_WS}/devel/setup.bash" >> /etc/bash.bashrc && \
    echo "source ${GIRA_WS}/gira3d-reconstruction/workon" >> /etc/bash.bashrc && \
    echo "export PYTHONPATH=\${PYTHONPATH}:/opt/gmmslam/python" >> /etc/bash.bashrc

CMD ["bash"]