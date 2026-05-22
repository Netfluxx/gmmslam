cmake_minimum_required(VERSION 3.16)
project(gmmslam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ===========================================================================
# Dependencies
# ===========================================================================

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf2_ros
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  std_msgs
)

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)   # System Eigen 3.4 (needed by GIRA3D / SOGMM)

# GTSAM 4.2 was built against its bundled Eigen 3.3.  GIRA3D / SOGMM was
# built against system Eigen 3.4.  The two are ABI-incompatible so we must
# compile SOGMM-dependent sources with system Eigen and GTSAM-dependent
# sources with the bundled copy.
set(_gtsam_bundled_eigen "${GTSAM_INCLUDE_DIR}/gtsam/3rdparty/Eigen")
if(IS_DIRECTORY "${_gtsam_bundled_eigen}")
  # Bundled Eigen MUST come first: system Eigen 3.4 also lives under
  # ${GTSAM_INCLUDE_DIR} (at /usr/local/include/Eigen/) and would win
  # the #include <Eigen/Core> race if searched first.
  set(GTSAM_ALL_INCLUDE_DIRS "${_gtsam_bundled_eigen}" "${GTSAM_INCLUDE_DIR}")
  message(STATUS "GTSAM include: ${GTSAM_INCLUDE_DIR}, bundled Eigen: ${_gtsam_bundled_eigen}")
else()
  set(GTSAM_ALL_INCLUDE_DIRS "${GTSAM_INCLUDE_DIR}" "${EIGEN3_INCLUDE_DIR}")
  message(STATUS "GTSAM include: ${GTSAM_INCLUDE_DIR}, system Eigen: ${EIGEN3_INCLUDE_DIR}")
endif()
message(STATUS "System Eigen (for SOGMM): ${EIGEN3_INCLUDE_DIR}")
find_package(yaml-cpp REQUIRED)

option(GMMSLAM_USE_GMMAP "Enable optional GMMap fitting backend" OFF)
if(GMMSLAM_USE_GMMAP)
  find_package(OpenMP REQUIRED)
  find_package(fmt REQUIRED)
  find_path(GMMAP_INCLUDE_DIR
    NAMES gmm_map/map.h
    HINTS ENV GMMAP_ROOT ENV GMMAP_SOURCE_DIR
    PATH_SUFFIXES include)
  find_library(GMMAP_LIBRARY
    NAMES gmm_map
    HINTS ENV GMMAP_ROOT ENV GMMAP_BUILD_DIR
    PATH_SUFFIXES lib build .)
  find_path(GMMAP_RTREE_INCLUDE_DIR
    NAMES RTree/RTree.h
    HINTS ENV GMMAP_ROOT ENV GMMAP_SOURCE_DIR
    PATH_SUFFIXES libs/rtree/include)
  find_path(GMMAP_NIGH_INCLUDE_DIR
    NAMES nigh/lp_space.hpp
    HINTS ENV GMMAP_ROOT ENV GMMAP_SOURCE_DIR
    PATH_SUFFIXES libs/nigh/src)
  if(NOT GMMAP_INCLUDE_DIR OR NOT GMMAP_LIBRARY OR NOT GMMAP_RTREE_INCLUDE_DIR)
    message(FATAL_ERROR
      "GMMSLAM_USE_GMMAP=ON but GMMap was not found completely. Set "
      "GMMAP_SOURCE_DIR=/opt/GMMap and GMMAP_BUILD_DIR=/opt/GMMap/build, "
      "or leave GMMSLAM_USE_GMMAP=OFF.")
  endif()
  message(STATUS "GMMap include: ${GMMAP_INCLUDE_DIR}")
  message(STATUS "GMMap RTree include: ${GMMAP_RTREE_INCLUDE_DIR}")
  if(GMMAP_NIGH_INCLUDE_DIR)
    message(STATUS "GMMap nigh include: ${GMMAP_NIGH_INCLUDE_DIR}")
  endif()
  message(STATUS "GMMap library: ${GMMAP_LIBRARY}")
endif()

# --- nlohmann/json (header-only) ------------------------------------------
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# --- GIRA3D packages ------------------------------------------------------
# Colcon (--symlink-install, no --merge-install) installs each package under
# its own subdirectory: <workspace>/install/<pkg_name>/{include,lib,...}
set(GIRA3D_REC_INSTALL /root/gira_ws/gira3d-reconstruction/wet/install)
set(GIRA3D_REC_DRY_INSTALL /root/gira_ws/gira3d-reconstruction/dry/install)
set(GIRA3D_REG_INSTALL /root/gira_ws/gira3d-registration/wet/install)
set(GIRA3D_REG_DRY_INSTALL /root/gira_ws/gira3d-registration/dry/install)

set(NANOFLANN_INCLUDE_DIR "${GIRA3D_REC_DRY_INSTALL}/nanoflann_colcon/include")

list(APPEND CMAKE_PREFIX_PATH
  ${GIRA3D_REC_INSTALL}
  ${GIRA3D_REG_INSTALL}
  ${GIRA3D_REG_DRY_INSTALL}
)
find_package(self_organizing_gmm REQUIRED)
find_package(gmm REQUIRED)
find_package(dlib REQUIRED)

# Resolve per-package include dirs: colcon may expose them via
# <pkg>_INCLUDE_DIRS, <pkg>_INCLUDE_DIR, or only via an imported target.
# Fall back to the known colcon install-tree layout if variables are empty.
macro(gira3d_include_dir _pkg _install_base _out_var)
  if(${_pkg}_INCLUDE_DIRS)
    set(${_out_var} "${${_pkg}_INCLUDE_DIRS}")
  elseif(${_pkg}_INCLUDE_DIR)
    set(${_out_var} "${${_pkg}_INCLUDE_DIR}")
  else()
    set(${_out_var} "${_install_base}/${_pkg}/include")
    message(STATUS "${_pkg}: no cmake variable found, using fallback ${${_out_var}}")
  endif()
endmacro()

gira3d_include_dir(self_organizing_gmm ${GIRA3D_REC_INSTALL} SOGMM_INCLUDE_DIR)
gira3d_include_dir(gmm                 ${GIRA3D_REG_INSTALL} GMM_INCLUDE_DIR)

message(STATUS "self_organizing_gmm include: ${SOGMM_INCLUDE_DIR}")
message(STATUS "gmm include:                 ${GMM_INCLUDE_DIR}")

# ===========================================================================
# Catkin
# ===========================================================================

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES    gmmslam_lib
  CATKIN_DEPENDS
    roscpp tf2_ros sensor_msgs geometry_msgs nav_msgs visualization_msgs std_msgs
)

# ===========================================================================
# Library
# ===========================================================================

# SOGMM headers require Eigen 3.4 features (Eigen::all, Eigen::Vector).
# Compiled as a separate object so its include-path order gives system
# Eigen 3.4 priority, while the rest of the library uses GTSAM's bundled 3.3.
add_library(sogmm_obj OBJECT src/sogmm_fitting.cpp)
set_target_properties(sogmm_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(sogmm_obj PRIVATE
  include
  ${EIGEN3_INCLUDE_DIR}        # System Eigen 3.4 — MUST come first here
  ${SOGMM_INCLUDE_DIR}
  ${NANOFLANN_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}       # ros/ros.h for DBG logs in sogmm_fitting.cpp
)

add_library(gmmslam_lib SHARED
  src/config.cpp
  src/ros_helpers.cpp
  src/util/gmm_utils.cpp
  src/d2d_registration.cpp
  src/fixed_lag_backend.cpp
  src/gmmap_fitting.cpp
  src/global_pose_graph.cpp
  src/registration_manager.cpp
  src/solid.cpp
  src/place_recognition.cpp
  src/visualizer.cpp
  $<TARGET_OBJECTS:sogmm_obj>
)

target_include_directories(gmmslam_lib PUBLIC
  include
  ${GTSAM_ALL_INCLUDE_DIRS}    # GTSAM + bundled Eigen 3.3 — MUST precede catkin
  ${catkin_INCLUDE_DIRS}
  ${GMM_INCLUDE_DIR}
  ${NANOFLANN_INCLUDE_DIR}
)
if(GMMSLAM_USE_GMMAP)
  target_include_directories(gmmslam_lib PRIVATE
    ${GMMAP_INCLUDE_DIR}
    ${GMMAP_RTREE_INCLUDE_DIR}
    ${GMMAP_NIGH_INCLUDE_DIR}
  )
  target_compile_definitions(gmmslam_lib PRIVATE GMMSLAM_HAS_GMMAP=1)
endif()
# util/rtree.h (header-only), included from gmm_utils.cpp only
target_include_directories(gmmslam_lib PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(gmmslam_lib
  ${catkin_LIBRARIES}
  gtsam
  gtsam_unstable
  yaml-cpp
  dlib::dlib
  nlohmann_json::nlohmann_json
)
if(GMMSLAM_USE_GMMAP)
  target_link_libraries(gmmslam_lib
    ${GMMAP_LIBRARY}
    OpenMP::OpenMP_CXX
    fmt::fmt
  )
endif()

# ===========================================================================
# Executables
# ===========================================================================

add_executable(gmmslam_node             src/gmmslam_node.cpp)
add_executable(d2d_registration_node    src/d2d_registration_node.cpp)
add_executable(noisy_gt_publisher_node  src/noisy_gt_publisher_node.cpp)
add_executable(solid_descriptor_benchmark src/testing/solid_descriptor_benchmark.cpp)

foreach(_target gmmslam_node d2d_registration_node noisy_gt_publisher_node
                solid_descriptor_benchmark)
  target_link_libraries(${_target} gmmslam_lib)
endforeach()

# ===========================================================================
# Install
# ===========================================================================

install(TARGETS gmmslam_lib gmmslam_node d2d_registration_node noisy_gt_publisher_node
                      solid_descriptor_benchmark
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
