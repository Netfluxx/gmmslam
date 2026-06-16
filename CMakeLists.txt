cmake_minimum_required(VERSION 3.16)
project(gmmslam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ===========================================================================
# ROS 2 / system dependencies
# ===========================================================================

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(fmt REQUIRED)

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

# GTSAM 4.2 commonly carries a bundled Eigen. Keep that include before system
# Eigen for GTSAM-dependent sources, while GMMap/GIRA dependencies still see
# system Eigen through their own exported include order.
set(_gtsam_bundled_eigen "${GTSAM_INCLUDE_DIR}/gtsam/3rdparty/Eigen")
if(IS_DIRECTORY "${_gtsam_bundled_eigen}")
  set(GTSAM_ALL_INCLUDE_DIRS "${_gtsam_bundled_eigen}" "${GTSAM_INCLUDE_DIR}")
else()
  set(GTSAM_ALL_INCLUDE_DIRS "${GTSAM_INCLUDE_DIR}" "${EIGEN3_INCLUDE_DIR}")
endif()

# ===========================================================================
# GMMap and GIRA3D registration dependencies
# ===========================================================================

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
    "GMMap is required for the ROS 2 Humble port. Set GMMAP_SOURCE_DIR and "
    "GMMAP_BUILD_DIR, or install GMMap so gmm_map/map.h and libgmm_map are visible.")
endif()

set(GIRA3D_REG_INSTALL "$ENV{GIRA3D_REG_INSTALL}")
set(GIRA3D_REG_DRY_INSTALL "$ENV{GIRA3D_REG_DRY_INSTALL}")
if(NOT GIRA3D_REG_INSTALL)
  set(GIRA3D_REG_INSTALL /root/gira_ws/gira3d-registration/wet/install)
endif()
if(NOT GIRA3D_REG_DRY_INSTALL)
  set(GIRA3D_REG_DRY_INSTALL /root/gira_ws/gira3d-registration/dry/install)
endif()

list(APPEND CMAKE_PREFIX_PATH
  ${GIRA3D_REG_INSTALL}
  ${GIRA3D_REG_DRY_INSTALL}
)
find_package(gmm REQUIRED)
find_package(dlib REQUIRED)

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

gira3d_include_dir(gmm ${GIRA3D_REG_INSTALL} GMM_INCLUDE_DIR)

# ===========================================================================
# Library
# ===========================================================================

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
)

target_include_directories(gmmslam_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GTSAM_ALL_INCLUDE_DIRS}
  ${GMM_INCLUDE_DIR}
  ${GMMAP_INCLUDE_DIR}
  ${GMMAP_RTREE_INCLUDE_DIR}
  ${GMMAP_NIGH_INCLUDE_DIR}
)
target_include_directories(gmmslam_lib PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(gmmslam_lib PRIVATE GMMSLAM_HAS_GMMAP=1)

ament_target_dependencies(gmmslam_lib
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  std_msgs
  tf2_msgs
  tf2_ros
)

target_link_libraries(gmmslam_lib
  gtsam
  gtsam_unstable
  yaml-cpp
  dlib::dlib
  nlohmann_json::nlohmann_json
  ${GMMAP_LIBRARY}
  OpenMP::OpenMP_CXX
  fmt::fmt
)

# ===========================================================================
# Executables
# ===========================================================================

add_executable(gmmslam_node src/gmmslam_node.cpp)
add_executable(d2d_registration_node src/d2d_registration_node.cpp)
add_executable(ext_odom_publisher_node src/ext_odom_publisher_node.cpp)
add_executable(solid_descriptor_benchmark src/testing/solid_descriptor_benchmark.cpp)

foreach(_target gmmslam_node d2d_registration_node ext_odom_publisher_node
                solid_descriptor_benchmark)
  target_link_libraries(${_target} gmmslam_lib)
  ament_target_dependencies(${_target}
    rclcpp
    sensor_msgs
    geometry_msgs
    nav_msgs
    visualization_msgs
    std_msgs
    tf2_msgs
    tf2_ros
  )
endforeach()

# ===========================================================================
# Install
# ===========================================================================

install(TARGETS
  gmmslam_lib
  gmmslam_node
  d2d_registration_node
  ext_odom_publisher_node
  solid_descriptor_benchmark
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(gmmslam_lib)
ament_export_dependencies(
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  std_msgs
  tf2_msgs
  tf2_ros
)
ament_package()
