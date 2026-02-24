# GIRA3D colcon install prefixes
list(APPEND CMAKE_PREFIX_PATH
  /root/gira_ws/gira3d-reconstruction/wet/install
  /root/gira_ws/gira3d-registration/wet/install
  /root/gira_ws/gira3d-registration/dry/install
)

find_package(self_organizing_gmm REQUIRED)  # → sogmm_INCLUDE_DIR
find_package(gmm REQUIRED)                  # → gmm_INCLUDE_DIR
find_package(gmm_d2d_registration REQUIRED) # → gmm_d2d_registration_INCLUDE_DIR
find_package(dlib REQUIRED)                 # → dlib::dlib

target_include_directories(gslam PUBLIC
  ...
    ${sogmm_INCLUDE_DIR}
    ${gmm_d2d_registration_INCLUDE_DIR}
    ${gmm_INCLUDE_DIR}
)

# dlib is a link-time dep for the optimizer inside d2d registration:
target_link_libraries(gslam
  ...
  dlib::dlib
)