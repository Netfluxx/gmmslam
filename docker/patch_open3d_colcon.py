import os

gira_ws = os.environ.get("GIRA_WS", "/root/gira_ws")
path = os.path.join(
    gira_ws, "gira3d-reconstruction/dry/src/open3d_colcon/CMakeLists.txt"
)

with open(path) as f:
    lines = f.readlines()

insert = (
    '\noption(OPEN3D_CUDA_ARCHS "CUDA architectures to target (e.g., 75;86)" "89")\n'
    "if(BUILD_CUDA_MODULE)\n"
    '    set(CMAKE_CUDA_ARCHITECTURES "${OPEN3D_CUDA_ARCHS}" CACHE STRING "CUDA architectures" FORCE)\n'
    '    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")\n'
    "endif()\n"
)

lines.insert(27, insert)

with open(path, "w") as f:
    f.writelines(lines)

print("Patch 1 applied successfully.")
