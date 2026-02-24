import os

gira_ws = os.environ.get("GIRA_WS", "/root/gira_ws")
path = os.path.join(
    gira_ws, "gira3d-reconstruction/dry/src/open3d_colcon/Open3D/CMakeLists.txt"
)

with open(path) as f:
    content = f.read()

old = (
    "include(Open3DMakeCudaArchitectures)\n"
    "open3d_make_cuda_architectures(CUDA_ARCHS)\n"
    "set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})"
)

new = (
    'if(NOT DEFINED OPEN3D_CUDA_ARCHS OR OPEN3D_CUDA_ARCHS STREQUAL "")\n'
    "    include(Open3DMakeCudaArchitectures)\n"
    "    open3d_make_cuda_architectures(CUDA_ARCHS)\n"
    '    set(OPEN3D_CUDA_ARCHS "${CUDA_ARCHS}")\n'
    "endif()\n"
    'if("${OPEN3D_CUDA_ARCHS}" MATCHES ".*86.*")\n'
    '    set(OPEN3D_CUDA_ARCHS "75" CACHE STRING "CUDA architectures override" FORCE)\n'
    '    message(WARNING "Overriding CUDA architectures to 75 to avoid unsupported compute_86.")\n'
    "endif()\n"
    "set(CMAKE_CUDA_ARCHITECTURES ${OPEN3D_CUDA_ARCHS})"
)

if old not in content:
    print("WARNING: target block not found, skipping patch 2.")
else:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("Patch 2 applied successfully.")
