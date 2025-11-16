cmake_minimum_required(VERSION 3.24)

project(CUDAWorks)
enable_language("CUDA")

include(FetchContent)

FetchContent_Declare(
    freeglut
    GIT_REPOSITORY https://github.com/freeglut/freeglut
    GIT_TAG v3.8.0-rc1
)
FetchContent_Declare(
    glew
    URL https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.tgz
)
FetchContent_MakeAvailable(freeglut glew)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
add_subdirectory(
    ${glew_SOURCE_DIR}/build/cmake
    ${glew_BINARY_DIR}
)

include_directories(inc)
include_directories(${glew_SOURCE_DIR}/include ${freeglut_SOURCE_DIR}/include)

add_subdirectory(VolRender)
add_subdirectory(Boids)