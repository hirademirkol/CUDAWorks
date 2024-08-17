cmake_minimum_required(VERSION 3.24)

project(CUDAWorks)
enable_language("CUDA")

find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories(${GLUT_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
include_directories(inc)

add_library(GLUT64 SHARED IMPORTED)
set_target_properties(GLUT64 PROPERTIES IMPORTED_LOCATION ${GLUT_INCLUDE_DIRS}/../bin/x64/freeglut.dll)
set_target_properties(GLUT64 PROPERTIES IMPORTED_IMPLIB ${GLUT_INCLUDE_DIRS}/../lib/x64/freeglut.lib)

add_library(GLEW64 SHARED IMPORTED)
set_target_properties(GLEW64 PROPERTIES IMPORTED_LOCATION ${GLEW_INCLUDE_DIRS}/../bin/Release/x64/glew32.dll)
set_target_properties(GLEW64 PROPERTIES IMPORTED_IMPLIB ${GLEW_INCLUDE_DIRS}/../lib/Release/x64/glew32.lib)

add_subdirectory(VolRender)
add_subdirectory(Boids)
add_subdirectory(FFT)