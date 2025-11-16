cmake_minimum_required(VERSION 3.24)

add_executable(Boids boids.cpp boidSystem.cpp renderer.cpp boidSystem_cuda.cu)

set_target_properties(Boids PROPERTIES CUDA_ARCHITECTURES 60)
target_link_libraries(Boids ${OPENGL_LIBRARIES} freeglut glew)

add_custom_command(TARGET Boids POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:Boids> $<TARGET_FILE_DIR:Boids>
  COMMAND_EXPAND_LISTS
)