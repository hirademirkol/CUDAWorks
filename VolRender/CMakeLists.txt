cmake_minimum_required(VERSION 3.24)

add_executable(VolumePrep volumePrep.cpp)

add_executable(VolRender volRender.cpp volRender_kernel.cu)

set_target_properties(VolRender PROPERTIES CUDA_ARCHITECTURES "60")
target_link_libraries(VolRender GLUT64 GLEW64)

add_custom_command(TARGET VolRender POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:VolRender> $<TARGET_FILE_DIR:VolRender>
  COMMAND_EXPAND_LISTS
  )