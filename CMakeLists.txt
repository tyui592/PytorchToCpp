cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch_to_cpp)

# torch
find_package(Torch REQUIRED)

# opencv
find_package(OpenCV REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET main PROPERTY CXX_STANDARD 14)
