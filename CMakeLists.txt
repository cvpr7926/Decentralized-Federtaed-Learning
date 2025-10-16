cmake_minimum_required(VERSION 3.18)
project(federated_learning)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add executable
add_executable(federated_learning federated_learning.cpp)

# Link libraries
target_link_libraries(federated_learning "${TORCH_LIBRARIES}" pthread)

# Set C++ flags
set_property(TARGET federated_learning PROPERTY CXX_STANDARD 17)