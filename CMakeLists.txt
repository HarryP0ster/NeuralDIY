cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(NeuralDIY)

set(SOURCEFILES "source/NeuralDIY.cpp")

add_executable(NeuralNetwork ${SOURCEFILES})

set_target_properties(NeuralNetwork PROPERTIES  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set_target_properties(NeuralNetwork PROPERTIES  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin)