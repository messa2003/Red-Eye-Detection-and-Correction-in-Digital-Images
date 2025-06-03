cmake_minimum_required(VERSION 3.30)
project(Proiect_PI)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Proiect_PI main.cpp
        diverse.h
        functii.cpp
        ai_algorithm.cpp)

target_link_libraries(Proiect_PI ${OpenCV_LIBS})

set(OpenCV_DIR "C:/opencv/build")