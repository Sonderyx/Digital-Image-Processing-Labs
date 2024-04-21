cmake_minimum_required(VERSION 3.15)
project(DIP_Lab1)

# Добавляем файлы
file(GLOB_RECURSE sources 
./Source/Top.cpp 
# ./Source/*.h 
./Source/Lab1.hpp
./Source/Lab2.hpp 
./Source/Lab3.hpp 
./Source/Lab4.hpp 
)

add_executable(${PROJECT_NAME} ${sources})

# OpenCV
set(OpenCV_DIR C:/work/libraries/opencv/build)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV include directories:" ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})