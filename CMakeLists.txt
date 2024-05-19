cmake_minimum_required(VERSION 3.15)
project(DIP)

# Добавляем файлы
file(GLOB_RECURSE sources
./Source/Top.cpp
./Source/*.hpp
)

add_executable(${PROJECT_NAME} ${sources})

# OpenCV
set(OpenCV_DIR C:/work/libraries/opencv/build)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV include directories:" ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})