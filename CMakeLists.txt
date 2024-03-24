cmake_minimum_required(VERSION 3.17)
project(DIP_Lab1)

# Указываем компилятор
set(CMAKE_CXX_STANDARD 11)

# Добавляем файлы
file(GLOB_RECURSE sources 
./Source/Top.cpp 
# ./Source/*.h 
./Source/Lab1.hpp
./Source/Lab2.hpp 
)

add_executable(${PROJECT_NAME} ${sources})

# OpenCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV include directories:" ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})