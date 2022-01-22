cmake_minimum_required(VERSION 3.8)

include_directories(../library)
add_executable(app main.cpp)
target_link_libraries(app library)