cmake_minimum_required(VERSION 2.8)
project(online_rf)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

include_directories(/usr/local/include/
                    /usr/include/eigen2
                    /usr/local/include/SFML
)


file(GLOB_RECURSE src_files *)

add_executable(test1 test1.cpp)
target_link_libraries(test1 oml_rf oml_experimenter config++ sfml-window sfml-system sfml-graphics)
