cmake_minimum_required(VERSION 2.8)
project(online-machine-learning)

set(LIBRARY_OUTPUT_PATH lib/${CMAKE_BUILD_TYPE})

find_package(Config++ REQUIRED)

link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(src)
include_directories(/usr/include/eigen3)

file(GLOB src_files src/*)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

add_library(oml_rf STATIC src/data.cpp src/utilities.cpp src/hyperparameters.cpp src/classifier.cpp src/online_rf.cpp)
add_library(oml_experimenter STATIC src/data.cpp src/utilities.cpp src/classifier.cpp src/experimenter.cpp )
#add_library(oml_mcboost STATIC src/data.cpp src/utilities.cpp src/hyperparameters.cpp src/classifier.cpp src/booster.cpp src/online_mcboost.cpp )
#add_library(oml_larank STATIC src/data.cpp src/utilities.cpp src/hyperparameters.cpp src/classifier.cpp src/linear_larank/LaRank.cpp src/linear_larank/vectors.cpp src/linear_larank.cpp )
#add_library(oml_mclpboost STATIC src/data.cpp src/utilities.cpp src/hyperparameters.cpp src/classifier.cpp src/booster.cpp src/online_mclpboost.cpp )

set(ALL_LIB oml_rf oml_experimenter )
#oml_mcboost oml_mclpboost oml_larank)

install(DIRECTORY src/
  DESTINATION /usr/local/include/${PROJECT_NAME}
  PATTERN "*.cpp" EXCLUDE
  PATTERN "*.o" EXCLUDE
#  PATTERN "linear_larank/*" EXCLUDE
#  PATTERN "linear_larank" EXCLUDE
)

install(TARGETS ${ALL_LIB}
    ARCHIVE DESTINATION /usr/local/lib
    LIBRARY DESTINATION /usr/local/lib)

#add_executable(OMCBoost src/OMCBoost.cpp)
#target_link_libraries(OMCBoost ${ALL_LIB} config++)

