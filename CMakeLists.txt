cmake_minimum_required(VERSION 2.8)
project(online-machine-learning)

set(LIBRARY_OUTPUT_PATH lib/${CMAKE_BUILD_TYPE})


link_directories(/usr/lib/x86_64-linux-gnu/ /usr/lib/)
include_directories(src)
include_directories(/usr/include/eigen2 /usr/include/)

file(GLOB src_files src/*)

add_library(oml_rf STATIC
src/data.cpp
src/utilities.cpp
src/hyperparameters.cpp
src/classifier.cpp
src/online_rf.cpp)
target_link_libraries(oml_rf config++)

add_library(oml_mcboost STATIC
src/data.cpp
src/utilities.cpp
src/hyperparameters.cpp
src/classifier.cpp
src/booster.cpp
src/online_mcboost.cpp)
target_link_libraries(oml_mcboost config++)


add_library(oml_mclpboost STATIC
src/data.cpp
src/utilities.cpp
src/hyperparameters.cpp
src/classifier.cpp
src/booster.cpp
src/online_mclpboost.cpp)
target_link_libraries(oml_mclpboost config++)


set(ALL_LIB oml_rf oml_mcboost oml_mclpboost)


install(DIRECTORY src/
  DESTINATION /usr/local/include/${PROJECT_NAME}
  PATTERN "*.cpp" EXCLUDE
  PATTERN "*.o" EXCLUDE
  PATTERN "linear_larank/*" EXCLUDE
  PATTERN "linear_larank" EXCLUDE)

install(TARGETS ${ALL_LIB}
    ARCHIVE DESTINATION /usr/local/lib
    LIBRARY DESTINATION /usr/local/lib)

#add_executable(OMCBoost src/OMCBoost.cpp)
#target_link_libraries(OMCBoost ${ALL_LIB} lconfig++)


#add_library(online-machine-learning ${src_files})
#target_link_libraries(online-multiclass-lpboost oml_booster oml_classifier oml_data oml_experimenter oml_hyperparameters oml_linear_larank oml_mcboost oml_mclpboost oml_rf oml_utilities lconfig++)

