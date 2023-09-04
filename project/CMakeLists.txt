cmake_minimum_required(VERSION 3.10)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# Add project name OPENCV_OPENVINO_TEST
project(FINAL_PROJECT)

# Setup CMake version
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(ENABLE_STATIC "Build static (.a) library" ON)

# OpenVINO required
find_package(OpenVINO REQUIRED)

# OpenCV required
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories( /opt/intel/openvino/samples/cpp/common/utils/include/samples )

# c++ npy required
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

# 找到 package 會根據後續的 CMAKE_INSTALL_PREFIX 的資料夾
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xsimd REQUIRED)

include_directories(/opt/xtensor/include/)
include_directories(/opt/xtl/include/)
include_directories(/opt/xsimd/include/)

# jsoncpp required
# find_package(jsoncpp CONFIG REQUIRED)
# include_directories(${JSON_INC_PATH})

# add_library(cnpy SHARED "/workspace/cnpy.cpp")
# target_link_libraries(cnpy ${ZLIB_LIBRARIES})
# install(TARGETS "cnpy" LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# if(ENABLE_STATIC)
#     add_library(cnpy-static STATIC "/workspace/cnpy.cpp")
#     set_target_properties(cnpy-static PROPERTIES OUTPUT_NAME "cnpy")
#     install(TARGETS "cnpy-static" ARCHIVE DESTINATION lib)
# endif(ENABLE_STATIC)

# install(FILES "cnpy.h" DESTINATION include)
# install(FILES "mat2npz" "npy2mat" "npz2mat" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)


# add_executable(main src/main.cpp)
# add_executable(drawpic src/drawpic.cpp)
# add_executable(objdraw0 src/obj-draw-case0.cpp)
# add_executable(objdraw1 src/obj-draw-case1.cpp)
# add_executable(objdraw2 src/obj-case2.cpp)
# add_executable(tryjson src/tryjson.cpp)
# add_executable(example1 src/example1.cpp)
# add_executable(first_example src/first.cpp)
# add_executable(loadNPY src/loadnpy.cpp)
add_executable(pp src/yolo-postprocess.cpp)

if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

# target_link_libraries( main ${OpenCV_LIBS} )
# target_link_libraries( drawpic ${OpenCV_LIBS} )
# target_link_libraries( objdraw0 ${OpenCV_LIBS} )
# target_link_libraries( objdraw1 ${OpenCV_LIBS} )
# target_link_libraries( objdraw2 ${OpenCV_LIBS} )
target_link_libraries( pp ${OpenCV_LIBS} )
# target_link_libraries( tryjson ${OpenCV_LIBS} )

# target_link_libraries(main openvino::runtime)
# target_link_libraries(drawpic openvino::runtime)
# target_link_libraries(objdraw0 openvino::runtime)
# target_link_libraries(objdraw1 openvino::runtime)
# target_link_libraries(objdraw2 openvino::runtime)
target_link_libraries(pp openvino::runtime)
# target_link_libraries(tryjson openvino::runtime)

# target_link_libraries(main jsoncpp_lib)
# target_link_libraries(drawpic jsoncpp_lib)
# target_link_libraries(objdraw0 jsoncpp_lib)
# target_link_libraries(objdraw1 jsoncpp_lib)
# target_link_libraries(objdraw2 jsoncpp_lib)
# target_link_libraries(tryjson jsoncpp_lib)
# target_link_libraries(pp jsoncpp_lib)

# target_link_libraries(example1 cnpy)

# 連結函式庫
# target_link_libraries(first_example xtensor xtensor::optimize )
target_link_libraries(pp xtensor xtensor::optimize )