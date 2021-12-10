cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Bub)

#before using cmake make sure to install libtorch, Eigen and boost::odeint
#update the include directories (lines: 24-25)
#then follow (from this directory in cmd):
#md build
#cd build
#cmake -DCMAKE_PREFIX_PATH=path\to\libtorch;path\to\eigen ..

#build from cmd: cmake --build . --config Release
#BUT recommended to use Visual Studio for building
#Open the project in VS and set the DEM project as Start Project

#Read more in the Readme

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#UPDATE these
include_directories(../boost_1_75_0)
include_directories(../eigen-3.3.9)

add_executable(Bub bub_hybrid.cpp)
target_link_libraries(Bub "${TORCH_LIBRARIES}")
set_property(TARGET Bub PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET Bub
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:Bub>)
endif (MSVC)