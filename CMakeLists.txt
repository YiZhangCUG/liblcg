cmake_minimum_required(VERSION 3.15.2)
# 设置工程名称
project(LibLCG VERSION 3.1 LANGUAGES CXX)
# 添加配置配件编写的函数
include(CMakePackageConfigHelpers)

message(STATUS "Platform: " ${CMAKE_HOST_SYSTEM_NAME})
# CMake默认的安装路径 Windows下为C:/Program\ Files/${Project_Name} Linux/Unix下为/usr/local
message(STATUS "Install prefix: " ${CMAKE_INSTALL_PREFIX})
# CMake默认的变异类型为空
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# 添加编译选项
option(LibLCG_OPENMP "Use OpenMP" ON) # Set OFF to disable the functionality 
option(LibLCG_EIGEN "Use Eigen" ON)
option(LibLCG_STD_COMPLEX "Use STD complex" ON)
option(LibLCG_CUDA "Use CUDA" ON)
message(STATUS "Use OpenMP: " ${LibLCG_OPENMP})
message(STATUS "Use Eigen: " ${LibLCG_EIGEN})
message(STATUS "Use STD complex: " ${LibLCG_STD_COMPLEX})
message(STATUS "Use CUDA: " ${LibLCG_CUDA})

# 加入一个头文件配置，让cmake对源码进行操作
configure_file(
	"${PROJECT_SOURCE_DIR}/config.h.in"
	"${PROJECT_SOURCE_DIR}/src/lib/config.h"
	)

# 添加源文件地址
add_subdirectory(src/)