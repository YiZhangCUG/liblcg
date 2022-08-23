# 设定源文件文件夹
aux_source_directory(lib LCGLIB_SRC)

if(NOT LibLCG_EIGEN)
	list(REMOVE_ITEM LCGLIB_SRC "lib/algebra_eigen.cpp")
	list(REMOVE_ITEM LCGLIB_SRC "lib/lcg_eigen.cpp")
	list(REMOVE_ITEM LCGLIB_SRC "lib/clcg_eigen.cpp")
	list(REMOVE_ITEM LCGLIB_SRC "lib/solver_eigen.cpp")
	list(REMOVE_ITEM LCGLIB_SRC "lib/preconditioner_eigen.cpp")
endif()

if(NOT LibLCG_CUDA)
	list(REMOVE_ITEM LCGLIB_SRC "lib/algebra_cuda.cu")
	list(REMOVE_ITEM LCGLIB_SRC "lib/lcg_complex_cuda.cu")
	list(REMOVE_ITEM LCGLIB_SRC "lib/lcg_cuda.cu")
	list(REMOVE_ITEM LCGLIB_SRC "lib/clcg_cuda.cu")
	list(REMOVE_ITEM LCGLIB_SRC "lib/clcg_cuda_f.cu")
	list(REMOVE_ITEM LCGLIB_SRC "lib/solver_cuda.cu")
	list(REMOVE_ITEM LCGLIB_SRC "lib/preconditioner_cuda.cu")
endif()

# 以下部分为库的编译
# 注意目标名必须唯一 所以不能直接生成相同名称的动态库与静态库
# 注意此处不必为目标名称添加lib前缀和相应后缀，cmake会自行添加
add_library(lcg SHARED ${LCGLIB_SRC})
# 首先添加静态库的生成命令
add_library(lcg_static STATIC ${LCGLIB_SRC})
# 设置静态库的输出名称从而获得与动态库名称相同的静态库
set_target_properties(lcg_static PROPERTIES OUTPUT_NAME "lcg")
# 设置输出目标属性以同时输出动态库与静态库
set_target_properties(lcg PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(lcg_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
# 设置动态库的版本号
set_target_properties(lcg PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
# 设置库文件的输出地址
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# 设置编译选项
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

if(LibLCG_EIGEN)
	find_package(Eigen3 REQUIRED)
	if(EIGEN3_FOUND)
		message(STATUS "Eigen3 Found.")
		include_directories(${EIGEN3_INCLUDE_DIR})
	endif()
endif()

if(LibLCG_CUDA)
	enable_language(CUDA)
	find_package(CUDA REQUIRED)
	if(CUDA_FOUND)
		message(STATUS "CUDA Found.")
		include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
		find_library(CUBLAS_LIBRARY cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		find_library(CUSPARSE_LIBRARY cusparse ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		target_link_libraries(lcg PUBLIC ${CUBLAS_LIBRARY})
		target_link_libraries(lcg_static ${CUBLAS_LIBRARY})
		target_link_libraries(lcg PUBLIC ${CUSPARSE_LIBRARY})
		target_link_libraries(lcg_static ${CUSPARSE_LIBRARY})
	endif()
endif()

if(LibLCG_OPENMP)
	# 添加openmp的编译命令 设置编译选项
	find_package(OpenMP REQUIRED)
	if (OpenMP_CXX_FOUND)
		message(STATUS "OpenMP Found.")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
		set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
		target_link_libraries(lcg PUBLIC OpenMP::OpenMP_CXX)
		target_link_libraries(lcg_static OpenMP::OpenMP_CXX)
	endif()
endif()

set(CONFIG_FILE_PATH lib/cmake/${PROJECT_NAME})

configure_package_config_file(${PROJECT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in 
	${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
	INSTALL_DESTINATION ${CONFIG_FILE_PATH}
	NO_CHECK_REQUIRED_COMPONENTS_MACRO)

write_basic_package_version_file(${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

# 库的安装命令
if(WIN32)
	install(TARGETS lcg DESTINATION lib)
	install(TARGETS lcg_static DESTINATION lib)
else()
	install(TARGETS lcg lcg_static
		EXPORT ${PROJECT_NAME}Targets
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib)
	install(EXPORT ${PROJECT_NAME}Targets
	        DESTINATION ${CONFIG_FILE_PATH})
	install(FILES
        ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${CONFIG_FILE_PATH})
endif()
# 头文件安装命令
install(FILES lib/config.h DESTINATION include/lcg)
install(FILES lib/algebra.h DESTINATION include/lcg)
install(FILES lib/lcg_complex.h DESTINATION include/lcg)
install(FILES lib/util.h DESTINATION include/lcg)
install(FILES lib/lcg.h DESTINATION include/lcg)
install(FILES lib/clcg.h DESTINATION include/lcg)
install(FILES lib/solver.h DESTINATION include/lcg)
install(FILES lib/preconditioner.h DESTINATION include/lcg)

if(LibLCG_CUDA)
	install(FILES lib/algebra_cuda.h DESTINATION include/lcg)
	install(FILES lib/lcg_complex_cuda.h DESTINATION include/lcg)
	install(FILES lib/lcg_cuda.h DESTINATION include/lcg)
	install(FILES lib/clcg_cuda.h DESTINATION include/lcg)
	install(FILES lib/clcg_cudaf.h DESTINATION include/lcg)
	install(FILES lib/solver_cuda.h DESTINATION include/lcg)
	install(FILES lib/preconditioner_cuda.h DESTINATION include/lcg)
endif()

if(LibLCG_EIGEN)
	install(FILES lib/algebra_eigen.h DESTINATION include/lcg)
	install(FILES lib/lcg_eigen.h DESTINATION include/lcg)
	install(FILES lib/clcg_eigen.h DESTINATION include/lcg)
	install(FILES lib/solver_eigen.h DESTINATION include/lcg)
	install(FILES lib/preconditioner_eigen.h DESTINATION include/lcg)
endif()

# 以下部分为例子程序的编译
# 设置可执行文件的输出地址
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# 例子的编译方法
macro(add_sample name file)
	# 添加可执行文件 命令行
	add_executable(${name} sample/${file})
	# 为安装文件添加动态库的搜索地址 在Windows下并没有什么用 直接忽略
	set_target_properties(${name} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
	# 链接动态库
	target_link_libraries(${name} PUBLIC lcg)
endmacro()

add_sample(lcg_sample1 sample1.cpp)
add_sample(lcg_sample2 sample2.cpp)
add_sample(lcg_sample3 sample3.cpp)
add_sample(lcg_sample4 sample4.cpp)

if(LibLCG_EIGEN)
	add_sample(lcg_sample5 sample5.cpp)
	add_sample(lcg_sample7 sample7.cpp)
	if(LibLCG_STD_COMPLEX)
		add_sample(lcg_sample6 sample6.cpp) 
	endif()
endif()

if(LibLCG_CUDA)
	add_sample(lcg_sample8 sample8.cu)
	add_sample(lcg_sample9 sample9.cu)
	add_sample(lcg_sample10 sample10.cu)
	add_sample(lcg_sample11 sample11.cu)
	add_sample(lcg_sample12 sample12.cu)
	add_sample(lcg_sample13 sample13.cu)
	add_sample(lcg_sample14 sample14.cu)
endif()
