# 设定源文件文件夹
aux_source_directory(lib/ LCGLIB_SRC)

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
set_target_properties(lcg PROPERTIES VERSION 1.6.0 SOVERSION 1.6)
# 设置库文件的输出地址
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -O2")

# 添加编译选项
option(LCG_FABS "use lcg's fabs() function" ON)
option(LCG_OPENMP "use openmp" ON)
message(STATUS "Use library's fabs(): " ${LCG_FABS})
message(STATUS "Use OpenMP: " ${LCG_OPENMP})
# 加入一个头文件配置，让cmake对源码进行操作
configure_file(
	"${PROJECT_SOURCE_DIR}/config.h.in"
	"${PROJECT_SOURCE_DIR}/src/lib/config.h"
	)

if(LCG_OPENMP)
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

# 库的安装命令
if(WIN32)
	install(TARGETS lcg DESTINATION lib)
	install(TARGETS lcg_static DESTINATION lib)
else()
	install(TARGETS lcg lcg_static
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)
endif()
# 头文件安装命令
install(FILES lib/lcg.h DESTINATION include)
install(FILES lib/lcg_cxx.h DESTINATION include)
install(FILES lib/clcg.h DESTINATION include)
install(FILES lib/clcg_cxx.h DESTINATION include)

# 以下部分为例子程序的编译
# 设置可执行文件的输出地址
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# 例子的编译方法
macro(add_sample name file)
	# 添加可执行文件 命令行
	add_executable(${name} sample/${file}.cpp)
	# 为安装文件添加动态库的搜索地址 在Windows下并没有什么用 直接忽略
	set_target_properties(${name} PROPERTIES INSTALL_RPATH "/usr/local/lib")
	# 链接动态库
	target_link_libraries(${name} PUBLIC lcg)
endmacro()

add_sample(lcg_sample1 sample1)
add_sample(lcg_sample2 sample2)
add_sample(lcg_sample3 sample3)
add_sample(lcg_sample4 sample4)
