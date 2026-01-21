# ARM 平台支持 - CMake 配置
# 用于 Raspberry Pi 等 ARM 设备
#
# 使用方法:
#   mkdir build && cd build
#   cmake .. -DCMAKE_TOOLCHAIN_FILE=../arm_toolchain.cmake
#   make

cmake_minimum_required(VERSION 3.16)
project(IMC22_ARM VERSION 2.1.0 LANGUAGES C CXX)

# 编译选项
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ARM 优化选项
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=cortex-a72 -mfpu=neon-fp-armv8")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=cortex-a72 -mfpu=neon-fp-armv8")
endif()

# 定义平台宏
add_definitions(-DPLATFORM_ARM)
add_definitions(-DCIM_SOFTWARE_EMULATION)

# 源文件
set(SDK_SOURCES
    ../../imc22_cim.c
    ../../imc22_power.c
    ../../imc22_dvfs.c
    ../../tinyml_adaptive.c
    arm_cim_emulator.c
    arm_gpio_adapter.c
)

# 头文件目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 静态库
add_library(imc22_arm STATIC ${SDK_SOURCES})

# 共享库
add_library(imc22_arm_shared SHARED ${SDK_SOURCES})
set_target_properties(imc22_arm_shared PROPERTIES OUTPUT_NAME imc22)

# 示例程序
add_executable(arm_demo arm_demo.c)
target_link_libraries(arm_demo imc22_arm m pthread)

# 安装规则
install(TARGETS imc22_arm imc22_arm_shared
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(FILES 
        ../../imc22.h
        ../../imc22_cim.h
        ../../imc22_power.h
        ../../imc22_dvfs.h
        ../../tinyml_adaptive.h
        ../../nn_topology.h
        arm_platform.h
        DESTINATION include/imc22)

# 打印配置信息
message(STATUS "=== IMC-22 ARM 平台构建 ===")
message(STATUS "处理器: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "C 编译器: ${CMAKE_C_COMPILER}")
message(STATUS "C++ 编译器: ${CMAKE_CXX_COMPILER}")
