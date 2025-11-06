cmake_minimum_required(VERSION 3.10)
project(chemlab)

set(CMAKE_CXX_STANDARD 14)

# ==== Armadillo 路径 ====
set(ARMADILLO_INCLUDE_DIR "C:/softwares/armadillo-15.2.1/include")
set(ARMADILLO_LIB_DIR "C:/softwares/armadillo-15.2.1/examples/lib_win64")

include_directories(${ARMADILLO_INCLUDE_DIR})
link_directories(${ARMADILLO_LIB_DIR})

# ==== 可执行文件 ====
add_executable(chemlab
        C_tem/svd_test.cpp
        C_tem/spin_adiabatic_state.C
)

# ==== 链接 OpenBLAS ====
target_link_libraries(chemlab PUBLIC openblas)
