cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(Fluids VERSION 1.0 LANGUAGES CXX CUDA)

# Specify C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Change this based on your GPU

# Set optimization level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wno-unknown-pragmas")

# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Define include directories
include_directories(include)
include_directories(src/imgui)
include_directories(src/rlimgui)

# Specify source files (add CUDA files)
set(SOURCES
    src/main.cpp
    src/Window.cpp
    src/Engine.cu
    src/UI.cpp
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui_tables.cpp
    src/imgui/imgui.cpp
    src/rlimgui/rlImGui.cpp
    src/kernels.cu
)

# Create an executable
add_executable(fluids ${SOURCES})

# Set CUDA properties
set_target_properties(fluids PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75 -Xcompiler -O3 -Xcompiler -march=native -Xcompiler -mtune=native -Xcompiler -ffast-math -Xcompiler -fno-strict-aliasing")

# Link raylib and other necessary libraries
find_library(RAYLIB_LIB raylib HINTS /usr/local/lib)
target_link_libraries(fluids PUBLIC ${RAYLIB_LIB} -Wl,--no-as-needed pthread dl rt)

# Enable compiler warnings
if(MSVC)
    target_compile_options(fluids PRIVATE /W4)
else()
    target_compile_options(fluids PRIVATE -Wall -Wextra -pedantic)
endif()