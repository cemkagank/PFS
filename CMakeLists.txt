cmake_minimum_required(VERSION 3.10)

# Project configuration
project(Fluids 
    VERSION 1.0
    DESCRIPTION "Fluid Simulation Project"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Enable warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# Find required packages
find_package(OpenMP REQUIRED)

# Define source files
set(SOURCES
    src/main.cpp
    src/Window.cpp
    src/Engine.cpp
    src/UI.cpp
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui_tables.cpp
    src/imgui/imgui.cpp
    src/rlimgui/rlImGui.cpp
)

# Create executable target
add_executable(fluids ${SOURCES})

# Set output directory
set_target_properties(fluids PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Include directories
target_include_directories(fluids PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rlimgui
)

# Find and link raylib
find_library(RAYLIB_LIB raylib HINTS /usr/local/lib)
if(NOT RAYLIB_LIB)
    message(FATAL_ERROR "raylib library not found")
endif()

# Link libraries
target_link_libraries(fluids PRIVATE
    ${RAYLIB_LIB}
    OpenMP::OpenMP_CXX
    GL
    m
    pthread
    dl
    rt
    X11
)

# Print configuration summary
message(STATUS "Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP found: ${OpenMP_CXX_FOUND}")
message(STATUS "  Output directory: ${CMAKE_BINARY_DIR}/bin")

