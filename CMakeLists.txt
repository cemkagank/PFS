cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(Fluids VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set optimization level (optional)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O2")

# Define the build type (you can specify Debug, Release, etc.)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Define include directories
include_directories(include)
include_directories(src/imgui)        # Add the path to ImGui headers
include_directories(src/rlimgui)     # Add the path to rlImGui headers

# Specify source files
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

# Create an executable from the source files
add_executable(fluids ${SOURCES})

# Specify the directory for output binaries
set_target_properties(fluids PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Link raylib and any other necessary libraries
find_library(RAYLIB_LIB raylib HINTS /usr/local/lib)
target_link_libraries(fluids PUBLIC ${RAYLIB_LIB} GL m pthread dl rt X11)

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_compile_options(fluids PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(fluids PUBLIC OpenMP::OpenMP_CXX)  # Link OpenMP
endif()

# Enable compiler warnings
if(MSVC)
    target_compile_options(fluids PRIVATE /W4) # For MSVC, warning level 4
else()
    target_compile_options(fluids PRIVATE -Wall -Wextra -pedantic)
endif()

