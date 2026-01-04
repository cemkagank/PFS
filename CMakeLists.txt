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
# On macOS, OpenMP is typically installed via Homebrew as libomp
if(APPLE)
    message(STATUS "Searching for OpenMP (libomp) on macOS...")
    
    # Try to find libomp in common Homebrew locations
    # Homebrew installs packages in opt directories: /opt/homebrew/opt/libomp (Apple Silicon) or /usr/local/opt/libomp (Intel)
    find_library(LIBOMP_LIB 
        NAMES omp libomp
        PATHS 
            /opt/homebrew/opt/libomp/lib
            /usr/local/opt/libomp/lib
            /opt/homebrew/lib
            /usr/local/lib
            /opt/local/lib
    )
    
    if(LIBOMP_LIB)
        message(STATUS "Found libomp library: ${LIBOMP_LIB}")
        
        # If we found libomp manually, set up OpenMP variables
        # Homebrew installs headers in opt directories
        find_path(OPENMP_INCLUDE_DIR
            NAMES omp.h
            PATHS
                /opt/homebrew/opt/libomp/include
                /usr/local/opt/libomp/include
                /opt/homebrew/include
                /usr/local/include
                /opt/local/include
                /usr/local/include/libomp
                /opt/homebrew/include/libomp
        )
        
        if(OPENMP_INCLUDE_DIR)
            set(OpenMP_CXX_FOUND TRUE)
            # Separate flags for compile options (not a single string)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor" "-fopenmp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY ${LIBOMP_LIB})
            message(STATUS "Found OpenMP via libomp:")
            message(STATUS "  Library: ${LIBOMP_LIB}")
            message(STATUS "  Headers: ${OPENMP_INCLUDE_DIR}")
        else()
            message(FATAL_ERROR 
                "OpenMP headers (omp.h) not found.\n"
                "Library found at: ${LIBOMP_LIB}\n"
                "Please ensure libomp is fully installed: brew install libomp\n"
                "Or check if headers are in a non-standard location.")
        endif()
    else()
        # Fall back to standard find_package
        message(STATUS "libomp not found in standard locations, trying find_package(OpenMP)...")
        find_package(OpenMP)
        if(NOT OpenMP_CXX_FOUND)
            message(FATAL_ERROR 
                "OpenMP not found on macOS.\n"
                "Please install it via Homebrew:\n"
                "  brew install libomp\n"
                "After installation, clean your build directory and reconfigure:\n"
                "  rm -rf build && mkdir build && cd build && cmake ..")
        endif()
    endif()
else()
    # For Linux and other platforms, use standard find_package
    find_package(OpenMP REQUIRED)
endif()

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

# Add OpenMP include directory on macOS if found manually
if(APPLE AND OPENMP_INCLUDE_DIR)
    target_include_directories(fluids PRIVATE ${OPENMP_INCLUDE_DIR})
endif()

# Find and link raylib
if(APPLE)
    # Check both Intel and Apple Silicon Homebrew locations (including opt directories)
    find_library(RAYLIB_LIB raylib 
        PATHS 
            /opt/homebrew/opt/raylib/lib
            /usr/local/opt/raylib/lib
            /opt/homebrew/lib
            /usr/local/lib
            /opt/local/lib
    )
    
    # Find raylib include directory
    find_path(RAYLIB_INCLUDE_DIR
        NAMES raylib.h
        PATHS
            /opt/homebrew/opt/raylib/include
            /usr/local/opt/raylib/include
            /opt/homebrew/Cellar/raylib/*/include
            /usr/local/Cellar/raylib/*/include
            /opt/homebrew/include
            /usr/local/include
    )
else()
    find_library(RAYLIB_LIB raylib HINTS /usr/local/lib)
    find_path(RAYLIB_INCLUDE_DIR NAMES raylib.h HINTS /usr/local/include)
endif()

if(NOT RAYLIB_LIB)
    message(FATAL_ERROR "raylib library not found")
endif()

if(NOT RAYLIB_INCLUDE_DIR)
    message(FATAL_ERROR "raylib headers not found. Please install raylib: brew install raylib")
endif()

# Add raylib include directory
target_include_directories(fluids PRIVATE ${RAYLIB_INCLUDE_DIR})

# Link libraries
if(APPLE)
    if(OpenMP_omp_LIBRARY)
        # On macOS with manually found libomp, link directly
        target_link_libraries(fluids PRIVATE
            ${RAYLIB_LIB}
            ${OpenMP_omp_LIBRARY}
            "-framework OpenGL"
            "-framework Cocoa"
            "-framework IOKit"
            "-framework CoreVideo"
            m
            pthread
            dl
        )
        # Add OpenMP compiler flags
        target_compile_options(fluids PRIVATE ${OpenMP_CXX_FLAGS})
    else()
        # Use standard OpenMP target
        target_link_libraries(fluids PRIVATE
            ${RAYLIB_LIB}
            OpenMP::OpenMP_CXX
            "-framework OpenGL"
            "-framework Cocoa"
            "-framework IOKit"
            "-framework CoreVideo"
            m
            pthread
            dl
        )
    endif()
else()
    # Use standard OpenMP target for other platforms
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
endif()

# Print configuration summary
message(STATUS "Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP found: ${OpenMP_CXX_FOUND}")
message(STATUS "  Output directory: ${CMAKE_BINARY_DIR}/bin")

