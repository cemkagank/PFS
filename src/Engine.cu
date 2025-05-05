#include "Engine.hpp"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <raylib.h>
#include <stdio.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

float Engine::particle_radius = 0.2f;
float Engine::particle_color[4] = {0, 0, 1, 1};

Engine::Engine() {
    // Initialize device pointers to nullptr
    d_positions = nullptr;
    d_velocities = nullptr;
    d_densities = nullptr;
    d_forces = nullptr;
    d_pressures = nullptr;
    d_spatialLookup = nullptr;
    d_startIndices = nullptr;
    
    // Initialize simulation parameters
    simulation_size = 0;
    gravity = -9.81f;
    
    // Make container larger
    container = box{Vector3{10,10,10}, Vector3{-10,-10,-10}};
    
    // Smaller particles for better fluid resolution
    particle_radius = 0.15f;
    
    // More water-like blue color with transparency
    particle_color[0] = 0.2f;
    particle_color[1] = 0.5f;
    particle_color[2] = 0.9f;
    particle_color[3] = 0.7f;
    
    // Create a smaller unit sphere mesh (radius 1.0) with fewer subdivisions
    particleMesh = GenMeshSphere(1.0f, 8, 8);
    
    // Load material with adjusted properties
    mat = LoadMaterialDefault();
    mat.maps[MATERIAL_MAP_DIFFUSE].color = Color{
        (unsigned char)(particle_color[0] * 255),
        (unsigned char)(particle_color[1] * 255),
        (unsigned char)(particle_color[2] * 255),
        (unsigned char)(particle_color[3] * 255)
    };
    UploadMesh(&particleMesh, false);
}

Engine::~Engine() {
    // Free CUDA device memory
    if (d_positions) CUDA_CHECK(cudaFree(d_positions));
    if (d_velocities) CUDA_CHECK(cudaFree(d_velocities));
    if (d_densities) CUDA_CHECK(cudaFree(d_densities));
    if (d_pressures) CUDA_CHECK(cudaFree(d_pressures));
    if (d_spatialLookup) CUDA_CHECK(cudaFree(d_spatialLookup));
    if (d_startIndices) CUDA_CHECK(cudaFree(d_startIndices));
}

void Engine::Draw() {
    // Copy positions from device to host for rendering
    CUDA_CHECK(cudaMemcpy(positions.data(), d_positions, 
        simulation_size * sizeof(float3), cudaMemcpyDeviceToHost));

    // Debug print positions before rendering
    printf("\nPositions before rendering:\n");
    for (int i = 0; i < std::min(5, simulation_size); i++) {
        printf("Position %d: (%f, %f, %f)\n", i, positions[i].x, positions[i].y, positions[i].z);
    }

    for (int i = 0; i < simulation_size; i++) {
        DrawMesh(particleMesh, mat, transforms[i]);
    }
    container.Draw();
}

void Engine::SimulationStep() {
    const int blockSize = 128;
    const int numBlocks = (simulation_size + blockSize - 1) / blockSize;
    
    // Calculate shared memory size for density kernel (positions only)
    size_t densitySharedMemSize = blockSize * sizeof(float3);
    
    // Calculate shared memory size for pressure kernel (positions, velocities, and densities)
    size_t pressureSharedMemSize = blockSize * (2 * sizeof(float3) + sizeof(float));

    // Update spatial lookup
    updateSpatialLookupKernel<<<numBlocks, blockSize>>>(d_positions, d_spatialLookup, d_startIndices, simulation_size, SMOOTHING_RADIUS);
    cudaDeviceSynchronize();
    
    // Sort particles
    sortParticlesKernel<<<numBlocks, blockSize>>>(d_spatialLookup, d_startIndices, simulation_size);
    cudaDeviceSynchronize();
    
    // Calculate density with shared memory
    calculateDensityKernel<<<numBlocks, blockSize, densitySharedMemSize>>>(
        d_positions, d_densities, d_spatialLookup, d_startIndices, simulation_size, SMOOTHING_RADIUS
    );
    cudaDeviceSynchronize();
    
    // Calculate pressure forces with shared memory
    int pressureBlocks = (simulation_size + blockSize - 1) / blockSize;
    calculatePressureForceKernel<<<pressureBlocks, blockSize, pressureSharedMemSize>>>(
        d_positions, d_velocities, d_densities, d_pressures, d_spatialLookup, d_startIndices, simulation_size, SMOOTHING_RADIUS
    );
    cudaDeviceSynchronize();
    
    // Update positions
    float3 containerMin = make_float3(container.min.x, container.min.y, container.min.z);
    float3 containerMax = make_float3(container.max.x, container.max.y, container.max.z);
    
    updatePositionsKernel<<<numBlocks, blockSize>>>(
        d_positions, d_velocities, d_pressures, containerMin, containerMax, gravity, GetFrameTime(), simulation_size
    );
    cudaDeviceSynchronize();
    
    // Copy positions for rendering
    CUDA_CHECK(cudaMemcpy(positions.data(), d_positions, simulation_size * sizeof(float3), cudaMemcpyDeviceToHost));
    
    // Update transforms for rendering
    for (int i = 0; i < simulation_size; i++) {
        Matrix scale = MatrixScale(particle_radius, particle_radius, particle_radius);
        Matrix translation = MatrixTranslate(positions[i].x, positions[i].y, positions[i].z);
        transforms[i] = MatrixMultiply(scale, translation);
    }
}

void Engine::Reset() {
    // Free existing device memory
    if (d_positions) CUDA_CHECK(cudaFree(d_positions));
    if (d_velocities) CUDA_CHECK(cudaFree(d_velocities));
    if (d_densities) CUDA_CHECK(cudaFree(d_densities));
    if (d_pressures) CUDA_CHECK(cudaFree(d_pressures));
    if (d_spatialLookup) CUDA_CHECK(cudaFree(d_spatialLookup));
    if (d_startIndices) CUDA_CHECK(cudaFree(d_startIndices));

    // Clear host vectors
    positions.clear();
    velocities.clear();
    densities.clear();
    forces.clear();
    pressures.clear();
    transforms.clear();
    simulation_size = 0;
}

void Engine::Populate() {
    // Create particles in a grid pattern
    const float spacing = 0.3f;  // Reduced spacing to fit more particles
    const int gridSize = 22;     // 22^3 = 10648 particles (slightly more than 10K)
    simulation_size = gridSize * gridSize * gridSize;
    
    // Resize host vectors
    positions.resize(simulation_size);
    velocities.resize(simulation_size);
    densities.resize(simulation_size);
    pressures.resize(simulation_size);
    forces.resize(simulation_size);
    transforms.resize(simulation_size);

    // Initialize particles in a grid
    int particleIndex = 0;
    for (int x = -gridSize/2; x < gridSize/2; x++) {
        for (int y = -gridSize/2; y < gridSize/2; y++) {
            for (int z = -gridSize/2; z < gridSize/2; z++) {
                if (particleIndex >= simulation_size) break;
                
                // Add small random offset to prevent perfect grid
                float offsetX = (rand() % 100) / 1000.0f;
                float offsetY = (rand() % 100) / 1000.0f;
                float offsetZ = (rand() % 100) / 1000.0f;
                
                // Store position as Vector3
                positions[particleIndex] = Vector3{
                    x * spacing + offsetX,
                    y * spacing + offsetY,
                    z * spacing + offsetZ
                };
                
                // Store velocity as Vector3
                velocities[particleIndex] = Vector3{0, 0, 0};
                
                // Create transform matrix
                Matrix scale = MatrixScale(particle_radius, particle_radius, particle_radius);
                Matrix translation = MatrixTranslate(
                    positions[particleIndex].x,
                    positions[particleIndex].y,
                    positions[particleIndex].z
                );
                transforms[particleIndex] = MatrixMultiply(scale, translation);
                
                particleIndex++;
            }
        }
    }

    // Free any existing device memory
    if (d_positions) CUDA_CHECK(cudaFree(d_positions));
    if (d_velocities) CUDA_CHECK(cudaFree(d_velocities));
    if (d_densities) CUDA_CHECK(cudaFree(d_densities));
    if (d_pressures) CUDA_CHECK(cudaFree(d_pressures));
    if (d_forces) CUDA_CHECK(cudaFree(d_forces));
    if (d_spatialLookup) CUDA_CHECK(cudaFree(d_spatialLookup));
    if (d_startIndices) CUDA_CHECK(cudaFree(d_startIndices));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_positions, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_velocities, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_densities, simulation_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pressures, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_forces, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_spatialLookup, simulation_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_startIndices, simulation_size * sizeof(int)));

    // Convert Vector3 to float3 for device memory
    std::vector<float3> positions_float3(simulation_size);
    std::vector<float3> velocities_float3(simulation_size);
    for (int i = 0; i < simulation_size; i++) {
        positions_float3[i] = make_float3(positions[i].x, positions[i].y, positions[i].z);
        velocities_float3[i] = make_float3(velocities[i].x, velocities[i].y, velocities[i].z);
    }

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_positions, positions_float3.data(), 
        simulation_size * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velocities, velocities_float3.data(), 
        simulation_size * sizeof(float3), cudaMemcpyHostToDevice));

    // Initialize device memory
    CUDA_CHECK(cudaMemset(d_densities, 0, simulation_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pressures, 0, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_forces, 0, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_spatialLookup, 0, simulation_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_startIndices, 0, simulation_size * sizeof(int)));

    // Print debug info
    printf("Created %d particles\n", simulation_size);
    printf("First particle position: (%f, %f, %f)\n", 
        positions[0].x, positions[0].y, positions[0].z);
}

void Engine::SpawnParticlesAtCenter() {
    // Create a small cube of particles at the center
    const float spacing = particle_radius * 2.0f;
    const int size = 5; // 5x5x5 cube
    
    // Clear existing particles
    positions.clear();
    velocities.clear();
    transforms.clear();
    
    // Create new particles
    for (int i = -size/2; i <= size/2; i++) {
        for (int j = -size/2; j <= size/2; j++) {
            for (int k = -size/2; k <= size/2; k++) {
                Vector3 pos = Vector3{
                    static_cast<float>(i) * spacing,
                    static_cast<float>(j) * spacing,
                    static_cast<float>(k) * spacing
                };
                positions.push_back(pos);
                velocities.push_back(Vector3{0,0,0});
                Matrix scale = MatrixScale(particle_radius, particle_radius, particle_radius);
                Matrix translation = MatrixTranslate(pos.x, pos.y, pos.z);
                transforms.push_back(MatrixMultiply(scale, translation));
            }
        }
    }
    
    // Update simulation size and resize vectors
    simulation_size = positions.size();
    densities.resize(simulation_size);
    pressures.resize(simulation_size);
    forces.resize(simulation_size);
    velocities.resize(simulation_size);
    transforms.resize(simulation_size);

    // Free any existing device memory
    if (d_positions) CUDA_CHECK(cudaFree(d_positions));
    if (d_velocities) CUDA_CHECK(cudaFree(d_velocities));
    if (d_densities) CUDA_CHECK(cudaFree(d_densities));
    if (d_pressures) CUDA_CHECK(cudaFree(d_pressures));
    if (d_forces) CUDA_CHECK(cudaFree(d_forces));
    if (d_spatialLookup) CUDA_CHECK(cudaFree(d_spatialLookup));
    if (d_startIndices) CUDA_CHECK(cudaFree(d_startIndices));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_positions, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_velocities, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_densities, simulation_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pressures, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_forces, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_spatialLookup, simulation_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_startIndices, simulation_size * sizeof(int)));

    // Convert Vector3 to float3 and copy to device
    std::vector<float3> positions_float3(simulation_size);
    std::vector<float3> velocities_float3(simulation_size);
    for (int i = 0; i < simulation_size; i++) {
        positions_float3[i] = make_float3(positions[i].x, positions[i].y, positions[i].z);
        velocities_float3[i] = make_float3(velocities[i].x, velocities[i].y, velocities[i].z);
    }

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_positions, positions_float3.data(), 
        simulation_size * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velocities, velocities_float3.data(), 
        simulation_size * sizeof(float3), cudaMemcpyHostToDevice));

    // Initialize device memory
    CUDA_CHECK(cudaMemset(d_densities, 0, simulation_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pressures, 0, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_forces, 0, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_spatialLookup, 0, simulation_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_startIndices, 0, simulation_size * sizeof(int)));

    TraceLog(LOG_INFO, "Spawned %d particles at center", simulation_size);
} 