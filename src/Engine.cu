#include "Engine.hpp"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <raylib.h>
#include <rlgl.h>
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
    
    // Nice water blue color
    particle_color[0] = 0.2f;  // R
    particle_color[1] = 0.5f;  // G
    particle_color[2] = 0.9f;  // B
    particle_color[3] = 0.8f;  // A
    
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
    
    // Initialize skybox and floor
    skyboxLoaded = false;
    floorLoaded = false;
    cameraPosition = Vector3Zero();
    LoadSkybox();
    LoadFloor();
    
    // Basic OpenGL setup
    rlEnableDepthTest();
    rlEnableBackfaceCulling();
}

Engine::~Engine() {
    // Free CUDA device memory
    if (d_positions) CUDA_CHECK(cudaFree(d_positions));
    if (d_velocities) CUDA_CHECK(cudaFree(d_velocities));
    if (d_densities) CUDA_CHECK(cudaFree(d_densities));
    if (d_pressures) CUDA_CHECK(cudaFree(d_pressures));
    if (d_spatialLookup) CUDA_CHECK(cudaFree(d_spatialLookup));
    if (d_startIndices) CUDA_CHECK(cudaFree(d_startIndices));
    
    // Unload skybox and floor
    UnloadSkybox();
    UnloadFloor();
}

int Engine::GetSimulationSize() {
    return simulation_size;
}

void Engine::Draw() {
    // Draw skybox first
    DrawSkybox();
    
    // Draw floor
    DrawFloor();
    
    // Copy positions from device to host for rendering
    CUDA_CHECK(cudaMemcpy(positions.data(), d_positions, 
        simulation_size * sizeof(float3), cudaMemcpyDeviceToHost));

    for (int i = 0; i < simulation_size; i++) {
        DrawMesh(particleMesh, mat, transforms[i]);
    }
    container.Draw();
}

void Engine::SimulationStep() {

    if (simulation_size == 0 || d_positions == nullptr) return; // Prevent invalid kernel launches

    const int blockSize = 256;
    const int numBlocks = (simulation_size + blockSize - 1) / blockSize;
    
    // Update spatial lookup
    build_spatial_lookup_kernel<<<numBlocks, blockSize>>>(d_positions, d_spatialLookup, d_startIndices, simulation_size);
    update_spatial_lookup_kernel<<<numBlocks, blockSize>>>(d_positions, d_spatialLookup, d_startIndices, simulation_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch kernels 
    density_kernel<<<numBlocks, blockSize>>>(d_positions, d_densities, d_spatialLookup, d_startIndices, simulation_size);
    pressure_kernel<<<numBlocks, blockSize>>>(d_positions, d_velocities, d_densities, d_forces, 
                                            d_spatialLookup, d_startIndices, simulation_size, GetFrameTime());
    update_positions_kernel<<<numBlocks, blockSize>>>(d_positions, d_velocities, d_forces, simulation_size, GetFrameTime());

    CUDA_CHECK(cudaDeviceSynchronize());

    // retrieve positions from device
    CUDA_CHECK(cudaMemcpy(positions.data(), d_positions, 
        simulation_size * sizeof(float3), cudaMemcpyDeviceToHost));

    // update transforms
    for (int i = 0; i < simulation_size; i++) {
        Matrix scale = MatrixScale(particle_radius, particle_radius, particle_radius);
        Matrix translation = MatrixTranslate(positions[i].x, positions[i].y, positions[i].z);
        transforms[i] = MatrixMultiply(scale, translation);
    }
}

void Engine::Reset() {
    // Free existing device memory
    if (d_positions) {
        CUDA_CHECK(cudaFree(d_positions));
        d_positions = nullptr;
    }
    if (d_velocities) {
        CUDA_CHECK(cudaFree(d_velocities));
        d_velocities = nullptr;
    }
    if (d_densities) {
        CUDA_CHECK(cudaFree(d_densities));
        d_densities = nullptr;
    }
    if (d_pressures) {
        CUDA_CHECK(cudaFree(d_pressures));
        d_pressures = nullptr;
    }
    if (d_forces) {
        CUDA_CHECK(cudaFree(d_forces));
        d_forces = nullptr;
    }
    if (d_spatialLookup) {
        CUDA_CHECK(cudaFree(d_spatialLookup));
        d_spatialLookup = nullptr;
    }
    if (d_startIndices) {
        CUDA_CHECK(cudaFree(d_startIndices));
        d_startIndices = nullptr;
    }

    // Reset simulation size
    simulation_size = 0;

    // Clear host vectors
    positions.clear();
    velocities.clear();
    densities.clear();
    forces.clear();
    pressures.clear();
    transforms.clear();

    // Ensure CUDA device is synchronized after cleanup
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Engine::Populate(int n) {
    // Create particles in a grid pattern
    const float spacing = 0.2f;  // Reduced spacing to fit more particles
    const int gridSize = n; //  56^3 = 175616 particles
    simulation_size = gridSize * gridSize * gridSize;
    
    // Resize all host vectors first
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
                velocities[particleIndex] = Vector3{0.0f, 0.0f, 0.0f};
                
                // Initialize other properties
                densities[particleIndex] = 0.0f;
                pressures[particleIndex] = 0.0f;
                forces[particleIndex] = Vector3{0.0f, 0.0f, 0.0f};
                
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



    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_positions, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_velocities, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_densities, simulation_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pressures, simulation_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_forces, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_spatialLookup, simulation_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_startIndices, simulation_size * sizeof(int)));

    // Convert Vector3 to float3 for device memory
    std::vector<float3> positions_float3(simulation_size);
    std::vector<float3> velocities_float3(simulation_size);
    std::vector<float3> forces_float3(simulation_size);
    
    for (int i = 0; i < simulation_size; i++) {
        positions_float3[i] = make_float3(positions[i].x, positions[i]  .y, positions[i].z);
        velocities_float3[i] = make_float3(velocities[i].x, velocities[i].y, velocities[i].z);
        forces_float3[i] = make_float3(forces[i].x, forces[i].y, forces[i].z);
    }

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_positions, positions_float3.data(), simulation_size * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velocities, velocities_float3.data(), simulation_size * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_densities, densities.data(), simulation_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pressures, pressures.data(), simulation_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_forces, forces_float3.data(), simulation_size * sizeof(float3), cudaMemcpyHostToDevice));

   

}




void Engine::SpawnParticlesAtCenter() {
    // Create a small cube of particles at the center
    const float spacing = particle_radius * 2.0f;
    const int size = 5; // 5x5x5 cube
    
    // Calculate total number of particles
    int totalParticles = (size + 1) * (size + 1) * (size + 1);
    
    // Clear and resize all vectors first
    positions.clear();
    velocities.clear();
    densities.clear();
    pressures.clear();
    forces.clear();
    transforms.clear();
    
    positions.resize(totalParticles);
    velocities.resize(totalParticles);
    densities.resize(totalParticles);
    pressures.resize(totalParticles);
    forces.resize(totalParticles);
    transforms.resize(totalParticles);
    
    // Create new particles
    int particleIndex = 0;
    for (int i = -size/2; i <= size/2; i++) {
        for (int j = -size/2; j <= size/2; j++) {
            for (int k = -size/2; k <= size/2; k++) {
                Vector3 pos = Vector3{
                    static_cast<float>(i) * spacing,
                    static_cast<float>(j) * spacing,
                    static_cast<float>(k) * spacing
                };
                positions[particleIndex] = pos;
                velocities[particleIndex] = Vector3{0.0f, 0.0f, 0.0f};
                densities[particleIndex] = 0.0f;
                pressures[particleIndex] = 0.0f;
                forces[particleIndex] = Vector3{0.0f, 0.0f, 0.0f};
                
                Matrix scale = MatrixScale(particle_radius, particle_radius, particle_radius);
                Matrix translation = MatrixTranslate(pos.x, pos.y, pos.z);
                transforms[particleIndex] = MatrixMultiply(scale, translation);
                
                particleIndex++;
            }
        }
    }
    
    // Update simulation size
    simulation_size = totalParticles;

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
    CUDA_CHECK(cudaMalloc(&d_pressures, simulation_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_forces, simulation_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_spatialLookup, simulation_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_startIndices, simulation_size * sizeof(int)));

    // Convert Vector3 to float3 for device memory
    std::vector<float3> positions_float3(simulation_size);
    std::vector<float3> velocities_float3(simulation_size);
    std::vector<float3> forces_float3(simulation_size);
    
    for (int i = 0; i < simulation_size; i++) {
        positions_float3[i] = make_float3(positions[i].x, positions[i].y, positions[i].z);
        velocities_float3[i] = make_float3(velocities[i].x, velocities[i].y, velocities[i].z);
        forces_float3[i] = make_float3(0.0f, 0.0f, 0.0f);
    }

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_positions, positions_float3.data(), simulation_size * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velocities, velocities_float3.data(), simulation_size * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_densities, 0, simulation_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pressures, 0, simulation_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_forces, forces_float3.data(), simulation_size * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_spatialLookup, 0, simulation_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_startIndices, 0, simulation_size * sizeof(int)));

    TraceLog(LOG_INFO, "Spawned %d particles at center", simulation_size);
}

void Engine::LoadSkybox() {
    // Create a skybox texture with horizon effect
    const int size = 512;
    Image img = GenImageColor(size, size, BLANK);
    
    // Create a gradient from dark blue at top to light blue at bottom
    for (int y = 0; y < size; y++) {
        float t = (float)y / size;  // 0 at top, 1 at bottom
        
        // Create a horizon effect
        Color color;
        if (t < 0.5f) {
            // Sky gradient (top half)
            color = {
                (unsigned char)(20 + t * 40),     // R: 20-40
                (unsigned char)(40 + t * 80),     // G: 40-80
                (unsigned char)(100 + t * 155),   // B: 100-180
                255
            };
        } else {
            // Horizon gradient (bottom half)
            float horizonT = (t - 0.5f) * 2.0f;  // 0 to 1 in bottom half
            color = {
                (unsigned char)(40 + horizonT * 20),    // R: 40-60
                (unsigned char)(80 + horizonT * 40),    // G: 80-120
                (unsigned char)(180 + horizonT * 75),   // B: 180-255
                255
            };
        }
        
        // Draw horizontal line
        for (int x = 0; x < size; x++) {
            ImageDrawPixel(&img, x, y, color);
        }
    }
    
    // Convert to cubemap
    skyboxTexture = LoadTextureCubemap(img, CUBEMAP_LAYOUT_AUTO_DETECT);
    UnloadImage(img);
    skyboxLoaded = true;
    TraceLog(LOG_INFO, "Skybox texture created successfully");
}

void Engine::DrawSkybox() {
    if (skyboxLoaded) {
        // Clear the background with sky blue color
        ClearBackground(SKYBLUE);
    }
}

void Engine::UnloadSkybox() {
    if (skyboxLoaded) {
        UnloadTexture(skyboxTexture);
        skyboxLoaded = false;
    }
}

void Engine::DrawFloor() {
    if (floorLoaded) {
        // Draw the floor using stored mesh and material
        float floorY = container.min.y;  // Place it at the bottom of the container
        DrawMesh(floorMesh, floorMat, MatrixTranslate(0, floorY, 0));
    }
}

void Engine::LoadFloor() {
    // Create a checkerboard texture
    const int size = 512;
    const int tileSize = 32;  // Smaller tiles for more detail
    Image img = GenImageColor(size, size, WHITE);
    
    // Create checkerboard pattern with high contrast
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            bool isEvenTile = ((x / tileSize) + (y / tileSize)) % 2 == 0;
            Color color = isEvenTile ? WHITE : Color{40, 40, 40, 255};  // Dark gray instead of black
            ImageDrawPixel(&img, x, y, color);
        }
    }
    
    // Load the texture
    floorTexture = LoadTextureFromImage(img);
    SetTextureFilter(floorTexture, TEXTURE_FILTER_BILINEAR);  // Enable texture filtering
    SetTextureWrap(floorTexture, TEXTURE_WRAP_REPEAT);  // Enable texture repeating
    UnloadImage(img);
    
    // Create floor mesh - make it much larger than the container
    float floorSize = 50.0f;  // Increased from 20.0f to 50.0f
    floorMesh = GenMeshPlane(floorSize, floorSize, 1, 1);
    UploadMesh(&floorMesh, false);
    
    // Create floor material
    floorMat = LoadMaterialDefault();
    floorMat.maps[MATERIAL_MAP_DIFFUSE].texture = floorTexture;
    floorMat.maps[MATERIAL_MAP_DIFFUSE].color = WHITE;  // Full brightness
    
    floorLoaded = true;
    TraceLog(LOG_INFO, "Floor texture created successfully");
}

void Engine::UnloadFloor() {
    if (floorLoaded) {
        UnloadTexture(floorTexture);
        UnloadMesh(floorMesh);
        UnloadMaterial(floorMat);
        floorLoaded = false;
    }
} 