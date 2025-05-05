#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <vector>
#include <raylib.h>
#include <raymath.h>
#include "box.hpp"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

class Engine {
public:
    static float particle_radius;
    static float particle_color[4];
    
    Engine();
    ~Engine();
    
    void Draw();
    void SimulationStep();
    void Update() { SimulationStep(); } // Alias for backward compatibility
    void Reset();
    void Populate();
    void SpawnParticlesAtCenter();
    
private:
    Color Interpolate(int index);
    
    // Host data
    std::vector<Vector3> positions;
    std::vector<Vector3> velocities;
    std::vector<float> densities;
    std::vector<Vector3> forces;
    std::vector<Vector3> pressures;
    std::vector<Matrix> transforms;
    int simulation_size;
    
    // Device data
    float3* d_positions;
    float3* d_velocities;
    float* d_densities;
    float3* d_forces;
    float3* d_pressures;
    int* d_spatialLookup;
    int* d_startIndices;
    
    // Simulation parameters
    box container;
    float gravity;
    
    // Rendering data
    Mesh particleMesh;
    Material mat;
};

#endif // ENGINE_HPP 