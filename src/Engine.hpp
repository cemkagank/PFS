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
    
    // Skybox methods
    void LoadSkybox();
    void DrawSkybox();
    void UnloadSkybox();
    void SetCameraPosition(Vector3 pos) { cameraPosition = pos; }
    
private:
    Color Interpolate(int index);
    
    // Host data
    std::vector<Vector3> positions;
    std::vector<Vector3> velocities;
    std::vector<float> densities;
    std::vector<Vector3> forces;
    std::vector<float> pressures;
    std::vector<Matrix> transforms;
    int simulation_size;
    
    // Device data
    float3* d_positions;
    float3* d_velocities;
    float* d_densities;
    float3* d_forces;
    float* d_pressures;
    int* d_spatialLookup;
    int* d_startIndices;
    
    // Simulation parameters
    box container;
    float gravity;
    
    // Rendering data
    Mesh particleMesh;
    Material mat;
    
    // Skybox
    Texture2D skyboxTexture;
    bool skyboxLoaded;
    Vector3 cameraPosition;
    
    // Floor
    Texture2D floorTexture;
    Mesh floorMesh;
    Material floorMat;
    bool floorLoaded;
    void LoadFloor();
    void DrawFloor();
    void UnloadFloor();
};

#endif // ENGINE_HPP 