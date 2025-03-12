#ifndef ENGINE_HPP
#define ENGINE_HPP
#include <vector>
#include <raylib.h>

typedef struct box {
    Vector3 max;
    Vector3 min;
    Vector3 boxSize = { max.x - min.x, max.y - min.y, max.z - min.z };
    void Draw(){
        DrawCubeWires(Vector3{ (min.x + max.x) / 2.0f, (min.y + max.y) / 2.0f, (min.z + max.z) / 2.0f }, boxSize.x, boxSize.y, boxSize.z, RED);
    }
} box;


class Engine {

private:
    box container;
    int simulation_size = 125;
    std::vector<Vector3> positions;
    std::vector<Vector3> velocities;
    std::vector<Vector3> forces;
    std::vector<Vector3> pressures;
    std::vector<Vector3> gradients;
    std::vector<float>   densities;
    std::vector<Matrix> transforms;

    Mesh particleMesh;
    Material mat;
    Shader shader;

    std::vector<int> start_indices;
    std::vector<std::pair<int, unsigned int>> spatial_lookup;

    float SmoothingKernel(float distance);
    float SmoothingKernelDerivative(float distance);
    float CalculateDensity(Vector3 point);
    float DensityToPressure(float density);
    float CalculateSharedPressure(float dens1, float dens2);   
    Vector3 CalculatePressureForce(Vector3 point);

    Color Interpolate(int index);

public:
    static float particle_radius;
     float targetDensity = 1.2f;           // Target density for water
     float pressureMultiplier = 0.000001f * 3;   // Pressure force multiplier
     float smoothing_radius = 2.0f;        // Particle interaction radius
    
    static float particle_color[4];
    float threshold = 0.8f;

    float gravity = -25.0f;    // Increased gravity for more realistic water behavior

    Engine();
    void Draw();
    void Update();
    void SimulationStep();
    void Reset();
    void Populate();
    void ResolveCollisions();
    void UpdateSpatialLookup();
    void ForEachPointinRadius(Vector3 point);
    std::pair<int, int > PositionToCellCoord(Vector3 point);
    unsigned int HashCell(int cellx, int celly);
    unsigned int GetKeyFromHash(unsigned int hash);
    unsigned int HashPosition(int x, int y);
    void SpawnParticlesAtCenter();
    

};

#endif // ENGINE_HPPs`