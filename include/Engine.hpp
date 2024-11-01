#ifndef ENGINE_HPP
#define ENGINE_HPP
#include <vector>
#include <raylib.h>
#include <functional>


class Engine {

private:

    int simulation_size = 1250;
    std::vector<Vector2> positions;
    std::vector<Vector2> velocities;
    std::vector<float> densities;
    std::vector<int> start_indices;
    std::vector<std::pair<int, unsigned int>> spatial_lookup;
    std::vector<Vector2> forces;
    std::vector<Vector2> pressures;
    std::vector<Vector2> gradients;

    float SmoothingKernel(float distance);
    float SmoothingKernelDerivative(float distance);
    float CalculateDensity(Vector2 point);
    float CalculateProperty(Vector2 point);
    float DensityToPressure(float density);
    float CalculateSharedPressure(float dens1, float dens2);   
    Vector2 CalculatePropertyGradient(Vector2 point);
    Vector2 CalculatePressureForce(Vector2 point);

public:
    static float particle_radius;
    static float particle_color[4];
    float smoothing_radius = 50;
    float threshold = 0.8f;

    float gravity = 0.1;    
    float targetDensity = 1;
    float pressureMultiplier = 0.0001;
    Engine();
    void Draw();
    void Update();
    void SimulationStep();
    void Reset();
    void Repopulate(Vector2 pos);
    void Populate();
    void ShowDensity();
    void ResolveCollisions();
    void UpdateSpatialLookup();
    void ForEachPointinRadius(Vector2 point);
    std::pair<int, int> PositionToCellCoord(Vector2 point);
    unsigned int HashCell(int cellx, int celly);
    unsigned int GetKeyFromHash(unsigned int hash);
    unsigned int HashPosition(int x, int y);
    

};

#endif // ENGINE_HPPs`




















