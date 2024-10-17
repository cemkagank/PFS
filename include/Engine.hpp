#ifndef ENGINE_HPP
#define ENGINE_HPP
#include <vector>
#include <raylib.h>
#include "Particle.hpp"


class Engine {

private:
    int simulation_size = 3000;
    float smoothing_radius = 50;
    float threshold = 0.8f;

    float gravity = 0.1;    
    float targetDensity = 1;
    float pressureMultiplier = 0.0001;

    std::vector<Particle> particles;
    std::vector<float> densities = std::vector<float>(simulation_size, 1);
    std::vector<Vector2> forces;
    std::vector<Vector2> pressures = std::vector<Vector2>(simulation_size, {0,0});
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
    Engine();
    void Draw();
    void Update();
    void SimulationStep();
    void Reset();

    void Populate();
    void ShowDensity();

    void DrawGradinet();


};

#endif // ENGINE_HPPs`




















