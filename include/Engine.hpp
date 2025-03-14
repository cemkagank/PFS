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
    std::vector<Matrix> transforms;

    Mesh particleMesh;
    Material mat;
    Shader shader;


public:
    static float particle_radius;
    float targetDensity = 1.2f;           // Target density for water
    float pressureMultiplier = 0.000001f * 3;   // Pressure force multiplier
    float smoothing_radius = 2.0f;        // Particle interaction radius
    
    static float particle_color[4];
    float threshold = 0.8f;

    float gravity = -25.0f;    // Increased gravity for more realistic water behavior

    Engine();
    ~Engine();
    void Draw();
    void Update();
    void Reset();
    void Populate();
    

};

#endif // ENGINE_HPPs`