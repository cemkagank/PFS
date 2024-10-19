#ifndef PARTICLE_HPP
#define PARTICLE_HPP
#include <raylib.h>

class Particle {
private:
    Vector2 position;
    Vector2 velocity;
    Color convert_to_color(float r, float g, float b);

public:
    static float color[4];
    static float radius;
    Particle(Vector2 pos);
    Particle(float x, float y);
    Vector2 get_position() const;
    void Draw();
    void Update();
    void ApplyForce(Vector2 force);
    void SetVelocity(Vector2 vel);
    void SetPosition(Vector2 pos);
};

#endif // PARTICLE_HPP