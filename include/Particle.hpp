#ifndef PARTICLE_HPP
#define PARTICLE_HPP
#include <raylib.h>

class Particle {
private:
    Vector2 position;
    Vector2 velocity;


public:
    Particle(Vector2 pos);
    Particle(float x, float y);
    Vector2 get_position() const;
    void Draw();
    void Update();
    void ApplyForce(Vector2 force);
    void SetPosition(Vector2 pos);
};

#endif // PARTICLE_HPP