#include "Particle.hpp"
#include <raymath.h>

float Particle::radius = 4;
float Particle::color[4] = {0, 0, 1, 1};

Particle::Particle(Vector2 pos) {
    this->position = pos;
    this->velocity = {0,0};
}

Particle::Particle(float x , float y) {
    this->position.x = x;
    this->position.y = y;
    this->velocity = {0,0};

}

void Particle::SetVelocity(Vector2 vel) {
    this->velocity = vel;
}

void Particle::ApplyForce(Vector2 force) {
    this->velocity = Vector2Add(this->velocity, force);
    // this->velocity = force;
}

void Particle::SetPosition(Vector2 pos) {
    this->position = pos;
}

void Particle::Update() {
    this->position = Vector2Add(this->position, this->velocity);
}

Vector2 Particle::get_position() const {
    return position;
}

Color Particle::convert_to_color(float r, float g, float b) {
    return {r * 255, g * 255, b * 255, 255};
}
void Particle::Draw() {
    DrawCircleV(this->position, radius, convert_to_color(color[0], color[1], color[2]));
}
