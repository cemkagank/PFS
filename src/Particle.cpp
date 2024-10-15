#include "Particle.hpp"
#include <raymath.h>

Particle::Particle(Vector2 pos) {
    this->position = pos;
}

Particle::Particle(float x , float y) {
    this->position.x = x;
    this->position.y = y;

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

void Particle::Draw() {
    DrawCircleV(position, 4 , BLUE);
}