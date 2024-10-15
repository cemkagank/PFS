#include "Engine.hpp"
#include "time.h"
#include <cmath>
#include <iostream>
#include <raymath.h>
#include <omp.h>

Engine::Engine() {
    SetRandomSeed(time(NULL));
}

void Engine::Draw() {
    for (Particle p : particles) {
        p.Draw();
    }
}

void Engine::SimulationStep() {

    // Apply gravity 
    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        particles[i].ApplyForce({0,gravity * GetFrameTime()});
    }

    // Apply pressure
    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        particles[i].ApplyForce(pressures[i]);
    }

    // Update positions and resolve collisions
    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        particles[i].Update();
        if (particles[i].get_position().y > 900) {
            particles[i].SetPosition({particles[i].get_position().x, 894});
        }
        if (particles[i].get_position().y < 0) {
            particles[i].SetPosition({particles[i].get_position().x, 10});
        }
        if (particles[i].get_position().x > 1600) {
            particles[i].SetPosition({1594, particles[i].get_position().y});
        }
        if (particles[i].get_position().x < 0) {
            particles[i].SetPosition({10, particles[i].get_position().y});
        }

    }
}

void Engine::Update() {
    // Do calculations
    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        densities[i] = CalculateDensity(particles[i].get_position());
    }

    // Calcluate and apply pressure
    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        Vector2 pressureForce = CalculatePressureForce(particles[i].get_position());
        pressures[i] = Vector2Scale(pressureForce, 1.0f / densities[i]);
    }

    SetWindowTitle(TextFormat("Fluids - %i",GetFPS()));
}

void Engine::Reset() {
    particles.clear();
    forces.clear();
    densities.clear();
    pressures.clear();
}

void Engine::Populate() {
    int count = 0;
    for (int i = 800; i < 1200; i+= 8) {
        for (int j = 400; j < 600; j+= 8) {
            particles.push_back(Particle(i,j));
            count++;
        }
    }
    simulation_size = count;
}

float Engine::CalculateDensity(Vector2 point) {
    float density = 0;
    const float mass = 1000;

    for (Particle p : particles) {
        float dist = Vector2Distance(p.get_position(), point);
        if (dist < smoothing_radius) {
            density += mass * SmoothingKernel(dist);
        }
    }
    return density;
}


float Engine::CalculateProperty(Vector2 point) {
    float property = 0;
    const float mass = 1000;

    for (int i = 0; i < particles.size(); i++) {
        float dist = Vector2Distance(particles[i].get_position(), point);
        if (dist < smoothing_radius) {
            property += mass * SmoothingKernel(dist) / densities[i];
        }
    }
    return property;
}

float Engine::DensityToPressure(float density)
{
    float densityError = density - targetDensity;
    float pressure = densityError * pressureMultiplier;
    return pressure;
}

Vector2 Engine::CalculatePropertyGradient(Vector2 point) {
    Vector2 gradient = {0,0};
    const float mass = 1000000;

    for (int i = 0; i < particles.size(); i++) {
        Vector2 dist = Vector2Subtract(particles[i].get_position(), point);
        float distance = Vector2Length(dist);
        if (distance < smoothing_radius) {
            Vector2 scale = Vector2Scale(dist, SmoothingKernelDerivative(distance) / densities[i]);
            scale = Vector2Scale(scale, mass);
            gradient = Vector2Add(gradient, scale);
        }
    }
    return gradient;
}

Vector2 Engine::CalculatePressureForce(Vector2 point) {
    Vector2 gradient = {0,0};
    const float mass = 1000000;
    float dentPoint = CalculateDensity(point); 
    for (int i = 0; i < particles.size(); i++) {
        Vector2 dist = Vector2Subtract(particles[i].get_position(), point);
        float distance = Vector2Length(dist);
        if (distance < smoothing_radius) {
            float sharedPressure = CalculateSharedPressure(dentPoint, densities[i]);
            Vector2 scale = Vector2Scale(dist, SmoothingKernelDerivative(distance) / densities[i]);
            scale = Vector2Scale(scale, mass);
            scale = Vector2Scale(scale , sharedPressure);
            gradient = Vector2Add(gradient, scale);
        }
    }
    return gradient;
}


float Engine::CalculateSharedPressure(float dens1, float dens2) {
    float pressure1 = DensityToPressure(dens1);
    float pressure2 = DensityToPressure(dens2);
    return (pressure1 + pressure2) / 2;
}

float Engine::SmoothingKernel(float dist) {
    if (dist >= smoothing_radius)
        return 0;
    float volume = PI * std::pow(smoothing_radius , 4) / 6;
    return (smoothing_radius - dist) * (smoothing_radius - dist) / volume;

}

float Engine::SmoothingKernelDerivative(float dist) {
    if (dist < smoothing_radius) {
        float scale = 12  / (PI * std::pow(smoothing_radius, 4));
        return (dist - smoothing_radius) * scale;
    }
    return 0;
}

#pragma region UI
void Engine::ShowDensity() {
    Vector2 pos = GetMousePosition();
    float density = CalculateDensity(pos);
    DrawCircleLinesV(pos, smoothing_radius, RED);
    DrawText(std::to_string(density).c_str(), pos.x, pos.y - 80, 20, RED);
}

void Engine::DrawGradinet() {
    for (int i = 4; i < 1600; i+=8) {
        for (int j = 4; j < 900; j+=8) {
            Vector2 pos = {i,j};
            Vector2 grad = CalculatePropertyGradient(pos);
            DrawLineV(pos, Vector2Add(pos, Vector2Scale(grad, 0.1)), RED);
        }
    }
}

#pragma endregion

