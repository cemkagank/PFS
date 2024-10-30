#include "Engine.hpp"
#include "time.h"
#include <cmath>
#include <iostream>
#include <raymath.h>
#include <omp.h>
#include <algorithm>

// TODO: Transform to 3D
// TODO: Add camera
// FIXME: simulatin_size to update vectors
// FIXME: Edge teleportation
// HACK : Parallelize calculations improve performance
// TODO : replave vector or minimize allocation, copy and move operations --emplace_back and reserve

Engine::Engine() {
    particles.reserve(simulation_size);
    spatial_lookup.reserve(simulation_size);
    start_indices.reserve(simulation_size);
    densities.reserve(simulation_size);
    pressures.reserve(simulation_size);
    forces.reserve(simulation_size);
    

}

void Engine::Draw() {
    for (Particle p : particles) {
        p.Draw();
    }
}

void Engine::SimulationStep() {

    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        particles[i].Update();
    }

    // Apply gravity
    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        particles[i].ApplyForce({0,gravity * GetFrameTime()});
    }

    //TODO: Add viscosity

    // Apply pressure
    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        particles[i].ApplyForce(pressures[i]);
    }

    // TODO: Fix the edges
    // TODO: Smarter way to handle collisions
    //  Update positions and resolve collisions

    ResolveCollisions();

}

void Engine::Update() {

    UpdateSpatialLookup();

    // Calculate densities
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

}


void Engine::ResolveCollisions() {
    const int width = 1200;
    const int height = 800;

    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        Particle& p = particles[i];
        if (p.get_position().x > width) {
            p.SetPosition({width - 4, p.get_position().y});
            p.SetVelocity({0,0});
            p.ApplyForce({-gravity * 3 ,0 });
        } else if (p.get_position().x < 600) {
            p.SetPosition({604, p.get_position().y});
            p.SetVelocity({0,0});
            p.ApplyForce({gravity * 3 ,0 });
        }
        if (p.get_position().y > height) {
            p.SetPosition({p.get_position().x, height - 4});
            p.SetVelocity({0,0});
            p.ApplyForce({0, -gravity * 3});
        } else if (p.get_position().y < 200) {
            p.SetPosition({p.get_position().x, 204});
            p.SetVelocity({0,0});
            p.ApplyForce({0, gravity * 3});
        }

    }

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
            particles.emplace_back(Particle(i,j));
            count++;
        }
    }
    simulation_size = count;
    densities.resize(count);
    pressures.resize(count);
    forces.resize(count);
    particles.shrink_to_fit();
}

void Engine::Repopulate(Vector2 pos) {
    for (int i = 0; i < 100; i++) {
        float x = pos.x + GetRandomValue(-10, 10);
        float y = pos.y + GetRandomValue(-10, 10);
        particles.emplace_back(Particle(x,y));
    }
}

void Engine::UpdateSpatialLookup(){
    
    spatial_lookup.resize(particles.size());
    start_indices.resize(particles.size());

    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        auto [cellx, celly] = PositionToCellCoord(particles[i].get_position());
        unsigned int cellkey = GetKeyFromHash(HashCell(cellx, celly));
        spatial_lookup[i] = {i, cellkey};
        start_indices[i] = -1;
    }
    // sort by cell key
    std::sort(spatial_lookup.begin(), spatial_lookup.end(), [](const std::pair<int, unsigned int>& a, const std::pair<int, unsigned int>& b) {
        return a.second < b.second;
    });

    // Calculate start indices of each unique cell key in the spatial lookup
    #pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        unsigned int key = spatial_lookup[i].second;
        unsigned int keyPrev = i == 0 ? -1 : spatial_lookup[i - 1].second;
        if (key != keyPrev) {
            start_indices[key] = i;
        }
    }
}

void Engine::ForEachPointinRadius(Vector2 point) {
    auto [cellx, celly] = PositionToCellCoord(point);
    float squared_radius = smoothing_radius * smoothing_radius;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            unsigned int key = GetKeyFromHash(HashCell(cellx + i, celly + j));
            if (start_indices[key] != -1) {
                for (int k = start_indices[key]; k < spatial_lookup.size(); k++) {
                    if (spatial_lookup[k].second != key) break;
                    int index = spatial_lookup[k].first;
                    Vector2 dist = Vector2Subtract(particles[index].get_position(), point);
                    float squared_distance = Vector2Length(dist);
                    if (squared_distance < squared_radius) {
                        densities[index] += CalculateDensity(point);
                    }
                }
            }
        }
    }

}



std::pair<int, int> Engine::PositionToCellCoord(Vector2 point) {
    int cellx = (point.x - 800) / smoothing_radius;
    int celly = (point.y - 400) / smoothing_radius;
    return {cellx, celly};
}

unsigned int Engine::HashCell(int cellx, int celly) {
    return cellx * 73856093 + celly * 19349663;
}

unsigned int Engine::GetKeyFromHash(unsigned int hash) {
    return hash % simulation_size;
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

void Engine::ShowDensity() {
    Vector2 pos = GetMousePosition();
    float density = CalculateDensity(pos);
    DrawCircleLinesV(pos, smoothing_radius, GREEN);
    DrawText(std::to_string(density).c_str(), pos.x, pos.y - 80, 20, GREEN);
}
