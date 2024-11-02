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

float Engine::particle_radius = 4.0f;  // Default value
float Engine::particle_color[4] = {0, 0, 1, 1};           // blue

Engine::Engine() {
    

}

void Engine::Draw() {
    for (size_t i = 0; i < positions.size(); i++) {
        DrawCircleV(positions[i], particle_radius, Color{(unsigned char)(particle_color[0] * 255), (unsigned char)(particle_color[1] * 255), (unsigned char)(particle_color[2] * 255), (unsigned char)(particle_color[3] * 255)});
    }
}

void Engine::SimulationStep() {

    #pragma omp parallel for
    for (size_t i = 0; i < positions.size(); i++) {
        positions[i] += Vector2Scale(velocities[i], GetFrameTime() * 100);
    }

    // Apply gravity
    #pragma omp parallel for
    for (size_t i = 0; i < positions.size(); i++) {
        velocities[i] += {0,gravity * GetFrameTime()};
    }

    //TODO: Add viscosity

    // Apply pressure
    #pragma omp parallel for
    for (size_t i = 0; i < positions.size(); i++) {
        velocities[i] += pressures[i];
    }

    // TODO: Fix the edges
    // TODO: Smarter way to handle collisions
    //  Update positions and resolve collisions

    ResolveCollisions();

}

void Engine::Update() {

    // UpdateSpatialLookup();

    // Calculate densities
    #pragma omp parallel for
    for (size_t i = 0; i < positions.size(); i++) {
        densities[i] = CalculateDensity(positions[i]);
    
    }   


    // Calcluate and apply pressure
    #pragma omp parallel for
    for (size_t i = 0; i < positions.size(); i++) {
        Vector2 pressureForce = CalculatePressureForce(positions[i]);
        pressures[i] = Vector2Scale(pressureForce, 1.0f / densities[i]);
    }

}


void Engine::ResolveCollisions() {
    const int width = 1200;
    const int height = 800;
    const float bounce = 0.7f; // Increased bounce factor
    const float buffer = 15.0f; // Increased buffer zone

    #pragma omp parallel for
    for (size_t i = 0; i < positions.size(); i++) {
        // Apply gradual repulsion force near walls
        const float repulsionZone = 50.0f; // Increased repulsion zone
        const float maxRepulsion = 2.0f; // Maximum repulsion force
        
        // Right wall repulsion
        if (positions[i].x > width - repulsionZone) {
            float distance = width - positions[i].x;
            float force = maxRepulsion * (1.0f - (distance / repulsionZone));
            forces[i].x -= force;
            velocities[i].x -= force * 0.1f; // Add immediate velocity change
        }
        // Left wall repulsion
        if (positions[i].x < 600 + repulsionZone) {
            float distance = positions[i].x - 600;
            float force = maxRepulsion * (1.0f - (distance / repulsionZone));
            forces[i].x += force;
            velocities[i].x += force * 0.1f;
        }
        // Bottom wall repulsion
        if (positions[i].y > height - repulsionZone) {
            float distance = height - positions[i].y;
            float force = maxRepulsion * (1.0f - (distance / repulsionZone));
            forces[i].y -= force;
            velocities[i].y -= force * 0.1f;
        }
        // Top wall repulsion
        if (positions[i].y < 200 + repulsionZone) {
            float distance = positions[i].y - 200;
            float force = maxRepulsion * (1.0f - (distance / repulsionZone));
            forces[i].y += force;
            velocities[i].y += force * 0.1f;
        }

        // Hard boundary collision handling
        if (positions[i].x > width - buffer) {
            positions[i].x = width - buffer;
            velocities[i].x *= -bounce;
            velocities[i].y += GetRandomValue(-20, 20) / 100.0f; // Increased randomness
        } 
        else if (positions[i].x < 600 + buffer) {
            positions[i].x = 600 + buffer;
            velocities[i].x *= -bounce;
            velocities[i].y += GetRandomValue(-20, 20) / 100.0f;
        }
        
        if (positions[i].y > height - buffer) {
            positions[i].y = height - buffer;
            velocities[i].y *= -bounce;
            velocities[i].x += GetRandomValue(-20, 20) / 100.0f;
        } 
        else if (positions[i].y < 200 + buffer) {
            positions[i].y = 200 + buffer;
            velocities[i].y *= -bounce;
            velocities[i].x += GetRandomValue(-20, 20) / 100.0f;
        }

        // Add some random motion to particles near walls to prevent stagnation
        if (positions[i].x < 600 + repulsionZone || 
            positions[i].x > width - repulsionZone ||
            positions[i].y < 200 + repulsionZone || 
            positions[i].y > height - repulsionZone) {
            velocities[i].x += GetRandomValue(-10, 10) / 100.0f;
            velocities[i].y += GetRandomValue(-10, 10) / 100.0f;
        }
    }

}

void Engine::Reset() {
    forces.clear();
    densities.clear();
    pressures.clear();
    positions.clear();
    velocities.clear();
}

void Engine::Populate() {
    int count = 0;
    for (int i = 800; i < 1200; i+= 8) {
        for (int j = 400; j < 600; j+= 8) {
            positions.push_back(Vector2{(float)i, (float)j});
            velocities.push_back(Vector2{0.0f, 0.0f});
            count++;
        }
    }
    simulation_size = count;
    densities.resize(count);
    pressures.resize(count);
    forces.resize(count);
}

void Engine::Repopulate(Vector2 pos) {
    for (int i = 0; i < 100; i++) {
        float x = pos.x + GetRandomValue(-10, 10);
        float y = pos.y + GetRandomValue(-10, 10);
        positions.push_back(Vector2{x, y});
        velocities.push_back(Vector2{0.0f, 0.0f});
    }
}

void Engine::UpdateSpatialLookup(){
    
    spatial_lookup.resize(positions.size());
    start_indices.resize(positions.size());

    #pragma omp parallel for
    for (size_t i = 0; i < positions.size(); i++) {
        std::pair<int, int> cell = PositionToCellCoord(positions[i]);
        int cellx = cell.first;
        int celly = cell.second;
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
    for (size_t i = 0; i < positions.size(); i++) {
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
                for (size_t k = start_indices[key]; k < spatial_lookup.size(); k++) {
                    if (spatial_lookup[k].second != key) break;
                    size_t index = spatial_lookup[k].first;
                    Vector2 dist = Vector2Subtract(positions[index], point);
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

    for (size_t i = 0; i < positions.size(); i++) {
        float dist = Vector2Distance(positions[i], point);
        if (dist < smoothing_radius) {
            density += mass * SmoothingKernel(dist);
        }
    }
    return density;
}


float Engine::CalculateProperty(Vector2 point) {
    float property = 0;
    const float mass = 1000;

    for (size_t i = 0; i < positions.size(); i++) {
        float dist = Vector2Distance(positions[i], point);
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

    for (size_t i = 0; i < positions.size(); i++) {
        Vector2 dist = Vector2Subtract(positions[i], point);
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
    for (size_t i = 0; i < positions.size(); i++) {
        Vector2 dist = Vector2Subtract(positions[i], point);
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
