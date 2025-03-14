#include "Engine.hpp"
#include "time.h"
#include <cmath>
#include <cstddef>
#include <raymath.h>
#include <omp.h>
#include <algorithm>
#include <raylib.h>

// FIXME: Collusion handling still sucks
// FIXME: all the spatil lookup stuff including cell and hasing is 2d make it 3d
// PERF : Improve general optimization , reach 10K particles
// TODO : Add rotation
// TODO : More watery behaviour
// TODO : replave vector or minimize allocation, copy and move operations --emplace_back and reserve

float Engine::particle_radius = 0.2f;                     // Default value
float Engine::particle_color[4] = {0, 0, 1, 1};           // blue

Engine::Engine() {
    container = box{Vector3{20,20,20},Vector3{-20,-20,-20}};
    // TODO : Init Cuda
    // TODO : init d_positions
    particle_radius = 0.15f;
    
    particle_color[0] = 0.2f;
    particle_color[1] = 0.5f;
    particle_color[2] = 0.9f;
    particle_color[3] = 0.7f;
    
    particleMesh = GenMeshSphere(1.0f, 8, 8);
    
    mat = LoadMaterialDefault();
    mat.maps[MATERIAL_MAP_DIFFUSE].color = Color{
        (unsigned char)(particle_color[0] * 255),
        (unsigned char)(particle_color[1] * 255),
        (unsigned char)(particle_color[2] * 255),
        (unsigned char)(particle_color[3] * 255)
    };
    UploadMesh(&particleMesh, false);
}

void Engine::Draw() {
    for (int i = 0; i < simulation_size; i++) {
    mat.maps[MATERIAL_MAP_DIFFUSE].color = Interpolate(i);
        DrawMesh(particleMesh, mat, transforms[i]);
    }
    container.Draw();

   
}

Color Engine::Interpolate(int index) {
    float vel_mag = Vector3Length(velocities[index]);
    float t = fminf(vel_mag / 15.0f, 1.0f);  // Changed from 3.0f to 15.0f for smoother transition
    
    if (t < 0.5f) {
        // Blue to Yellow (0.0 -> 0.5)
        float t2 = t * 2.0f;
        return Color{
            (unsigned char)(t2 * 255),          // Red
            (unsigned char)(t2 * 255),          // Green
            (unsigned char)((1.0f - t2) * 255), // Blue
            255                                 // Alpha
        };
    } else {
        // Yellow to Red (0.5 -> 1.0)
        float t2 = (t - 0.5f) * 2.0f;
        return Color{
            255,                                // Red
            (unsigned char)((1.0f - t2) * 255), // Green
            0,                                  // Blue
            255                                 // Alpha
        };
    }
}


void Engine::Update() {
    // TODO : Copy positions to CUDA
    // TODO : Launch kernels
    // TODO : Get positions from CUDA
    // TODO : Update Transformations
}

void Engine::ResolveCollisions() {
    const float dampening = 0.6f;        // Less dampening on collision
    const float epsilon = 0.1f;          // Boundary buffer
    const float friction = 0.98f;        // Reduced friction to allow sliding
    
    #pragma omp parallel for
    for (int i = 0; i < simulation_size; i++) {
        bool collision = false;
        
        // X-axis collisions with less dampening
        if (positions[i].x < container.min.x + epsilon) {
            positions[i].x = container.min.x + epsilon;
            velocities[i].x *= -dampening;
            collision = true;
        }
        if (positions[i].x > container.max.x - epsilon) {
            positions[i].x = container.max.x - epsilon;
            velocities[i].x *= -dampening;
            collision = true;
        }
        
        // Y-axis collisions
        if (positions[i].y < container.min.y + epsilon) {
            positions[i].y = container.min.y + epsilon;
            velocities[i].y *= -dampening;
            collision = true;
        }
        if (positions[i].y > container.max.y - epsilon) {
            positions[i].y = container.max.y - epsilon;
            velocities[i].y *= -dampening;
            collision = true;
        }
        
        // Z-axis collisions with less dampening
        if (positions[i].z < container.min.z + epsilon) {
            positions[i].z = container.min.z + epsilon;
            velocities[i].z *= -dampening;
            collision = true;
        }
        if (positions[i].z > container.max.z - epsilon) {
            positions[i].z = container.max.z - epsilon;
            velocities[i].z *= -dampening;
            collision = true;
        }
        
        // Apply lighter friction when colliding
        if (collision) {
            velocities[i].x *= friction;
            velocities[i].z *= friction;
        }
    }
}

void Engine::Reset() {
    positions.clear();
    velocities.clear();
    densities.clear();
    forces.clear();
    pressures.clear();
    transforms.clear();
    simulation_size = 0;
}

void Engine::Populate() {
    int count = 0;
    
    // Clear existing data
    Reset();
    
    for (int i = -10; i < 10; i+=1) {
        for (int j = -5; j < 5; j+=1) {
            for (int k = -5; k < 5; k+= 1) {
                positions.push_back(Vector3{static_cast<float>(i),static_cast<float>(j),static_cast<float>(k)});
                velocities.push_back(Vector3{0,0,0});
                Matrix scale = MatrixScale(particle_radius, particle_radius, particle_radius);
                Matrix translation = MatrixTranslate(i, j, k);
                transforms.push_back(MatrixMultiply(scale, translation));
                count++;
            }
        }
    }
    
    simulation_size = count;
    transforms.resize(count);
    densities.resize(count);
    pressures.resize(count);
    forces.resize(count);
    velocities.resize(count);
    
}












void Engine::SpawnParticlesAtCenter() {
    // Calculate box center
    Vector3 center = {
        (container.max.x + container.min.x) / 2.0f,
        (container.max.y + container.min.y) / 2.0f,
        (container.max.z + container.min.z) / 2.0f
    };
    
    // Spawn radius - how far from center particles can appear
    float spawn_radius = 2.0f;
    
    // Add 100 particles
    for (int i = 0; i < 1000; i++) {
        // Random position within a sphere around center
        Vector3 pos = {
            center.x + (GetRandomValue(-100, 100) / 100.0f) * spawn_radius,
            center.y + (GetRandomValue(-100, 100) / 100.0f) * spawn_radius,
            center.z + (GetRandomValue(-100, 100) / 100.0f) * spawn_radius
        };
        
        positions.push_back(pos);
        
        
        Matrix scale = MatrixScale(particle_radius, particle_radius, particle_radius);
        Matrix translation = MatrixTranslate(pos.x, pos.y, pos.z);
        transforms.push_back(MatrixMultiply(scale, translation));
    }
    
    // Update simulation size and resize other vectors
    simulation_size += 1000;
   
    
   
}

