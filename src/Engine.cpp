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
    
    // Smaller particles for better fluid resolution
    particle_radius = 0.15f;
    
    // More water-like blue color with transparency
    particle_color[0] = 0.2f;
    particle_color[1] = 0.5f;
    particle_color[2] = 0.9f;
    particle_color[3] = 0.7f;
    
    // Create a smaller unit sphere mesh (radius 1.0) with fewer subdivisions
    particleMesh = GenMeshSphere(1.0f, 8, 8);
    
    // Load material with adjusted properties
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

void Engine::SimulationStep() {
    const float rest_damping = 0.98f;    // General movement damping
    const float floor_friction = 0.92f;   // Ground friction
    
    
    UpdateSpatialLookup();

    #pragma omp parallel for
    for (int i = 0; i < simulation_size; i++) {
        // Apply pressure forces
        Vector3 pressureForce = CalculatePressureForce(positions[i]);
        Vector3 viscosityForce = CalculateViscosityForce(positions[i]);
        
        // Combine forces with proper scaling
        velocities[i] = Vector3Add(velocities[i], 
            Vector3Scale(pressureForce, 1.0f / densities[i]));
        velocities[i] = Vector3Add(velocities[i], 
            Vector3Scale(viscosityForce, viscosity));
        
        // Update position
        positions[i] = Vector3Add(positions[i], 
            Vector3Scale(velocities[i], GetFrameTime()));
            
        // Apply general damping
        velocities[i] = Vector3Scale(velocities[i], rest_damping);
    }

    // Gravity
    #pragma omp parallel for
    for (int i = 0; i < simulation_size; i++) {
        velocities[i].y += gravity * GetFrameTime();
    }

    // Enhanced collision handling with improved boundary behavior
    #pragma omp parallel for
    for (int i = 0; i < simulation_size; i++) {
        const float bounce = 0.3f;
        const float boundary_friction = 0.8f;
        const float min_velocity = 0.1f;
        const float epsilon = particle_radius * 0.5f;  // Boundary buffer based on particle size
        
        // X boundaries with improved collision response
        if (positions[i].x < container.min.x + epsilon) {
            positions[i].x = container.min.x + epsilon;
            if (velocities[i].x < 0) {
                velocities[i].x = -velocities[i].x * bounce;
                // Apply friction to other components
                velocities[i].y *= boundary_friction;
                velocities[i].z *= boundary_friction;
            }
        }
        if (positions[i].x > container.max.x - epsilon) {
            positions[i].x = container.max.x - epsilon;
            if (velocities[i].x > 0) {
                velocities[i].x = -velocities[i].x * bounce;
                velocities[i].y *= boundary_friction;
                velocities[i].z *= boundary_friction;
            }
        }
        
        // Y boundaries with improved ground interaction
        if (positions[i].y < container.min.y + epsilon) {
            positions[i].y = container.min.y + epsilon;
            
            // Check if particle is moving downward
            if (velocities[i].y < 0) {
                // If velocity is very small, let the particle settle
                if (fabsf(velocities[i].y) < min_velocity) {
                    velocities[i].y = 0;
                    // Stronger friction for settled particles
                    velocities[i].x *= floor_friction;
                    velocities[i].z *= floor_friction;
                } else {
                    // Bounce with energy loss
                    velocities[i].y = -velocities[i].y * bounce;
                    // Apply horizontal friction
                    velocities[i].x *= boundary_friction;
                    velocities[i].z *= boundary_friction;
                }
            }
        }
        if (positions[i].y > container.max.y - epsilon) {
            positions[i].y = container.max.y - epsilon;
            if (velocities[i].y > 0) {
                velocities[i].y = -velocities[i].y * bounce;
                velocities[i].x *= boundary_friction;
                velocities[i].z *= boundary_friction;
            }
        }
        
        // Z boundaries with improved collision response
        if (positions[i].z < container.min.z + epsilon) {
            positions[i].z = container.min.z + epsilon;
            if (velocities[i].z < 0) {
                velocities[i].z = -velocities[i].z * bounce;
                velocities[i].x *= boundary_friction;
                velocities[i].y *= boundary_friction;
            }
        }
        if (positions[i].z > container.max.z - epsilon) {
            positions[i].z = container.max.z - epsilon;
            if (velocities[i].z > 0) {
                velocities[i].z = -velocities[i].z * bounce;
                velocities[i].x *= boundary_friction;
                velocities[i].y *= boundary_friction;
            }
        }

        // Apply velocity damping for very small velocities
        if (Vector3Length(velocities[i]) < min_velocity) {
            velocities[i] = Vector3Scale(velocities[i], 0.5f);
        }
    }

    // Update transforms
    #pragma omp parallel for 
    for (int i = 0; i < simulation_size; i++) {
        Matrix scale = MatrixScale(particle_radius, particle_radius, particle_radius);
        Matrix translation = MatrixTranslate(positions[i].x, positions[i].y, positions[i].z);
        transforms[i] = MatrixMultiply(scale, translation);
    }
}

void Engine::Update() {
    UpdateSpatialLookup();

    #pragma omp parallel for
    for (int i = 0; i < simulation_size; i++) {
        // Increase density influence
        densities[i] = CalculateDensity(positions[i]) * 2.0f;
    }   

    #pragma omp parallel for
    for (int i = 0; i < simulation_size; i++) {
        Vector3 pressureForce = CalculatePressureForce(positions[i]);
        pressures[i] = Vector3Scale(pressureForce, 3.0f / densities[i]);
    }
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
    
    // Create a more compact initial distribution
    float spacing = particle_radius * 2.0f;  // Reduced spacing between particles
    
    for (int i = -8; i < 8; i++) {
        for (int j = -4; j < 4; j++) {
            for (int k = -4; k < 4; k++) {
                positions.push_back(Vector3{
                    static_cast<float>(i) * spacing,
                    static_cast<float>(j) * spacing,
                    static_cast<float>(k) * spacing
                });
                velocities.push_back(Vector3{0,0,0});
                Matrix scale = MatrixScale(particle_radius, particle_radius, particle_radius);
                Matrix translation = MatrixTranslate(i * spacing, j * spacing, k * spacing);
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
    
    TraceLog(LOG_INFO, "Populated %d particles", count);
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

void Engine::ForEachPointinRadius(Vector3 point) {
    auto [cellx, celly] = PositionToCellCoord(point);
    float squared_radius = smoothing_radius * smoothing_radius;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            unsigned int key = GetKeyFromHash(HashCell(cellx + i, celly + j));
            if (start_indices[key] != -1) {
                for (size_t k = start_indices[key]; k < spatial_lookup.size(); k++) {
                    if (spatial_lookup[k].second != key) break;
                    size_t index = spatial_lookup[k].first;
                    Vector3 dist = positions[index] - point;
                    float squared_distance = Vector3Length(dist);
                    if (squared_distance < squared_radius) {
                        densities[index] += CalculateDensity(point);
                    }
                }
            }
        }
    }
}

std::pair<int, int> Engine::PositionToCellCoord(Vector3 point) {
    int cellx = point.x  / smoothing_radius;
    int celly = point.y  / smoothing_radius;
    return {cellx, celly};
}

unsigned int Engine::HashCell(int cellx, int celly) {
    return cellx * 73856093 + celly * 19349663;
}

unsigned int Engine::GetKeyFromHash(unsigned int hash) {
    return hash % simulation_size;
}

float Engine::CalculateDensity(Vector3 point) {
    float density = 0;
    const float mass = 1.5f;  // Adjusted mass for better pressure response
    
    auto [cellx, celly] = PositionToCellCoord(point);

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            unsigned int key = GetKeyFromHash(HashCell(cellx + i, celly + j));
            if (start_indices[key] != -1) {
                for (size_t k = start_indices[key]; k < spatial_lookup.size(); k++) {
                    if (spatial_lookup[k].second != key) break;
                    size_t index = spatial_lookup[k].first;
                    float dist = Vector3Distance(positions[index], point);
                    if (dist < smoothing_radius) {
                        float factor = (smoothing_radius - dist) / smoothing_radius;
                        density += mass * SmoothingKernel(dist) * factor;
                    }
                }
            }
        }
    }
    return density;
}

float Engine::DensityToPressure(float density) {
    float densityError = density - targetDensity;
    float pressure = densityError * pressureMultiplier;
    // Add non-linear pressure response
    if (density > targetDensity) {
        pressure *= (1.0f + (density - targetDensity) * 0.5f);
    }
    return pressure;
}

Vector3 Engine::CalculatePressureForce(Vector3 point) {
    Vector3 gradient = {0, 0, 0};
    const float mass = 1000000;
    float densPoint = CalculateDensity(point);
    
    auto [cellx, celly] = PositionToCellCoord(point);
    // Check neighboring cells
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            unsigned int key = GetKeyFromHash(HashCell(cellx + i, celly + j));
            if (start_indices[key] != -1) {
                // Iterate through particles in this cell
                for (size_t k = start_indices[key]; k < spatial_lookup.size(); k++) {
                    if (spatial_lookup[k].second != key) break;
                    size_t index = spatial_lookup[k].first;
                    Vector3 dist = positions[index] - point;
                    float distance = Vector3Length(dist);
                    if (distance < smoothing_radius) {
                        float sharedPressure = CalculateSharedPressure(densPoint, densities[index]);
                        Vector3 scale = Vector3Scale(dist, SmoothingKernelDerivative(distance) / densities[index]);
                        scale = Vector3Scale(scale, mass);
                        scale = Vector3Scale(scale, sharedPressure);
                        gradient = gradient + scale;
                    }
                }
            }
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
        velocities.push_back(Vector3{0, 0, 0});
        
        Matrix scale = MatrixScale(particle_radius, particle_radius, particle_radius);
        Matrix translation = MatrixTranslate(pos.x, pos.y, pos.z);
        transforms.push_back(MatrixMultiply(scale, translation));
    }
    
    // Update simulation size and resize other vectors
    simulation_size += 1000;
    densities.resize(simulation_size);
    pressures.resize(simulation_size);
    forces.resize(simulation_size);
    
    // Debug output
    TraceLog(LOG_INFO, "Spawned 100 particles at center. Total particles: %d", simulation_size);
}

Vector3 Engine::CalculateViscosityForce(Vector3 point) {
    Vector3 viscosityForce = {0, 0, 0};
    const float mass = 1.0f;
    
    auto [cellx, celly] = PositionToCellCoord(point);

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            unsigned int key = GetKeyFromHash(HashCell(cellx + i, celly + j));
            if (start_indices[key] != -1) {
                for (size_t k = start_indices[key]; k < spatial_lookup.size(); k++) {
                    if (spatial_lookup[k].second != key) break;
                    size_t index = spatial_lookup[k].first;
                    Vector3 dist = positions[index] - point;
                    float distance = Vector3Length(dist);
                    if (distance < smoothing_radius && distance > 0.0001f) {
                        Vector3 velocityDiff = Vector3Subtract(velocities[index], velocities[spatial_lookup[k].first]);
                        float factor = SmoothingKernel(distance) / densities[index];
                        viscosityForce = Vector3Add(viscosityForce, 
                            Vector3Scale(velocityDiff, factor * mass));
                    }
                }
            }
        }
    }
    return viscosityForce;
}

