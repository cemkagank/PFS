#include "Engine.hpp"
#include "time.h"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <raymath.h>
#include <omp.h>
#include <algorithm>
#include <raylib.h>

// TODO: Transform to 3D
// TODO: Add camera
// FIXME: simulaton_size to update vectors
// FIXME: Edge teleportation
// HACK : Parallelize calculations improve performance
// TODO : replave vector or minimize allocation, copy and move operations --emplace_back and reserve

float Engine::particle_radius = 0.2f;                     // Default value
float Engine::particle_color[4] = {0, 0, 1, 1};           // blue

Engine::Engine() {
    container = box{Vector3{10,10,10},Vector3{-10,-10,-10}};
    particleMesh = GenMeshSphere(1.0f,  8, 8);
    mat = LoadMaterialDefault();
    shader = LoadShader("./shaders/instancing.vs", "./shaders/instancing.fs" );
    shader.locs[SHADER_LOC_MATRIX_MVP] = GetShaderLocation(shader, "mvp");

    mat.shader = shader;
    mat.maps[MATERIAL_MAP_DIFFUSE].color = BLUE;
}

void Engine::Draw() {
    //for (size_t i = 0; i < positions.size(); i++) {
    //   DrawSphere(positions[i], particle_radius, Color{(unsigned char)(particle_color[0] * 255), (unsigned char)(particle_color[1] * 255), (unsigned char)(particle_color[2] * 255), (unsigned char)(particle_color[3] * 255)});
    //}
    DrawMeshInstanced( particleMesh,  mat, transforms.data(),  simulation_size);
    container.Draw();
}

void Engine::SimulationStep() {

    #pragma omp parallel for
    for (size_t i = 0; i < positions.size(); i++) {
        positions[i] += velocities[i] *  GetFrameTime() * 1;
    }

    #pragma omp parallel for 
    for (int i = 0; i < simulation_size; i++) {
        transforms[i] = MatrixTranslate(positions[i].x, positions[i].y, positions[i].z);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < positions.size(); i++) {
        velocities[i] += {0,gravity * GetFrameTime(), 0};
    }

    #pragma omp parallel for
    for (size_t i = 0; i < positions.size(); i++) {
        velocities[i] += pressures[i];
    }

    ResolveCollisions();

}

void Engine::Update() {


    #pragma omp parallel for
    for (size_t i = 0; i < positions.size(); i++) {
        densities[i] = CalculateDensity(positions[i]);
    
    }   

    #pragma omp parallel for
    for (size_t i = 0; i < positions.size(); i++) {
        Vector3 pressureForce = CalculatePressureForce(positions[i]);
        pressures[i] = Vector3Scale(pressureForce, 1.0f / densities[i]);
    }


}


void Engine::ResolveCollisions() {
    
    #pragma omp parallel for
    for (size_t i = 0 ; i < positions.size(); i++ ) {
        if (positions[i].x < container.min.x || positions[i].x > container.max.x)
            velocities[i].x *= -1;
        if (positions[i].y < container.min.y || positions[i].y > container.max.y)
            velocities[i].y *= -1;
        if (positions[i].z < container.min.z || positions[i].z > container.max.z)
            velocities[i].z *= -1;
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
        
    for (int i = -10; i < 10; i+=1) {
        for (int j = -5; j < 5; j+=1) {
            for (int k = -5; k < 5; k+= 1){
                positions.push_back(Vector3{static_cast<float>(i),static_cast<float>(j),static_cast<float>(k)});
                velocities.push_back(Vector3{ 0,0,0 });
                transforms.push_back(MatrixTranslate(i, j, j));
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

float Engine::CalculateDensity(Vector3 point) {
    float density = 0;
    const float mass = 1000;

    for (size_t i = 0; i < positions.size(); i++) {
        float dist = Vector3Distance(positions[i], point);
        if (dist < smoothing_radius) {
            density += mass * SmoothingKernel(dist);
        }
    }
    return density;
}

float Engine::DensityToPressure(float density)
{
    float densityError = density - targetDensity;
    float pressure = densityError * pressureMultiplier;
    return pressure;
}


Vector3 Engine::CalculatePressureForce(Vector3 point) {
    Vector3 gradient = {0,0, 0};
    const float mass = 1000000;
    float dentPoint = CalculateDensity(point);
    for (size_t i = 0; i < positions.size(); i++) {
        Vector3 dist = positions[i] -  point;
        float distance = Vector3Length(dist);
        if (distance < smoothing_radius) {
            float sharedPressure = CalculateSharedPressure(dentPoint, densities[i]);
            Vector3 scale = Vector3Scale(dist, SmoothingKernelDerivative(distance) / densities[i]);
            scale = Vector3Scale(scale, mass);
            scale = Vector3Scale(scale , sharedPressure);
            gradient = gradient + scale;
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

