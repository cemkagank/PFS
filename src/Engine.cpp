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

float Engine::particle_radius = 0.2f;                   
float Engine::particle_color[4] = {0, 0, 1, 1};           

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

Engine::~Engine()
{
    // TODO: Cuda free
}

void Engine::Draw() {
    for (int i = 0; i < simulation_size; i++) {
        DrawMesh(particleMesh, mat, transforms[i]);
    }
    container.Draw();

   
}


void Engine::Update() {
    // TODO : Copy positions to CUDA
    // TODO : Launch kernels
    // TODO : Get positions from CUDA
    // TODO : Update Transformations
}


void Engine::Reset() {
    positions.clear();
    simulation_size = 0;
}

void Engine::Populate() {
    int count = 0;
    for (int i = -10; i < 10; i+=1) {
        for (int j = -5; j < 5; j+=1) {
            for (int k = -5; k < 5; k+= 1) {
                positions.push_back(Vector3{static_cast<float>(i),static_cast<float>(j),static_cast<float>(k)});
                Matrix scale = MatrixScale(particle_radius, particle_radius, particle_radius);
                Matrix translation = MatrixTranslate(i, j, k);
                transforms.push_back(MatrixMultiply(scale, translation));
                count++;
            }
        }
    }
    
    simulation_size = count;
    transforms.resize(count);
}