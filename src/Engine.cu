#include "Engine.hpp"
#include "time.h"
#include <cmath>
#include <cstddef> 
#include <omp.h>
#include <algorithm>
#include <raylib.h>
#include "kernels.cuh"



// FIXME: Collusion handling still sucks
// FIXME: all the spatil lookup stuff including cell and hasing is 2d make it 3d
// PERF : Improve general optimization , reach 10K particles
// TODO : Add rotation
// TODO : More watery behaviour

float Engine::particle_radius = 0.2f;                   
float Engine::particle_color[4] = {0, 0, 1, 1};           

Engine::Engine() {
    h_positions = (float3 * )malloc(sizeof(float3) * simulation_size);

    container = box{Vector3{20,20,20},Vector3{-20,-20,-20}};
    
    particle_radius = 0.15f;
    
    particle_color[0] = 0.2f;
    particle_color[1] = 0.5f;
    particle_color[2] = 0.9f;
    particle_color[3] = 0.7f;
    
    particleMesh = GenMeshSphere(0.2f, 8, 8);
    
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
    free(h_positions);
    cudaFree(d_postions);
    cudaFree(d_velocities);
}

void Engine::init_cuda() {
    size_t size = sizeof(float3) * simulation_size;
    cudaMalloc((void**)&d_postions , size);
    cudaDeviceSynchronize();
    
}


void Engine::Draw() {
    for (int i = 0; i < simulation_size; i++) {
        DrawMesh(particleMesh, mat, transforms[i]);
    }
    container.Draw();

   
}


void Engine::Update() {
    // TODO : Copy positions to CUDA bi dakka
    
    // TODO : Launch kernels

    int threadsPerBlock = 256;
    int blocksPerGrid = (simulation_size + threadsPerBlock - 1) / threadsPerBlock;
    
    testkernel<<<blocksPerGrid, threadsPerBlock>>>(d_postions,simulation_size);
    cudaDeviceSynchronize();

    // TODO : Get positions from CUDA
    cudaMemcpy(h_positions,d_postions, simulation_size * sizeof(float3),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    // TODO : Update Transformations
    for (int i = 0; i < simulation_size; i++) {
        transforms[i] = MatrixTranslate(h_positions[i].x,h_positions[i].y, h_positions[i].z);
    }
}


void Engine::Reset() {
    simulation_size = 0;
}

void Engine::Populate() {
    int count = 0;
    for (int i = -10; i < 10; i+=1) {
        for (int j = -5; j < 5; j+=1) {
            for (int k = -5; k < 5; k+= 1) {
                h_positions[count].x = i;
                h_positions[count].y = j;
                h_positions[count].z = k;
                Matrix translation = MatrixTranslate(i, j, k);
                transforms.push_back( translation);
                count++;
            }
        }
    }
    
    simulation_size = count;
    transforms.resize(count);
    init_cuda();
}