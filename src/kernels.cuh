#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

// Device constants
extern __device__ __constant__ float smoothing_radius;
extern __device__ __constant__ float gravity;

// Helper functions
__device__ float distance(float3 a, float3 b);
__device__ float3 scale_vector(float3 a, float b);
__device__ float smoothing_kernel(float dist);
__device__ float smoothing_kernel_derivative(float dist);
__device__ float shared_pressure(float dens1, float dens2);

// Main kernels
__global__ void density_kernel(float3* positions, float* densities, int* spatial_lookup, int* start_indices, int num_particles);
__global__ void pressure_kernel(float3* positions, float3* velocities, float* densities, float3* forces, float dt);
__global__ void update_positions_kernel(float3* positions, float3* velocities, float3* forces, float dt);

// Spatial lookup kernels
__global__ void build_spatial_lookup_kernel(float3* positions, int* spatial_lookup, int* start_indices, int num_particles);
__global__ void update_spatial_lookup_kernel(float3* positions, int* spatial_lookup, int* start_indices, int num_particles); 