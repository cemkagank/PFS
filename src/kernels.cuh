#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

// Device constants
extern __device__ __constant__ float smoothing_radius;
extern __device__ __constant__ float gravity;
extern __device__ __constant__ float REST_DENSITY;
extern __device__ __constant__ float GAS_CONSTANT;
extern __device__ __constant__ float VISCOSITY;
extern __device__ __constant__ float PARTICLE_MASS;
extern __device__ __constant__ float MAX_VELOCITY;

// Spatial lookup constants
extern __device__ __constant__ int grid_size_x;
extern __device__ __constant__ int grid_size_y;
extern __device__ __constant__ int grid_size_z;
extern __device__ __constant__ float grid_min_x;
extern __device__ __constant__ float grid_min_y;
extern __device__ __constant__ float grid_min_z;
extern __device__ __constant__ float grid_max_x;
extern __device__ __constant__ float grid_max_y;
extern __device__ __constant__ float grid_max_z;

// Pre-computed constants
extern __device__ __constant__ float smoothing_radius_squared;
extern __device__ __constant__ float smoothing_radius_cubed;
extern __device__ __constant__ float smoothing_radius_fourth;

// Helper functions
__device__ float distance_squared(float3 a, float3 b);
__device__ float distance(float3 a, float3 b);
__device__ float3 scale_vector(float3 a, float b);
__device__ float smoothing_kernel(float dist);
__device__ float smoothing_kernel_derivative(float dist);
__device__ float density_to_pressure(float density);
__device__ int3 get_cell_coords(float3 pos);
__device__ int hash_cell(int3 cell);
__device__ float shared_pressure(float dens1, float dens2);

// Main kernels
__global__ void density_kernel(float3* positions, float* densities, int* spatial_lookup, int* start_indices, int num_particles);
__global__ void pressure_kernel(float3* positions, float3* velocities, float* densities, float3* forces, 
                              int* spatial_lookup, int* start_indices, int num_particles, float dt);
__global__ void update_positions_kernel(float3* positions, float3* velocities, float3* forces, int num_particles, float dt);

// Spatial lookup kernels
__global__ void build_spatial_lookup_kernel(float3* positions, int* spatial_lookup, int* start_indices, int num_particles);
__global__ void update_spatial_lookup_kernel(float3* positions, int* spatial_lookup, int* start_indices, int num_particles); 