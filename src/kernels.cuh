#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <vector_types.h>

// Constants
#define TARGET_DENSITY 1.0f
#define STIFFNESS 0.00001f
#define SMOOTHING_RADIUS 2.0f
#define VISCOSITY 0.1f
#define DAMPING 0.99f
#define REST_DENSITY 1.0f
#define GAS_CONSTANT 20.0f

// Helper functions
__device__ float smoothingKernel(float dist, float smoothingRadius);
__device__ float smoothingKernelDerivative(float dist, float smoothingRadius);
__device__ float densityToPressure(float density);
__device__ float length(float3 v);
__device__ float3 normalize(float3 v);
__device__ int getCellHash(int3 cell);

// Main kernels
__global__ void updateSpatialLookupKernel(
    float3* positions,
    int* spatialLookup,
    int* startIndices,
    int numParticles,
    float smoothingRadius
);

__global__ void sortParticlesKernel(
    int* spatialLookup,
    int* startIndices,
    int numParticles
);

__global__ void calculateDensityKernel(
    float3* positions,
    float* densities,
    int* spatialLookup,
    int* startIndices,
    int numParticles,
    float smoothingRadius
);

__global__ void calculatePressureForceKernel(
    float3* positions,
    float3* velocities,
    float* densities,
    float3* pressures,
    int* spatialLookup,
    int* startIndices,
    int numParticles,
    float smoothingRadius
);

__global__ void updatePositionsKernel(
    float3* positions,
    float3* velocities,
    float3* pressures,
    float3 containerMin,
    float3 containerMax,
    float gravity,
    float deltaTime,
    int numParticles
);

#endif // KERNELS_CUH 