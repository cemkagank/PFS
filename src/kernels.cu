#include "kernels.cuh"
#include <math.h>

// Helper functions
__device__ float smoothingKernel(float dist, float smoothingRadius) {
    if (dist >= smoothingRadius) return 0.0f;
    float q = dist / smoothingRadius;
    float volume = (M_PI * smoothingRadius * smoothingRadius * smoothingRadius) / 6.0f;
    return (1.0f - q) * (1.0f - q) * (1.0f - q) / volume;
}

__device__ float smoothingKernelDerivative(float dist, float smoothingRadius) {
    if (dist >= smoothingRadius) return 0.0f;
    float q = dist / smoothingRadius;
    float scale = 3.0f / (M_PI * smoothingRadius * smoothingRadius * smoothingRadius);
    return -scale * (1.0f - q) * (1.0f - q);
}

__device__ float densityToPressure(float density) {
    float densityError = density - REST_DENSITY;
    return GAS_CONSTANT * densityError * densityError;  // Quadratic pressure response
}

// Vector operations
__device__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float3 normalize(float3 v) {
    float len = length(v);
    if (len > 0.0f) {
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

// Spatial lookup helper
__device__ int getCellHash(int3 cell) {
    // Use a larger prime number to reduce hash collisions
    const int p1 = 73856093;
    const int p2 = 19349663;
    const int p3 = 83492791;
    const int p4 = 1234567891;
    return (cell.x * p1 ^ cell.y * p2 ^ cell.z * p3) % p4;
}

// Main kernels
__global__ void updateSpatialLookupKernel(
    float3* positions,
    int* spatialLookup,
    int* startIndices,
    int numParticles,
    float smoothingRadius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Calculate cell coordinates
    int3 cell = make_int3(
        (int)(positions[idx].x / smoothingRadius),
        (int)(positions[idx].y / smoothingRadius),
        (int)(positions[idx].z / smoothingRadius)
    );

    // Store particle index and cell hash
    spatialLookup[idx] = getCellHash(cell);
    startIndices[idx] = idx;
}

__global__ void sortParticlesKernel(
    int* spatialLookup,
    int* startIndices,
    int numParticles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Simple bubble sort within the block
    for (int i = 0; i < blockDim.x - 1; i++) {
        int j = idx + i;
        if (j >= numParticles - 1) break;
        
        if (spatialLookup[j] > spatialLookup[j + 1]) {
            // Swap hashes
            int tempHash = spatialLookup[j];
            spatialLookup[j] = spatialLookup[j + 1];
            spatialLookup[j + 1] = tempHash;
            
            // Swap indices
            int tempIdx = startIndices[j];
            startIndices[j] = startIndices[j + 1];
            startIndices[j + 1] = tempIdx;
        }
    }
}

__global__ void calculateDensityKernel(
    float3* positions,
    float* densities,
    int* spatialLookup,
    int* startIndices,
    int numParticles,
    float smoothingRadius
) {
    extern __shared__ float3 sharedPositions[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float3 pos = positions[idx];
    float density = 0.0f;

    // Calculate cell coordinates
    int3 cell = make_int3(
        (int)(pos.x / smoothingRadius),
        (int)(pos.y / smoothingRadius),
        (int)(pos.z / smoothingRadius)
    );

    // Pre-calculate cell hash for current particle
    int currentCellHash = getCellHash(cell);

    // Pre-allocate arrays for neighboring cells
    int neighborHashes[27];  // 3x3x3 grid
    int neighborCount = 0;

    // Pre-calculate all neighbor cell hashes
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                int3 neighborCell = make_int3(cell.x + x, cell.y + y, cell.z + z);
                neighborHashes[neighborCount++] = getCellHash(neighborCell);
            }
        }
    }

    // Process all neighbor cells
    for (int n = 0; n < 27; n++) {
        int cellHash = neighborHashes[n];
        
        // Find the range of particles in this cell
        int left = 0;
        int right = numParticles - 1;
        int start = -1;
        int end = -1;

        // Binary search for the start of this cell's particles
        while (left <= right) {
            int mid = (left + right) / 2;
            if (spatialLookup[mid] == cellHash) {
                start = mid;
                right = mid - 1;
            } else if (spatialLookup[mid] < cellHash) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        // If we found the start, find the end
        if (start != -1) {
            left = start;
            right = numParticles - 1;
            while (left <= right) {
                int mid = (left + right) / 2;
                if (spatialLookup[mid] == cellHash) {
                    end = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }

            // Load particles into shared memory in chunks
            int numParticlesInCell = end - start + 1;
            int numChunks = (numParticlesInCell + blockDim.x - 1) / blockDim.x;
            
            for (int chunk = 0; chunk < numChunks; chunk++) {
                int chunkStart = start + chunk * blockDim.x;
                int chunkEnd = min(chunkStart + blockDim.x, end + 1);
                
                // Load particles into shared memory
                for (int i = chunkStart + threadIdx.x; i < chunkEnd; i += blockDim.x) {
                    int particleIdx = startIndices[i];
                    sharedPositions[threadIdx.x] = positions[particleIdx];
                }
                __syncthreads();

                // Process particles in shared memory
                for (int i = 0; i < min(blockDim.x, chunkEnd - chunkStart); i++) {
                    float3 diff = make_float3(
                        pos.x - sharedPositions[i].x,
                        pos.y - sharedPositions[i].y,
                        pos.z - sharedPositions[i].z
                    );
                    float dist = length(diff);
                    if (dist < smoothingRadius) {
                        density += smoothingKernel(dist, smoothingRadius);
                    }
                }
                __syncthreads();
            }
        }
    }

    densities[idx] = density * REST_DENSITY;
}

__global__ void calculatePressureForceKernel(
    float3* positions,
    float3* velocities,
    float* densities,
    float3* pressures,
    int* spatialLookup,
    int* startIndices,
    int numParticles,
    float smoothingRadius
) {
    extern __shared__ float3 sharedData[];
    float3* sharedPositions = sharedData;
    float3* sharedVelocities = &sharedData[blockDim.x];
    float* sharedDensities = (float*)&sharedData[2 * blockDim.x];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float3 pos = positions[idx];
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float density = densities[idx];
    float pressure = densityToPressure(density);

    // Calculate cell coordinates
    int3 cell = make_int3(
        (int)(pos.x / smoothingRadius),
        (int)(pos.y / smoothingRadius),
        (int)(pos.z / smoothingRadius)
    );

    // Pre-calculate cell hash for current particle
    int currentCellHash = getCellHash(cell);

    // Pre-allocate arrays for neighboring cells
    int neighborHashes[27];  // 3x3x3 grid
    int neighborCount = 0;

    // Pre-calculate all neighbor cell hashes
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                int3 neighborCell = make_int3(cell.x + x, cell.y + y, cell.z + z);
                neighborHashes[neighborCount++] = getCellHash(neighborCell);
            }
        }
    }

    // Process all neighbor cells
    for (int n = 0; n < 27; n++) {
        int cellHash = neighborHashes[n];
        
        // Find the range of particles in this cell
        int left = 0;
        int right = numParticles - 1;
        int start = -1;
        int end = -1;

        // Binary search for the start of this cell's particles
        while (left <= right) {
            int mid = (left + right) / 2;
            if (spatialLookup[mid] == cellHash) {
                start = mid;
                right = mid - 1;
            } else if (spatialLookup[mid] < cellHash) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        // If we found the start, find the end
        if (start != -1) {
            left = start;
            right = numParticles - 1;
            while (left <= right) {
                int mid = (left + right) / 2;
                if (spatialLookup[mid] == cellHash) {
                    end = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }

            // Load particles into shared memory in chunks
            int numParticlesInCell = end - start + 1;
            int numChunks = (numParticlesInCell + blockDim.x - 1) / blockDim.x;
            
            for (int chunk = 0; chunk < numChunks; chunk++) {
                int chunkStart = start + chunk * blockDim.x;
                int chunkEnd = min(chunkStart + blockDim.x, end + 1);
                
                // Load particles into shared memory
                for (int i = chunkStart + threadIdx.x; i < chunkEnd; i += blockDim.x) {
                    int particleIdx = startIndices[i];
                    sharedPositions[threadIdx.x] = positions[particleIdx];
                    sharedVelocities[threadIdx.x] = velocities[particleIdx];
                    sharedDensities[threadIdx.x] = densities[particleIdx];
                }
                __syncthreads();

                // Process particles in shared memory
                for (int i = 0; i < min(blockDim.x, chunkEnd - chunkStart); i++) {
                    int particleIdx = startIndices[chunkStart + i];
                    if (particleIdx != idx) {
                        float3 diff = make_float3(
                            pos.x - sharedPositions[i].x,
                            pos.y - sharedPositions[i].y,
                            pos.z - sharedPositions[i].z
                        );
                        float dist = length(diff);
                        if (dist < smoothingRadius && dist > 0.0f) {
                            float3 dir = normalize(diff);
                            float neighborPressure = densityToPressure(sharedDensities[i]);
                            float pressureForce = (pressure + neighborPressure) * 
                                smoothingKernelDerivative(dist, smoothingRadius);
                            
                            // Add viscosity
                            float3 velDiff = make_float3(
                                velocities[idx].x - sharedVelocities[i].x,
                                velocities[idx].y - sharedVelocities[i].y,
                                velocities[idx].z - sharedVelocities[i].z
                            );
                            float viscosityForce = VISCOSITY * smoothingKernel(dist, smoothingRadius);
                            
                            force.x += dir.x * pressureForce + velDiff.x * viscosityForce;
                            force.y += dir.y * pressureForce + velDiff.y * viscosityForce;
                            force.z += dir.z * pressureForce + velDiff.z * viscosityForce;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    // Scale the force by the inverse of density
    if (density > 0.0f) {
        force.x /= density;
        force.y /= density;
        force.z /= density;
    }

    pressures[idx] = force;
}

__global__ void updatePositionsKernel(
    float3* positions,
    float3* velocities,
    float3* pressures,
    float3 containerMin,
    float3 containerMax,
    float gravity,
    float deltaTime,
    int numParticles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Update velocity
    velocities[idx].x += pressures[idx].x * deltaTime;
    velocities[idx].y += (pressures[idx].y + gravity) * deltaTime;
    velocities[idx].z += pressures[idx].z * deltaTime;

    // Apply damping
    velocities[idx].x *= DAMPING;
    velocities[idx].y *= DAMPING;
    velocities[idx].z *= DAMPING;

    // Update position
    positions[idx].x += velocities[idx].x * deltaTime;
    positions[idx].y += velocities[idx].y * deltaTime;
    positions[idx].z += velocities[idx].z * deltaTime;

    // Boundary collision with proper bounce and friction
    const float bounce = 0.2f;
    const float friction = 0.1f;
    
    if (positions[idx].x < containerMin.x) {
        positions[idx].x = containerMin.x;
        velocities[idx].x = fabsf(velocities[idx].x) * bounce;
        velocities[idx].y *= (1.0f - friction);
        velocities[idx].z *= (1.0f - friction);
    }
    if (positions[idx].x > containerMax.x) {
        positions[idx].x = containerMax.x;
        velocities[idx].x = -fabsf(velocities[idx].x) * bounce;
        velocities[idx].y *= (1.0f - friction);
        velocities[idx].z *= (1.0f - friction);
    }
    if (positions[idx].y < containerMin.y) {
        positions[idx].y = containerMin.y;
        velocities[idx].y = fabsf(velocities[idx].y) * bounce;
        velocities[idx].x *= (1.0f - friction);
        velocities[idx].z *= (1.0f - friction);
    }
    if (positions[idx].y > containerMax.y) {
        positions[idx].y = containerMax.y;
        velocities[idx].y = -fabsf(velocities[idx].y) * bounce;
        velocities[idx].x *= (1.0f - friction);
        velocities[idx].z *= (1.0f - friction);
    }
    if (positions[idx].z < containerMin.z) {
        positions[idx].z = containerMin.z;
        velocities[idx].z = fabsf(velocities[idx].z) * bounce;
        velocities[idx].x *= (1.0f - friction);
        velocities[idx].y *= (1.0f - friction);
    }
    if (positions[idx].z > containerMax.z) {
        positions[idx].z = containerMax.z;
        velocities[idx].z = -fabsf(velocities[idx].z) * bounce;
        velocities[idx].x *= (1.0f - friction);
        velocities[idx].y *= (1.0f - friction);
    }
} 