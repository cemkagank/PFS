#include "kernels.cuh"
#include <curand_kernel.h>

__device__ __constant__ float smoothing_radius = 0.3f;
__device__ __constant__ float gravity = 9.81f;
__device__ __constant__ float REST_DENSITY = 1000.0f;
__device__ __constant__ float GAS_CONSTANT = 500.0f;
__device__ __constant__ float VISCOSITY = 50.0f;
__device__ __constant__ float PARTICLE_MASS = 0.05f;
__device__ __constant__ float MAX_VELOCITY = 5.0f;

// Spatial lookup constants
__device__ __constant__ int grid_size_x = 32;
__device__ __constant__ int grid_size_y = 32;
__device__ __constant__ int grid_size_z = 32;
__device__ __constant__ float grid_min_x = -10.0f;
__device__ __constant__ float grid_min_y = -10.0f;
__device__ __constant__ float grid_min_z = -10.0f;
__device__ __constant__ float grid_max_x = 10.0f;
__device__ __constant__ float grid_max_y = 10.0f;
__device__ __constant__ float grid_max_z = 10.0f;

// Helper function to get cell coordinates from position
__device__ int3 get_cell_coords(float3 pos) {
    float3 cell_size = make_float3(
        (grid_max_x - grid_min_x) / grid_size_x,
        (grid_max_y - grid_min_y) / grid_size_y,
        (grid_max_z - grid_min_z) / grid_size_z
    );
    
    int3 cell = make_int3(
        (int)((pos.x - grid_min_x) / cell_size.x),
        (int)((pos.y - grid_min_y) / cell_size.y),
        (int)((pos.z - grid_min_z) / cell_size.z)
    );
    
    // Clamp to grid bounds
    cell.x = max(0, min(cell.x, grid_size_x - 1));
    cell.y = max(0, min(cell.y, grid_size_y - 1));
    cell.z = max(0, min(cell.z, grid_size_z - 1));
    
    return cell;
}

// Hash function for 3D grid cell
__device__ int hash_cell(int3 cell) {
    return cell.x + cell.y * grid_size_x + cell.z * grid_size_x * grid_size_y;
}

// Build spatial lookup table
__global__ void build_spatial_lookup_kernel(float3* positions, int* spatial_lookup, int* start_indices, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    // Get cell coordinates for this particle
    int3 cell = get_cell_coords(positions[idx]);
    int hash = hash_cell(cell);
    
    // Store particle index and its cell hash
    spatial_lookup[idx] = hash;
    
    // Initialize start indices
    start_indices[idx] = -1;
}

// Update spatial lookup table
__global__ void update_spatial_lookup_kernel(float3* positions, int* spatial_lookup, int* start_indices, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    // Get cell coordinates for this particle
    int3 cell = get_cell_coords(positions[idx]);
    int hash = hash_cell(cell);
    
    // Update spatial lookup
    spatial_lookup[idx] = hash;
    
    // Update start indices
    if (idx == 0 || spatial_lookup[idx] != spatial_lookup[idx - 1]) {
        start_indices[hash] = idx;
    }
}

__device__ float distance(float3 a, float3 b) {
    float3 dist = make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    float dist_mag = sqrt(dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
    return dist_mag;
}

__device__ float3 scale_vector(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float smoothing_kernel(float dist) {
    if (dist >= smoothing_radius)
        return 0;
    float volume = M_PI * smoothing_radius * smoothing_radius * smoothing_radius / 6.0f;
    float q = dist / smoothing_radius;
    return (1.0f - q) * (1.0f - q) * (1.0f - q) / volume;
}

__device__ float smoothing_kernel_derivative(float dist) {
    if (dist >= smoothing_radius)
        return 0;
    float q = dist / smoothing_radius;
    float scale = -3.0f / (M_PI * smoothing_radius * smoothing_radius * smoothing_radius);
    return scale * (1.0f - q) * (1.0f - q);
}

__device__ float density_to_pressure(float density) {
    float gamma = 7.0f;
    return GAS_CONSTANT * (powf(density / REST_DENSITY, gamma) - 1.0f);
}

__global__ void density_kernel(float3* positions, float* densities, int* spatial_lookup, int* start_indices, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    float density = 0.0f;
    float3 pos = positions[idx];
    int3 cell = get_cell_coords(pos);
    
    // Check neighboring cells
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                int3 neighbor_cell = make_int3(
                    cell.x + x,
                    cell.y + y,
                    cell.z + z
                );
                
                // Skip if neighbor cell is out of bounds
                if (neighbor_cell.x < 0 || neighbor_cell.x >= grid_size_x ||
                    neighbor_cell.y < 0 || neighbor_cell.y >= grid_size_y ||
                    neighbor_cell.z < 0 || neighbor_cell.z >= grid_size_z) {
                    continue;
                }
                
                int neighbor_hash = hash_cell(neighbor_cell);
                int start_idx = start_indices[neighbor_hash];
                
                // Process particles in this cell
                while (start_idx != -1 && spatial_lookup[start_idx] == neighbor_hash) {
                    float dist = distance(pos, positions[start_idx]);
                    if (dist < smoothing_radius) {
                        density += PARTICLE_MASS * smoothing_kernel(dist);
                    }
                    start_idx++;
                }
            }
        }
    }
    
    densities[idx] = density;
}

__global__ void pressure_kernel(float3 * positions, float3 * velocities, float * densities, float3 * forces, float dt) {
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    
    __shared__ float3 poss[256];
    __shared__ float3 vels[256];
    __shared__ float dens[256];

    int i = threadIdx.x;
    int blockSize = blockDim.x;
    
    poss[i] = positions[blockIdx.x * blockSize + i];
    vels[i] = velocities[blockIdx.x * blockSize + i];
    dens[i] = densities[blockIdx.x * blockSize + i];

    __syncthreads();

    float pressure_i = density_to_pressure(dens[i]);

    // Add gravity first
    force.y -= gravity * PARTICLE_MASS;

    for(int j = 0; j < blockSize; j++) {
        if(i != j) {
            float dist_mag = distance(poss[i], poss[j]);
            if(dist_mag < smoothing_radius && dist_mag > 0.0f) {
                float3 diff = make_float3(
                    poss[i].x - poss[j].x,
                    poss[i].y - poss[j].y,
                    poss[i].z - poss[j].z
                );
                
                // Normalize the difference vector
                float inv_dist = 1.0f / dist_mag;
                diff.x *= inv_dist;
                diff.y *= inv_dist;
                diff.z *= inv_dist;
                
                // Pressure force
                float pressure_j = density_to_pressure(dens[j]);
                float3 pressure_grad = scale_vector(diff, smoothing_kernel_derivative(dist_mag));
                float pressure_force = (pressure_i + pressure_j) / (2.0f * dens[i]);
                force.x += pressure_force * pressure_grad.x;
                force.y += pressure_force * pressure_grad.y;
                force.z += pressure_force * pressure_grad.z;

                // Viscosity force
                float3 vel_diff = make_float3(
                    vels[i].x - vels[j].x,
                    vels[i].y - vels[j].y,
                    vels[i].z - vels[j].z
                );
                float viscosity_factor = VISCOSITY * PARTICLE_MASS / (dens[i] * dens[j]);
                force.x += viscosity_factor * vel_diff.x;
                force.y += viscosity_factor * vel_diff.y;
                force.z += viscosity_factor * vel_diff.z;
            }
        }
    }

    forces[blockIdx.x * blockSize + i] = force;
}

__global__ void update_positions_kernel(float3 * positions, float3 * velocities, float3 * forces, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Update velocity with smaller timestep for stability
    velocities[i].x += forces[i].x * dt * 0.05f;
    velocities[i].y += forces[i].y * dt * 0.05f;
    velocities[i].z += forces[i].z * dt * 0.05f;
    
    // Clamp velocities to prevent light speed
    float speed = sqrtf(velocities[i].x * velocities[i].x + 
                       velocities[i].y * velocities[i].y + 
                       velocities[i].z * velocities[i].z);
    if (speed > MAX_VELOCITY) {
        float scale = MAX_VELOCITY / speed;
        velocities[i].x *= scale;
        velocities[i].y *= scale;
        velocities[i].z *= scale;
    }
    
    // Apply damping
    const float damping = 0.99f;
    velocities[i].x *= damping;
    velocities[i].y *= damping;
    velocities[i].z *= damping;
    
    // Update position
    positions[i].x += velocities[i].x * dt;
    positions[i].y += velocities[i].y * dt;
    positions[i].z += velocities[i].z * dt;
    
    // Boundary conditions with water-like bounce
    const float boundary = 10.0f;  // Match container size
    const float bounce = 0.1f;     // Very small bounce
    const float friction = 0.2f;   // More friction
    const float min_velocity = 0.01f;  // Very small minimum velocity
    
    // X boundaries
    if(positions[i].x < -boundary) {
        positions[i].x = -boundary;
        if(fabsf(velocities[i].x) > min_velocity) {
            velocities[i].x = fabsf(velocities[i].x) * bounce;
        } else {
            velocities[i].x = 0.0f;
        }
        velocities[i].y *= (1.0f - friction);
        velocities[i].z *= (1.0f - friction);
    }
    if(positions[i].x > boundary) {
        positions[i].x = boundary;
        if(fabsf(velocities[i].x) > min_velocity) {
            velocities[i].x = -fabsf(velocities[i].x) * bounce;
        } else {
            velocities[i].x = 0.0f;
        }
        velocities[i].y *= (1.0f - friction);
        velocities[i].z *= (1.0f - friction);
    }
    
    // Y boundaries
    if(positions[i].y < -boundary) {
        positions[i].y = -boundary;
        if(fabsf(velocities[i].y) > min_velocity) {
            velocities[i].y = fabsf(velocities[i].y) * bounce;
        } else {
            velocities[i].y = 0.0f;
        }
        velocities[i].x *= (1.0f - friction);
        velocities[i].z *= (1.0f - friction);
    }
    if(positions[i].y > boundary) {
        positions[i].y = boundary;
        if(fabsf(velocities[i].y) > min_velocity) {
            velocities[i].y = -fabsf(velocities[i].y) * bounce;
        } else {
            velocities[i].y = 0.0f;
        }
        velocities[i].x *= (1.0f - friction);
        velocities[i].z *= (1.0f - friction);
    }
    
    // Z boundaries
    if(positions[i].z < -boundary) {
        positions[i].z = -boundary;
        if(fabsf(velocities[i].z) > min_velocity) {
            velocities[i].z = fabsf(velocities[i].z) * bounce;
        } else {
            velocities[i].z = 0.0f;
        }
        velocities[i].x *= (1.0f - friction);
        velocities[i].y *= (1.0f - friction);
    }
    if(positions[i].z > boundary) {
        positions[i].z = boundary;
        if(fabsf(velocities[i].z) > min_velocity) {
            velocities[i].z = -fabsf(velocities[i].z) * bounce;
        } else {
            velocities[i].z = 0.0f;
        }
        velocities[i].x *= (1.0f - friction);
        velocities[i].y *= (1.0f - friction);
    }
}






    


