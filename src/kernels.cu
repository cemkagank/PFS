#include "kernels.cuh"
#include <curand_kernel.h>

// Base constants
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

// Pre-computed constants (using static values)
__device__ __constant__ float smoothing_radius_squared = 0.09f;  // 0.3 * 0.3
__device__ __constant__ float smoothing_radius_cubed = 0.027f;   // 0.3 * 0.3 * 0.3
__device__ __constant__ float smoothing_radius_fourth = 0.0081f; // 0.3 * 0.3 * 0.3 * 0.3

// Optimized distance calculation without sqrt when possible
__device__ float distance_squared(float3 a, float3 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

__device__ float distance(float3 a, float3 b) {
    return sqrtf(distance_squared(a, b));
}

__device__ float3 scale_vector(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

// Optimized smoothing kernel using pre-computed constants
__device__ float smoothing_kernel(float dist) {
    if (dist >= smoothing_radius)
        return 0;
    float volume = M_PI * smoothing_radius_cubed / 6.0f;
    float q = dist / smoothing_radius;
    float q2 = q * q;
    float q3 = q2 * q;
    return (1.0f - q3) / volume;
}

// Optimized smoothing kernel derivative
__device__ float smoothing_kernel_derivative(float dist) {
    if (dist >= smoothing_radius)
        return 0;
    float q = dist / smoothing_radius;
    float scale = -3.0f / (M_PI * smoothing_radius_cubed);
    float q2 = q * q;
    return scale * (1.0f - q2);
}

__device__ float density_to_pressure(float density) {
    float gamma = 7.0f;
    return GAS_CONSTANT * (powf(density / REST_DENSITY, gamma) - 1.0f);
}

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

// Optimized density kernel with shared memory and improved memory access
__global__ void density_kernel(float3* positions, float* densities, int* spatial_lookup, int* start_indices, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    float density = 0.0f;
    float3 pos = positions[idx];
    int3 cell = get_cell_coords(pos);
    
    // Use shared memory for frequently accessed data
    __shared__ float3 shared_positions[256];
    __shared__ float shared_densities[256];
    
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
                    float dist_squared = distance_squared(pos, positions[start_idx]);
                    if (dist_squared < smoothing_radius_squared) {
                        float dist = sqrtf(dist_squared);
                        density += PARTICLE_MASS * smoothing_kernel(dist);
                    }
                    start_idx++;
                }
            }
        }
    }
    
    densities[idx] = density;
}

// Optimized pressure kernel with shared memory and improved memory access
__global__ void pressure_kernel(float3* positions, float3* velocities, float* densities, float3* forces, 
                              int* spatial_lookup, int* start_indices, int num_particles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float3 pos = positions[idx];
    float3 vel = velocities[idx];
    float density = densities[idx];
    float pressure_i = density_to_pressure(density);
    
    // Add gravity first
    force.y -= gravity * PARTICLE_MASS;
    
    // Check neighboring cells
    int3 cell = get_cell_coords(pos);
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
                    if (start_idx != idx) {
                        float dist_squared = distance_squared(pos, positions[start_idx]);
                        if (dist_squared < smoothing_radius_squared && dist_squared > 0.0f) {
                            float dist = sqrtf(dist_squared);
                            float3 diff = make_float3(
                                pos.x - positions[start_idx].x,
                                pos.y - positions[start_idx].y,
                                pos.z - positions[start_idx].z
                            );
                            
                            // Normalize the difference vector
                            float inv_dist = 1.0f / dist;
                            diff.x *= inv_dist;
                            diff.y *= inv_dist;
                            diff.z *= inv_dist;
                            
                            // Pressure force
                            float pressure_j = density_to_pressure(densities[start_idx]);
                            float3 pressure_grad = scale_vector(diff, smoothing_kernel_derivative(dist));
                            float pressure_force = (pressure_i + pressure_j) / (2.0f * density);
                            force.x += pressure_force * pressure_grad.x;
                            force.y += pressure_force * pressure_grad.y;
                            force.z += pressure_force * pressure_grad.z;
                            
                            // Viscosity force
                            float3 vel_diff = make_float3(
                                vel.x - velocities[start_idx].x,
                                vel.y - velocities[start_idx].y,
                                vel.z - velocities[start_idx].z
                            );
                            float viscosity_factor = VISCOSITY * PARTICLE_MASS / (density * densities[start_idx]);
                            force.x += viscosity_factor * vel_diff.x;
                            force.y += viscosity_factor * vel_diff.y;
                            force.z += viscosity_factor * vel_diff.z;
                        }
                    }
                    start_idx++;
                }
            }
        }
    }
    
    forces[idx] = force;
}

// Optimized position update kernel
__global__ void update_positions_kernel(float3* positions, float3* velocities, float3* forces, int num_particles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    // Update velocity with smaller timestep for stability
    float3 vel = velocities[idx];
    float3 force = forces[idx];
    
    vel.x += force.x * dt * 0.05f;
    vel.y += force.y * dt * 0.05f;
    vel.z += force.z * dt * 0.05f;
    
    // Clamp velocities to prevent light speed
    float speed_squared = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    if (speed_squared > MAX_VELOCITY * MAX_VELOCITY) {
        float scale = MAX_VELOCITY / sqrtf(speed_squared);
        vel.x *= scale;
        vel.y *= scale;
        vel.z *= scale;
    }
    
    // Apply damping
    const float damping = 0.99f;
    vel.x *= damping;
    vel.y *= damping;
    vel.z *= damping;
    
    // Update position
    float3 pos = positions[idx];
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
    
    // Boundary conditions with water-like bounce
    const float boundary = 10.0f;  // Match container size
    const float bounce = 0.1f;     // Very small bounce
    const float friction = 0.2f;   // More friction
    const float min_velocity = 0.01f;  // Very small minimum velocity
    
    // X boundaries
    if(pos.x < -boundary) {
        pos.x = -boundary;
        if(fabsf(vel.x) > min_velocity) {
            vel.x = fabsf(vel.x) * bounce;
        } else {
            vel.x = 0.0f;
        }
        vel.y *= (1.0f - friction);
        vel.z *= (1.0f - friction);
    }
    if(pos.x > boundary) {
        pos.x = boundary;
        if(fabsf(vel.x) > min_velocity) {
            vel.x = -fabsf(vel.x) * bounce;
        } else {
            vel.x = 0.0f;
        }
        vel.y *= (1.0f - friction);
        vel.z *= (1.0f - friction);
    }
    
    // Y boundaries
    if(pos.y < -boundary) {
        pos.y = -boundary;
        if(fabsf(vel.y) > min_velocity) {
            vel.y = fabsf(vel.y) * bounce;
        } else {
            vel.y = 0.0f;
        }
        vel.x *= (1.0f - friction);
        vel.z *= (1.0f - friction);
    }
    if(pos.y > boundary) {
        pos.y = boundary;
        if(fabsf(vel.y) > min_velocity) {
            vel.y = -fabsf(vel.y) * bounce;
        } else {
            vel.y = 0.0f;
        }
        vel.x *= (1.0f - friction);
        vel.z *= (1.0f - friction);
    }
    
    // Z boundaries
    if(pos.z < -boundary) {
        pos.z = -boundary;
        if(fabsf(vel.z) > min_velocity) {
            vel.z = fabsf(vel.z) * bounce;
        } else {
            vel.z = 0.0f;
        }
        vel.x *= (1.0f - friction);
        vel.y *= (1.0f - friction);
    }
    if(pos.z > boundary) {
        pos.z = boundary;
        if(fabsf(vel.z) > min_velocity) {
            vel.z = -fabsf(vel.z) * bounce;
        } else {
            vel.z = 0.0f;
        }
        vel.x *= (1.0f - friction);
        vel.y *= (1.0f - friction);
    }
    
    // Write back results
    positions[idx] = pos;
    velocities[idx] = vel;
}






    


