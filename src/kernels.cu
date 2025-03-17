/*
TODO: define static variables ? shared mem
TODO: Denisty Kernel
TODO: Pressure Kernel
TODO: Collusion Kernel
TODO: Update Kernel

*/

__constant__ float smoothing_radius = 2.0f;
__constant__ float gravity = -25.0f;
__constant__ float target_density = 1.2f;
__constant__ float pressure_multiplier = 0.000001f * 3.0f;
__constant__ float treshold = 0.8f;

__device__ float dot(float3 v) {}              // Not NEEDED BUILTIN
__device__ float v3dist(flaot3 a, float3 b) {} // NOT NEEDED BUILTIN
__device__ float v3len(float3 v) {}            // NOT NEEDED BUILTIN

__global__ void testkernel(float3 *pos, int numpar) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numpar) {
    pos[idx].x += 0.1f;
  }
}

__global__ void computeDensity(float3 *pos, float *density, int numpar,
                               float h) {
  extern __shared__ float3 shared_pos[]; // Shared memory for neighbors

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numpar)
    return;

  float3 p_i = pos[idx];         // Current particle position
  shared_pos[threadIdx.x] = p_i; // Load into shared memory
  __syncthreads();               // Ensure all threads load data

  float rho = 0.0f;
  for (int j = 0; j < blockDim.x; j++) {
    float3 p_j = shared_pos[j];
    float dist = length(p_i - p_j); // Compute distance
    if (dist < h) {
      float q = dist / h;
      rho += (1.0f - q) * (1.0f - q); // TODO: DEGISTIR
    }
  }

  density[idx] = rho; // Store computed density
}

__global__ void resolve_collusions(float3 *pos, int numpar) {}

__global__ void pressure_kernel(float3 *pos, int numpar) {}
