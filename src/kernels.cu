__global__ void testkernel(float3* pos, int numpar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numpar) {
        pos[idx].x += 0.1f;
    }
}