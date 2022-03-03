__device__ uint rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) |
           (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

__global__ void render(uint *out, uint imageW, uint imageH)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float4 col = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    out[y * imageW + x] = rgbaFloatToInt(col);
}

extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint* output, uint imageW, uint imageH)
{
    render<<<gridSize,blockSize>>>(output, imageW, imageH);
}