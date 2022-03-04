#include "helper_math.h"
#include "helper_cuda.h"
__device__ uint rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) |
           (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

typedef struct { float4 m[3]; } float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray {
  float3 o;  // origin
  float3 d;  // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__ int intersectBox(Ray r, float3 boxMin, float3 boxMax, float *tNear, float *tFar)
{
    // Compute min and max intersections
    float3 invR = make_float3(1.0f) / r.d;
    float3 tBottom = invR * (boxMin - r.o);
    float3 tTop = invR * (boxMax - r.o);

    // Reorder intersections to fing min and max on each axis
    float3 tMin = fminf(tTop, tBottom);
    float3 tMax = fmaxf(tTop, tBottom);

    // Find largest tMin and smallest tMax,
    // respectively corresponding to close and far intersections
    float largest_tMin = fmaxf(fmaxf(tMin.x, tMin.y), fmaxf(tMin.x, tMin.z));
    float smallest_tMax= fminf(fminf(tMax.x, tMax.y), fminf(tMax.x, tMax.z));

    *tNear = largest_tMin;
    *tFar = smallest_tMax;

    return smallest_tMax > largest_tMin;
}

// transform vector by matrix (no translation)
__device__ float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}
  
// transform vector by matrix with translation
__device__ float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ float3 clampToMax(float3 in)
{
    float3 out;
    float max = fmaxf(fmaxf(in.x, in.y), fmaxf(in.x, in.z));

    out.x = (in.x == max) ? 1.0f : 0.0f;
    out.y = (in.y == max) ? 1.0f : 0.0f;
    out.z = (in.z == max) ? 1.0f : 0.0f;

    return out;
}

__global__ void render(uint *out, uint imageW, uint imageH)
{
    const float3 boxMin = make_float3(-1.0f);
    const float3 boxMax = make_float3(1.0f);

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float u = (x / (float)imageW) * 2.0f - 1.0f;
    float v = (y / (float)imageH) * 2.0f - 1.0f;

    Ray eye;
    eye.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eye.d = normalize(make_float3(u, v, -2.0f));
    eye.d = mul(c_invViewMatrix, eye.d);

    float tNear, tFar;

    int hit = intersectBox(eye, boxMin, boxMax, &tNear, &tFar);

    if(!hit) return;

    float3 frontPos = eye.o + eye.d * tNear;
    float3 backPos = eye.o + eye.d * tFar;
    float length = tFar - tNear;


    float3 frontColor = clampToMax(fabs(frontPos));
    float3 backColor = clampToMax(fabs(backPos));
    float4 col = make_float4(frontColor);
    if (length < 2.0f)
    {
        float a = length/2.0f;
        col = make_float4(a*frontColor + (1-a)*backColor);
    }

    out[y * imageW + x] = rgbaFloatToInt(col);
}

extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint* output, uint imageW, uint imageH)
{
    render<<<gridSize,blockSize>>>(output, imageW, imageH);
}

extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeOfMatrix)
{
    checkCudaErrors(
        cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeOfMatrix)
    );
}