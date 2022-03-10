#include "helper_math.h"
#include "helper_cuda.h"

#ifndef NV_PI
#define NV_PI   float(3.1415926535897932384626433832795)
#endif

cudaArray *volumeArray = 0;

typedef unsigned char VolumeType;

cudaTextureObject_t texObject;    // For 3D texture

typedef struct { float4 m[3]; } float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

__device__ float3 light;

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

__device__ uint rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) |
           (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
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

__device__ float getScatteredLight(float3 d, float3 light)
{
    float g = 0.7f;
    float cosAngle = dot(light, d)/length(light)/length(d);
    float c = 0.25f/ NV_PI;

    return c * (1 - g*g)/(1 + g*g -2*g*cosAngle);
}

__global__ void render(uint *out, uint imageW, uint imageH, cudaTextureObject_t tex)
{
    const int maxSteps = 500;
    const float tStep = 0.05f;

    const float3 boxMin = make_float3(-1.0f);
    const float3 boxMax = make_float3(1.0f);

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float u = (x / (float)imageW) * 2.0f - 1.0f;
    float v = (y / (float)imageH) * 2.0f - 1.0f;

    light = make_float3(-1.0f, 1.0f, -1.0f);

    Ray eye;
    eye.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eye.d = normalize(make_float3(u, v, -2.0f));
    eye.d = mul(c_invViewMatrix, eye.d);

    // A sky rendering with sun as smooth gradient
    float colorStep = smoothstep(0.95f, 0.99f, dot(light, -eye.d)/length(light)/length(eye.d));
    const float4 background = make_float4(colorStep, 0.6+colorStep*0.4f, 1.0f, 1.0f);

    float tNear, tFar;

    int hit = intersectBox(eye, boxMin, boxMax, &tNear, &tFar);

    if(!hit) 
    {
        out[y * imageW + x] = rgbaFloatToInt(background);
        return;
    }
    
    if (tNear < 0.0f) tNear = 0.0f;  // clamp to near plane

    //float3 frontPos = eye.o + eye.d * tNear;
    float3 pos = eye.o + eye.d * tFar;
    float3 step = -eye.d * tStep;

    float4 sum = make_float4(0.0f);
    float t = tFar;

    for (int i = 0; i < maxSteps; i++)
    {
        float density = tex3D<float>(tex, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f,
                                     pos.z * 0.5f + 0.5f);
        
        
        float4 col = make_float4(getScatteredLight(-eye.d, light));
        col *= density;

        sum += col * (1 - sum.w);

        t -= tStep;
        if (t < tNear)
            break;
        
        pos += step;
    }

    out[y * imageW + x] = rgbaFloatToInt(background + sum);
}

extern "C" void initCuda(void *volume, cudaExtent volumeSize)
{
    // Create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&volumeArray, &channelDesc, volumeSize));

    // Copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_cudaPitchedPtr(volume, volumeSize.width * sizeof(VolumeType),
                            volumeSize.width, volumeSize.height);
    copyParams.dstArray = volumeArray;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = volumeArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords =
        true;  // access with normalized texture coordinates
    texDescr.filterMode = cudaFilterModeLinear;  // linear interpolation

    texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;

    texDescr.readMode = cudaReadModeNormalizedFloat;

    checkCudaErrors(
        cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));
}

extern "C" void freeCudaBuffers()
{
    checkCudaErrors(cudaDestroyTextureObject(texObject));
    checkCudaErrors(cudaFreeArray(volumeArray));
}

extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint* output, uint imageW, uint imageH)
{
    render<<<gridSize,blockSize>>>(output, imageW, imageH, texObject);
}

extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeOfMatrix)
{
    checkCudaErrors(
        cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeOfMatrix)
    );
}