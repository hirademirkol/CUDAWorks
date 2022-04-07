#ifndef _BOIDS_KERNEL_H_
#define _BOIDS_KERNEL_H_

#include "helper_math.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "sim_params.cuh"

typedef unsigned int uint;

// Simulation parameters in the constant CUDA memory
__constant__ SimParams params;

struct integrate_functor
{
    float _deltaTime;
    
    __host__ __device__ integrate_functor(float deltaTime) : _deltaTime(deltaTime){}

    template <typename Tuple>
    __device__ void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);

        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);

        pos += vel * _deltaTime;

        // Boids go through bounds to the other side
        if(pos.x < -1.0f)
            pos.x = 1.0f;
        if(pos.y < -1.0f)
            pos.y = 1.0f;
        if(pos.z < -1.0f)
            pos.z = 1.0f;

        if(pos.x > 1.0f)
            pos.x = -1.0f;
        if(pos.y > 1.0f)
            pos.y = -1.0f;
        if(pos.z > 1.0f)
            pos.z = -1.0f;

        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};

// Calculate in which cell the boid is
__device__ int3 calcGridPos(float3 pos)
{
    int3 gridPos;
    
    gridPos.x = floorf((pos.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floorf((pos.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floorf((pos.z - params.worldOrigin.z) / params.cellSize.z);

    return gridPos;
}

// Calculate hash for a grid cell
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x - 1);
    gridPos.y = gridPos.y & (params.gridSize.y - 1);
    gridPos.z = gridPos.z & (params.gridSize.z - 1);

    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) +
           __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// Calculate cell hash value for each boid
__global__ void calcHash_kernel(uint *gridParticleHash,
                          uint *gridParticleIndex,
                          float4 *pos,
                          uint numBoids)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if(index >= numBoids) return;

    volatile float4 _pos = pos[index];

    int3 gridPos = calcGridPos(make_float3(_pos.x, _pos.y, _pos.z));
    uint hash = calcGridHash(gridPos);

    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;    
}

__global__ void reorderDataAndFindCellStart_kernel(
    uint *cellStart,          // output: cell start index
    uint *cellEnd,            // output: cell end index
    float4 *sortedPos,        // output: sorted positions
    float4 *sortedVel,        // output: sorted velocities
    float4 *sortedUp,          // output: sorted up vectors
    uint *gridParticleHash,   // input: sorted grid hashes
    uint *gridParticleIndex,  // input: sorted particle indices
    float4 *oldPos,           // input: sorted position array
    float4 *oldVel,           // input: sorted velocity array
    float4 *oldUp,             // input: sorted up vector array
    uint numBoids) 
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[];  // blockSize + 1 elements
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    uint hash;

    // Handle case when no. of boids is not multiple of block size
    if (index < numBoids) 
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0) 
        {
            // First thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index - 1];
        }
    }

    // Sync threads to ensure that the shared data is completed
    cg::sync(cta);

    if (index < numBoids) {
    // If this particle has a different cell index to the previous
    // particle then it must be the first particle in the cell,
    // so store the index of this particle in the cell.
    // As it isn't the first particle, it must also be the cell end of
    // the previous particle's cell

    if (index == 0 || hash != sharedHash[threadIdx.x])
    {
        cellStart[hash] = index;

        if (index > 0) cellEnd[sharedHash[threadIdx.x]] = index;
    }

    if (index == numBoids - 1)
    {
        cellEnd[hash] = index + 1;
    }

    // Now use the sorted index to reorder the pos and vel data
    uint sortedIndex = gridParticleIndex[index];
    float4 pos = oldPos[sortedIndex];
    float4 vel = oldVel[sortedIndex];
    float4 up = oldUp[sortedIndex];

    sortedPos[index] = pos;
    sortedVel[index] = vel;
    sortedUp[index] = up;
    }
}

// Calculate avoidance effect of boid 2 to boid 1
__device__ float3 calcAvoid(float3 deltaPos, uint* avoidAmount)
{
    float3 effect = make_float3(0.0f);

    // Calculate only if the boid 2 is in avoidance region
    if(length(deltaPos) < params.avoidLength)
    {
        effect = deltaPos;
        // Atomically add the boid to the number of avoided boids, as the reference is also used by other threads
        atomicAdd(avoidAmount, 1);
    }

    return effect;
}

// Calculate alignment effect of boid 2 to boid 1
__device__ float3 calcAlign(float3 deltaPos, float3 deltaVel, uint* alignAmount)
{
    float3 effect = make_float3(0.0f);

    // Calculate only if the boid 2 is in alignment region and out of avoidance region
    if(length(deltaPos) > params.avoidLength && length(deltaPos) < params.alignLength)
    {
        effect = -deltaVel;
        // Atomically add the boid to the number of aligned boids, as the reference is also used by other threads
        atomicAdd(alignAmount, 1);
    }

    return effect;
}

// Calculate cohesion effect of boid 2 to boid 1
__device__ float3 calcCohesion(float3 deltaPos, uint* cohesionAmount)
{
    float3 effect = make_float3(0.0f);

    // Calculate only if the boid 2 is in alignment region and out of avoidance region
    if(length(deltaPos) > params.avoidLength && length(deltaPos) < params.alignLength)
    {
        effect = -deltaPos;
        // Atomically add the boid to the number of coheisoned boids, as the reference is also used by other threads
        atomicAdd(cohesionAmount, 1);
    }

    return effect;
}

// Atomically add each element of a float3 vector to another
__device__ void atomicAddv(float3 *a, float3 b)
{
    atomicAdd(&a->x, b.x);
    atomicAdd(&a->y, b.y);
    atomicAdd(&a->z, b.z);

}

// Calculate interactions of a boid with other boids in a cell
__device__ void interactWithCell(int3 gridPos, uint index, float3 pos, float3 vel,
                                 float4 *oldPos, float4 *oldVel,
                                 uint *cellStart, uint *cellEnd,
                                 float3 *avoidVel, uint *avoidAmount,
                                 float3 *alignVel, uint *alignAmount,
                                 float3 *cohesionVel, uint *cohesionAmount)
{
    uint gridHash = calcGridHash(gridPos);

    uint startIndex = cellStart[gridHash];
    
    if(startIndex != 0xffffffff)
    {
        uint endIndex = cellEnd[gridHash];

        // Find start and end boids of a cell and looop between them
        for(uint j = startIndex; j < endIndex; j++)
        {
            if(j != index) // Don't calculate for itself
            {
                float3 deltaPos = pos - make_float3(oldPos[j]);
                // Check whether boids are closer from the other side
                if(deltaPos.x > 1.0f) deltaPos.x = 2.0f - deltaPos.x;
                if(deltaPos.y > 1.0f) deltaPos.y = 2.0f - deltaPos.y;
                if(deltaPos.z > 1.0f) deltaPos.z = 2.0f - deltaPos.z;

                float3 deltaVel = vel - make_float3(oldVel[j]);
                

                // Calculate each effect of boid 2 and atomically add to the effect vector references
                atomicAddv(avoidVel, calcAvoid(deltaPos, avoidAmount));
                atomicAddv(alignVel, calcAlign(deltaPos, deltaVel, alignAmount));
                atomicAddv(cohesionVel, calcCohesion(deltaPos, cohesionAmount));
            }
        }
    }
}

__global__ void interact_kernel(
    float4 *newVel,           // output: new velocity
    float4 *newUp,            // output: new up vector
    float4 *oldPos,           // input: sorted positions
    float4 *oldVel,           // input: sorted velocities
    float4 *oldUp,            // input: sorted up vectors
    uint *gridParticleIndex,  // input: sorted particle indices
    uint *cellStart, uint *cellEnd, uint numBoids)
{
    uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numBoids) return;

    float3 pos = make_float3(oldPos[index]);
    float3 vel = make_float3(oldVel[index]);
    float3 up = make_float3(oldUp[index]);

    int3 gridPos = calcGridPos(pos);

    float3 velEffect = make_float3(0.0f);

    float3 avoidVel = make_float3(0.0f);
    float3 alignVel = make_float3(0.0f);
    float3 cohesionVel = make_float3(0.0f);

    uint avoidAmount = 0;
    uint alignAmount = 0;
    uint cohesionAmount = 0;

    // Loop through each neighbouring cell
    for(int z = -1; z <= 1; z++)
    {
        for(int y = -1; y <= 1; y++)
            {
                for(int x = -1; x <= 1; x++)
                {
                    int3 neighbourPos = gridPos + make_int3(x, y, z);
                    interactWithCell(neighbourPos, index, pos, vel, oldPos, oldVel,
                                    cellStart, cellEnd,
                                    &avoidVel, &avoidAmount,
                                    &alignVel, &alignAmount,
                                    &cohesionVel, &cohesionAmount);
                }   
            }
    }

    // If no boid affects this boid, these prevent division by 0
    if(avoidAmount == 0) avoidAmount = 1;
    if(alignAmount == 0) alignAmount = 1;
    if(cohesionAmount == 0) cohesionAmount = 1;

    // Average each effect and multiply with its corresponding factor, then add them all
    velEffect += avoidVel / avoidAmount * params.avoidFactor;
    velEffect += alignVel / alignAmount * params.alignFactor;
    velEffect += cohesionVel / cohesionAmount * params.cohesionFactor;

    vel += velEffect;
    // Project up vector to the plane with normal of next velocity value
    up -= dot(up, vel)/dot(vel,vel)*vel;
    up = normalize(up);

    uint originalIndex = gridParticleIndex[index];
    newVel[originalIndex] = make_float4(vel, 0.0f);
    newUp[originalIndex] = make_float4(up, 0.0f);
}

#endif //_BOIDS_KERNEL_H_