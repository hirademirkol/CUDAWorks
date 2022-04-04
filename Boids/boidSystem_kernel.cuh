#ifndef _BOIDS_KERNEL_H_
#define _BOIDS_KERNEL_H_

#include "helper_math.h"

#include "vector_types.h"

typedef unsigned int uint;

struct integrate_functor
{
    float _deltaTime;
    
    __host__ __device__ integrate_functor(float deltaTime) : _deltaTime(deltaTime){}

    template <typename Tuple>
    __device__ void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        volatile float4 upData = thrust::get<2>(t);

        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);
        float3 up = make_float3(upData.x, upData.y, upData.z);

        pos += vel * _deltaTime;
        up -= dot(up, vel)/dot(vel,vel)*vel; // TODO: Must use next velocity value 
        up = normalize(up);

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
        thrust::get<2>(t) = make_float4(up, upData.w);
    }
};

#endif //_BOIDS_KERNEL_H_