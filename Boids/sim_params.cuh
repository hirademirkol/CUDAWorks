#ifndef SIM_PARAMS_H
#define SIM_PARAMS_H

// Simulation parameters to be used by kernels

#include "vector_types.h"

struct SimParams
{
    float3 worldOrigin;     // Bound start (-1,-1,-1)
    uint3 gridSize;
    float3 cellSize;

    float avoidLength;      // Range of avoidance of other boids
    float alignLength;      // Range of alignment and cohesion to other boids ("Visual range")

    float avoidFactor;      // In range 0 to 1
    float alignFactor;      // In range 0 to 1
    float cohesionFactor;   // In range 0 to 1

    float minSpeed;
    float maxSpeed;
};

#endif //SIM_PARAMS_H