#ifndef __BOIDSYSTEM_H__
#define __BOIDSYSTEM_H__

#include "sim_params.cuh"

typedef unsigned int uint;
enum DataArray { POSITION, VELOCITY, UPVECTOR, };

class BoidSystem
{
    public:
        BoidSystem(uint numBoids, uint3 gridSize);
        ~BoidSystem();

        void update(float deltaTime);
        void reset();

        float *getArray(DataArray array);
        void setArray(DataArray array, const float *data, int start, int count);

        int getNumBoids() const { return m_numBoids; }

        uint getCurrentPositionBuffer() const { return m_posVBO; }
        uint getCurrentVelocityBuffer() const { return m_velVBO; }
        uint getCurrentUpVectorBuffer() const { return m_upVBO; }

    protected:  // methods
        uint createVBO(uint size);

        void _initialize();
        void _finalize();

    protected:  // data
        bool m_bInitialized;
        uint m_numBoids;

        // CPU data
        float *m_hPos;
        float *m_hVel;
        float *m_hUp;

        // GPU data
        float *m_dSortedPos;
        float *m_dSortedVel;
        float *m_dSortedUp;

        // Grid data for sorting method
        uint *m_dGridParticleHash;   // grid hash value for each particle
        uint *m_dGridParticleIndex;  // particle index for each particle
        uint *m_dCellStart;          // index of start of each cell in sorted list
        uint *m_dCellEnd;            // index of end of cell

        // Buffer objects for OpenGL and CUDA
        uint m_posVBO;
        uint m_velVBO;
        uint m_upVBO;
        struct cudaGraphicsResource *m_cuda_posvbo_resource;
        struct cudaGraphicsResource *m_cuda_velvbo_resource;
        struct cudaGraphicsResource *m_cuda_upvbo_resource;

        SimParams m_params;
        uint3 m_gridSize;
        uint m_numGridCells;
};

#endif  //__BOIDSYSTEM_H__