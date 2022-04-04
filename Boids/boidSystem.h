#ifndef __BOIDSYSTEM_H__
#define __BOIDSYSTEM_H__

typedef unsigned int uint;
enum DataArray { POSITION, VELOCITY, UPVECTOR, };

class BoidSystem
{
    public:
        BoidSystem(uint numBoids);
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

        uint m_posVBO;
        uint m_velVBO;
        uint m_upVBO;

        struct cudaGraphicsResource *m_cuda_posvbo_resource;
        struct cudaGraphicsResource *m_cuda_velvbo_resource;
        struct cudaGraphicsResource *m_cuda_upvbo_resource;
};

#endif  //__BOIDSYSTEM_H__