#ifndef __BOIDSYSTEM_H__
#define __BOIDSYSTEM_H__

typedef unsigned int uint;
enum DataArray { POSITION, VELOCITY, };

class BoidSystem
{
    public:
        BoidSystem(uint numBoids);
        ~BoidSystem();

        void reset();

        float *getArray(DataArray array);
        void setArray(DataArray array, const float *data, int start, int count);

        int getNumBoids() const { return m_numBoids; }

        uint getCurrentPositionBuffer() const { return m_posVBO; }
        uint getColorBuffer() const { return m_colorVBO; }

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

        uint m_posVBO;
        uint m_colorVBO;

};

#endif  //__BOIDSYSTEM_H__