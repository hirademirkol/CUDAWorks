// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include <memory.h>

#include "boidSystem.h"

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

BoidSystem::BoidSystem(uint numBoids)
        :   m_bInitialized(false),
            m_numBoids(numBoids),
            m_hPos(0),
            m_hVel(0)
            {
                _initialize();
            }

BoidSystem::~BoidSystem()
{
    _finalize();
    m_numBoids = 0;
}

uint BoidSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

void BoidSystem::_initialize()
{
    assert(!m_bInitialized);

    uint memSize = m_numBoids*4*sizeof(float);

    // Allocate host data
    m_hPos = new float[m_numBoids * 4];
    m_hVel = new float[m_numBoids * 4];
    
    memset(m_hPos, 0, memSize);
    memset(m_hVel, 0, memSize);

    m_posVBO = createVBO(memSize);
    m_colorVBO = createVBO(memSize);

    // Fill color buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
    float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    float *ptr = data;

    for(uint i = 0; i < m_numBoids; i++)
    {
        *ptr++ = frand();
        *ptr++ = frand();
        *ptr++ = frand();
        *ptr++ = 1.0f;
    }

    glUnmapBuffer(GL_ARRAY_BUFFER);

    m_bInitialized = true;
    
}

void BoidSystem::_finalize()
{
    assert(m_bInitialized);

    delete[] m_hPos;
    delete[] m_hVel;

    glDeleteBuffers(1, (const GLuint *)&m_posVBO);
    glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
}

float *BoidSystem::getArray(DataArray array)
{
    assert(m_bInitialized);

    float *hdata = 0;

    switch (array)
    {
    case POSITION:
        hdata = m_hPos;
        break;
    case VELOCITY:
        hdata = m_hVel;
        break;
    default:
        break;
    }

    return hdata;
}

void BoidSystem::setArray(DataArray array, const float *data, int start, int count)
{
    assert(m_bInitialized);

    switch (array)
    {
    case POSITION:
        glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
        glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        break;
    case VELOCITY:
        break;
    default:
        break;
    }
}

void BoidSystem::reset()
{
    int p = 0, v = 0;

    for (uint i = 0; i < m_numBoids; i++)
    {
        float point[3];
        point[0] = frand();
        point[1] = frand();
        point[2] = frand();
        m_hPos[p++] = 2 * (point[0] - 0.5f);
        m_hPos[p++] = 2 * (point[1] - 0.5f);
        m_hPos[p++] = 2 * (point[2] - 0.5f);
        m_hPos[p++] = 1.0f;
        m_hVel[v++] = 0.0f;
        m_hVel[v++] = 0.0f;
        m_hVel[v++] = 0.0f;
        m_hVel[v++] = 0.0f;
    }

    setArray(POSITION, m_hPos, 0, m_numBoids);
    setArray(VELOCITY, m_hVel, 0,  m_numBoids);
}