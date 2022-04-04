// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include <memory.h>

#include "boidSystem.h"
#include "boidSystem.cuh"

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

BoidSystem::BoidSystem(uint numBoids)
        :   m_bInitialized(false),
            m_numBoids(numBoids),
            m_hPos(0),
            m_hVel(0),
            m_hUp(0)
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
    m_hUp = new float[m_numBoids * 4];
    
    memset(m_hPos, 0, memSize);
    memset(m_hVel, 0, memSize);
    memset(m_hUp, 0, memSize);

    m_posVBO = createVBO(memSize);
    m_velVBO = createVBO(memSize);
    m_upVBO = createVBO(memSize);

    registerGLBufferObject(m_posVBO, &m_cuda_posvbo_resource);
    registerGLBufferObject(m_velVBO, &m_cuda_velvbo_resource);
    registerGLBufferObject(m_upVBO, &m_cuda_upvbo_resource);

    m_bInitialized = true;    
}

void BoidSystem::_finalize()
{
    assert(m_bInitialized);

    delete[] m_hPos;
    delete[] m_hVel;
    delete[] m_hUp;

    unregisterGLBufferObject(m_cuda_posvbo_resource);
    unregisterGLBufferObject(m_cuda_velvbo_resource);
    unregisterGLBufferObject(m_cuda_upvbo_resource);
    
    glDeleteBuffers(1, (const GLuint *)&m_posVBO);
    glDeleteBuffers(1, (const GLuint *)&m_velVBO);
    glDeleteBuffers(1, (const GLuint *)&m_upVBO);
}



void BoidSystem::setArray(DataArray array, const float *data, int start, int count)
{
    assert(m_bInitialized);

    switch (array)
    {
    case POSITION:
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
        break;
    case VELOCITY:
        unregisterGLBufferObject(m_cuda_velvbo_resource);
        glBindBuffer(GL_ARRAY_BUFFER, m_velVBO);
        break;
    case UPVECTOR:
        unregisterGLBufferObject(m_cuda_upvbo_resource);
        glBindBuffer(GL_ARRAY_BUFFER, m_upVBO);
        break;
    default:
        break;
    }

    glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    switch (array)
    {
    case POSITION:
        registerGLBufferObject(m_posVBO, &m_cuda_posvbo_resource);
        break;
    case VELOCITY:
        registerGLBufferObject(m_velVBO, &m_cuda_velvbo_resource);
        break;
    case UPVECTOR:
        registerGLBufferObject(m_upVBO, &m_cuda_upvbo_resource);
        break;
    default:
        break;
    }
}

void BoidSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    float *dPos;
    float *dVel;
    float *dUp;

    dPos = (float*) mapGLBufferObject(&m_cuda_posvbo_resource);
    dVel = (float*) mapGLBufferObject(&m_cuda_velvbo_resource);
    dUp = (float*) mapGLBufferObject(&m_cuda_upvbo_resource);

    integrateSystem(dPos, dVel, dUp, deltaTime, m_numBoids);

    unmapGLBufferObject(m_cuda_posvbo_resource);
    unmapGLBufferObject(m_cuda_velvbo_resource);
    unmapGLBufferObject(m_cuda_upvbo_resource);
}

void BoidSystem::reset()
{
    int p = 0, v = 0, u = 0;

    for (uint i = 0; i < m_numBoids; i++)
    {
        float pos[3];
        float vel[3];
        
        pos[0] = frand();
        pos[1] = frand();
        pos[2] = frand();
        
        vel[0] = frand();
        vel[1] = frand();
        vel[2] = frand();

        m_hPos[p++] = 2 * (pos[0] - 0.5f);
        m_hPos[p++] = 2 * (pos[1] - 0.5f);
        m_hPos[p++] = 2 * (pos[2] - 0.5f);
        m_hPos[p++] = 1.0f;

        m_hVel[v++] = 2 * (vel[0] - 0.5f);
        m_hVel[v++] = 2 * (vel[1] - 0.5f);
        m_hVel[v++] = 2 * (vel[2] - 0.5f);
        m_hVel[v++] = 1.0f;

        m_hUp[u++] = 0.0f;
        m_hUp[u++] = 1.0f;
        m_hUp[u++] = 0.0f;
        m_hUp[u++] = 0.0f;
    }

    setArray(POSITION, m_hPos, 0, m_numBoids);
    setArray(VELOCITY, m_hVel, 0,  m_numBoids);
    setArray(UPVECTOR, m_hUp, 0, m_numBoids);
}