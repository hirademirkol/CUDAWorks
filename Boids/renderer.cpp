// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>
#include <GL/freeglut.h>

#include "renderer.h"

BoidRenderer::BoidRenderer()
        :   m_pos(0),
            m_numBoids(0),
            m_pointSize(1.0f),
            m_posVBO(0),
            m_colorVBO(0)
            {}

BoidRenderer::~BoidRenderer() { m_pos = 0; }

void BoidRenderer::setPositions(float *pos, int numBoids)
{
    m_pos = pos;
    m_numBoids = numBoids;
}

void BoidRenderer::setVertexBuffer(unsigned int vbo, int numBoids)
{
    m_posVBO = vbo;
    m_numBoids = numBoids;
}

void BoidRenderer::display()
{
    glColor3f(1, 1, 1);
    glPointSize(m_pointSize);
    _drawPoints();
}

void BoidRenderer::_drawPoints()
{
    if(!m_posVBO)
    {
        glBegin(GL_POINTS);
        {
            int k = 0;

            for(int i = 0; i < m_numBoids; ++i)
            {
                glVertex3fv(&m_pos[k]);
                k +=4;
            }
        }
        glEnd();
    } else
    {
        glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
        glVertexPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);

        if(m_colorVBO)
        {
            glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
            glColorPointer(4, GL_FLOAT, 0, 0);
            glEnableClientState(GL_COLOR_ARRAY);
        }

        glDrawArrays(GL_POINTS, 0, m_numBoids);
            glutReportErrors();

        glBindBuffer(GL_ARRAY_BUFFER,0);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);

    }
}