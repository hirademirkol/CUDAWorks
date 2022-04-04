// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION

#include <helper_gl.h>
#include <GL/freeglut.h>

#include <math.h>

#include "renderer.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

float triangleVertices[] =
{
    -1.0f, -1.0f,
     0.0f,  1.0f,
     1.0f, -1.0f
};

BoidRenderer::BoidRenderer()
        :   m_pos(0),
            m_numBoids(0),
            m_boidScale(0.02),
            m_program(0),
            m_posVBO(0),
            m_velVBO(0),
            m_upVBO(0),
            m_meshVAO(0)
            {
                _initGL();
            }

BoidRenderer::~BoidRenderer()
{
    m_pos = 0;
    glDeleteBuffers(1, (const GLuint *)&m_posVBO);
    glDeleteBuffers(1, (const GLuint *)&m_velVBO);
    glDeleteBuffers(1, (const GLuint *)&m_upVBO);
    glDeleteBuffers(1, (const GLuint *)&m_meshVAO);
}

void BoidRenderer::setBuffers(unsigned int posVBO, unsigned int velVBO, unsigned int upVBO, int numBoids)
{
    m_posVBO = posVBO;
    m_velVBO = velVBO;
    m_upVBO = upVBO;
    m_numBoids = numBoids;
}

void BoidRenderer::display()
{
    //glEnable(GL_POINT_SPRITE_ARB);
    //glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
    //glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glUseProgram(m_program);
    glUniformMatrix4fv(glGetUniformLocation(m_program, "projectionMat"), 1, GL_FALSE, m_projectionMatrix);
    glUniformMatrix4fv(glGetUniformLocation(m_program, "modelViewMat"), 1, GL_FALSE, m_modelviewMatrix);
    glUniform1f(glGetUniformLocation(m_program, "scale"), m_boidScale);
    glColor3f(1, 1, 1);
    _drawBoids();

    glUseProgram(0);
}

void BoidRenderer::_initGL()
{
    m_program = _compileProgram(vertexShader, fragmentShader);

    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);

    GLuint vertexVBO;
    glGenVertexArrays(1, &m_meshVAO);
    glBindVertexArray(m_meshVAO);
    glGenBuffers(1, &vertexVBO);
    glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
    glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(GLfloat), triangleVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GL_FLOAT), 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void BoidRenderer::_drawBoids()
{
    glBindVertexArray(m_meshVAO);

        glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GL_FLOAT), 0);
        glVertexAttribDivisor(1,1);

        glBindBuffer(GL_ARRAY_BUFFER, m_velVBO);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GL_FLOAT), 0);
        glVertexAttribDivisor(2,1);

        glBindBuffer(GL_ARRAY_BUFFER, m_upVBO);
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GL_FLOAT), 0);
        glVertexAttribDivisor(3,1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, m_numBoids);

    glBindVertexArray(0);
}

GLuint BoidRenderer::_compileProgram(const char *vSource, const char *fSource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vSource, 0);
    glShaderSource(fragmentShader, 1, &fSource, 0);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // Check link status
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if(!success)
    {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}