#include <stdlib.h>

#include <GL/freeglut.h>

#include <helper_gl.h>

#include "renderer.h"
#include "boidSystem.h"

#define NUM_BOIDS 100

const uint width = 800, height = 600;

float camera_trans_lag[] = {0, 0, -3};
float modelView[16];

BoidRenderer *boidRenderer = 0;
BoidSystem *bsystem = 0;

void initBoidSystem(int numBoids)
{
    bsystem = new BoidSystem(numBoids);
    bsystem->reset();

    boidRenderer = new BoidRenderer;
    boidRenderer->setColorBuffer(bsystem->getColorBuffer());
}

void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA Boids");

    if(!isGLVersionSupported(2,0) ||
       !areGLExtensionsSupported(
          "GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);

    glutReportErrors();
}

void display()
{
    if(boidRenderer)
        boidRenderer->setVertexBuffer(bsystem->getCurrentPositionBuffer(),
                                      bsystem->getNumBoids());

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    //glRotatef(120, -1, 1, -1);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(2.0);

    if(boidRenderer)
        boidRenderer->display();

    glutSwapBuffers();
    //glutReportErrors();
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float)w / (float)h, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    if (boidRenderer) {
    boidRenderer->setWindowSize(w, h);
    boidRenderer->setFOV(60.0);
  }
}

void idle()
{
    glutPostRedisplay();
}

void close()
{
    if(bsystem)
        delete bsystem;
}

int main(int argc, char **argv) 
{
    setenv("DIDSPLAY", ":0", 0);

    initGL(&argc, argv);
    initBoidSystem(NUM_BOIDS);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glutCloseFunc(close);

    glutMainLoop();
}