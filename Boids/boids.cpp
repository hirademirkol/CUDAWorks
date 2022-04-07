#include <stdlib.h>
#include <algorithm>

#include <GL/freeglut.h>

#include <math.h>
#include <helper_gl.h>

#include "renderer.h"
#include "boidSystem.h"

#define NUM_BOIDS 4096
#define GRID_SIZE 64

// Window parameters
const uint width = 800, height = 600;

// Mouse parameters
int ox, oy;
int buttonState = 0;

// View parameters
float camera_trans[] = {0, 0, -4};
float camera_rot[] = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -4};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1f;

float modelView[16];
float projection[16];

uint numBoids;
uint3 gridSize;
float timestep = 0.01f;

BoidRenderer *boidRenderer = 0;
BoidSystem *bsystem = 0;

extern "C" void cudaInit(int argc, char **argv);

void initBoidSystem(int numBoids, uint3 gridSize)
{
    bsystem = new BoidSystem(numBoids, gridSize);
    bsystem->reset();

    boidRenderer = new BoidRenderer;
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
    // Advance the simulation one step 
    bsystem->update(timestep);

    // Transfer buffers to the renderer
    if(boidRenderer)
        boidRenderer->setBuffers(bsystem->getCurrentPositionBuffer(),
                                     bsystem->getCurrentVelocityBuffer(),
                                     bsystem->getCurrentUpVectorBuffer(),
                                     bsystem->getNumBoids());

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //Create ModelView matrix according to motion
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    for (int c = 0; c < 3; c++)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;

    }

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1, 0, 0);
    glRotatef(camera_rot_lag[1], 0, 1, 0);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(2.0);

    if(boidRenderer)
    {
        // Transfer ModelView matrix to the renderer
        boidRenderer->setModelView(modelView);
        boidRenderer->display();
    }

    glutSwapBuffers();
    glutReportErrors();
}

void reshape(int w, int h)
{
    // Recalculate ModelView and Projection matrices and transfer to the renderer

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float)w / (float)h, 0.1, 100.0);

    glGetFloatv(GL_PROJECTION_MATRIX, projection);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    if (boidRenderer)
    {
        boidRenderer->setProjection(projection);
        boidRenderer->setModelView(modelView);    
    }
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        if(button == 3 || button == 4 )
        {
            int dir = 7 - 2*button; // Mouse buttons 3-4 correspond to scroll up and down
            camera_trans[2] += dir * 0.05f * fabs(camera_trans[2]);
        }
        else
            buttonState = button + 1;
    } else if (state == GLUT_UP) {
        buttonState = 0;
    }

    ox = x;
    oy = y;
    
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);


    if(buttonState == 1)// Mouse 1: Rotate
    {
        camera_rot[0] += dy / 5.0f;
        camera_rot[1] += dx / 5.0f;
    }
    else if(buttonState == 2)// Mouse 2: Pan
    {
        camera_trans[0] += dx / 100.0f;
        camera_trans[1] -= dy / 100.0f;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
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

    numBoids = NUM_BOIDS;
    uint gridDim = GRID_SIZE;
    gridSize.x = gridSize.y = gridSize.z = gridDim;

    initGL(&argc, argv);
    cudaInit(argc, argv);
    
    initBoidSystem(numBoids, gridSize);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutIdleFunc(idle);

    glutCloseFunc(close);

    glutMainLoop();
}