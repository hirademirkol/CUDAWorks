//OpenGL
#include <GL/freeglut.h>

//CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

//Helpers from NVIDIA Samples
#include "helper_gl.h"
#include "helper_cuda.h"
#include "helper_math.h"

int *pArgc;
char **pArgv;

GLuint pbo = 0;  // OpenGL pixel buffer object
GLuint tex = 0;  // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource for OpenGL exchange

uint width = 800, height = 800;
dim3 blockSize(16, 16);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output,
                              uint imageW, uint imageH);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeOfMatrix);

void initPixelBuffer();

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


void initGL(int *argc, char **argv)
{
  // initialize GLUT callback functions
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowSize(width, height);
  glutCreateWindow("Volume Rendering");

  if (!isGLVersionSupported(2, 0) ||
      !areGLExtensionsSupported("GL_ARB_pixel_buffer_object")) {
    printf("Required OpenGL extensions are missing.");
    exit(EXIT_SUCCESS);
  }
}

void render()
{
    copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);

    uint *output;
    checkCudaErrors(cudaGraphicsMapResources(1,&cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
        (void **)&output, &num_bytes, cuda_pbo_resource));

    checkCudaErrors(cudaMemset(output, 0, width * height * 4));

    render_kernel(gridSize, blockSize, output, width, height);
    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

void display()
{
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
    glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    render();
    
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                    GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    //glColor3s(128,12,255);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(1, 0);
    glVertex2f(1, -1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(-1, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    glutReportErrors();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y) {
  if (state == GLUT_DOWN) {
    buttonState |= 1 << button;
  } else if (state == GLUT_UP) {
    buttonState = 0;
  }

  ox = x;
  oy = y;
  glutPostRedisplay();
}

void motion(int x, int y) {
  float dx, dy;
  dx = (float)(x - ox);
  dy = (float)(y - oy);

  if (buttonState == 4) {
    // right = zoom
    viewTranslation.z += dy / 100.0f;
  } else if (buttonState == 2) {
    // middle = translate
    viewTranslation.x += dx / 100.0f;
    viewTranslation.y -= dy / 100.0f;
  } else if (buttonState == 1) {
    // left = rotate
    viewRotation.x += dy / 5.0f;
    viewRotation.y += dx / 5.0f;
  }

  ox = x;
  oy = y;
  glutPostRedisplay();
}

void idle()
{
    glutPostRedisplay();
}

void initPixelBuffer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
        
        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(GLubyte) * 4,
                0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(
        &cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void close()
{
    //freeCudaBuffers();
    
    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
}

//Program Main
int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    setenv("DISPLAY", ":0", 0);

    initGL(&argc, argv);
    findCudaDevice(argc, (const char**)argv);

    //int cubeSide = 32;
    //size_t size = cubeSide * cubeSide * cubeSide;
    //void *volume = malloc(size);

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
    
    glutDisplayFunc(display);
    //glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    //glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    initPixelBuffer();

    glutCloseFunc(close);

    glutMainLoop();
}