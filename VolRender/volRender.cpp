//OpenGL
///#include <helper_gl.h>
#include "../../cuda-samples/Common/helper_gl.h"
#include <GL/freeglut.h>

int *pArgc;
char **pArgv;

GLuint pbo = 0;  // OpenGL pixel buffer object
GLuint tex = 0;  // OpenGL texture object

uint width = 512, height = 512;

void initPixelBuffer();

void initGL(int *argc, char **argv) {
  // initialize GLUT callback functions
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowSize(width, height);
  glutCreateWindow("Volume rendering");

  if (!isGLVersionSupported(2, 0) ||
      !areGLExtensionsSupported("GL_ARB_pixel_buffer_object")) {
    printf("Required OpenGL extensions are missing.");
    exit(EXIT_SUCCESS);
  }
}

void render()
{

}

void display()
{
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
    glTexCoord2f(-1, 1);
    glVertex2f(-1, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    glutReportErrors();
}

void idle()
{
    glutPostRedisplay();
}

void initPixelBuffer()
{

}

void close()
{
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
}

//Program Main
int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    setenv("DISPLAY", ":0", 0);

    initGL(&argc, argv);

    int cubeSide = 32;
    size_t size = cubeSide * cubeSide * cubeSide;

    //void *volume = malloc(size);

    glutDisplayFunc(display);
    //glutKeyboardFunc(keyboard);
    //glutMouseFunc(mouse);
    //glutMotionFunc(motion);
    //glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    initPixelBuffer();

    glutCloseFunc(close);

    glutMainLoop();
}