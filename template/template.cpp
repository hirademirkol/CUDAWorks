#include <stdlib.h>

#include <GL/freeglut.h>

#include <helper_gl.h>

const uint width = 800, height = 600;

void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("Template");

    if(!isGLVersionSupported(2,0) ||
       !areGLExtensionsSupported(
          "GL_ARB_vertex_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);

    glutReportErrors();
}

void display()
{

}

void idle()
{
    glutPostRedisplay();
}

void close()
{

}

int main(int argc, char **argv) 
{
    setenv("DIDSPLAY", ":0", 0);

    initGL(&argc, argv);

    glutDisplayFunc(display);
    glutIdleFunc(idle);

    glutCloseFunc(close);

    glutMainLoop();
}