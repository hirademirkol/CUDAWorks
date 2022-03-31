#ifndef __RENDERER__
#define __RENDERER__

class BoidRenderer
{
    public:
        BoidRenderer();
        ~BoidRenderer();

        void setPositions(float *pos, int numBoids);
        void setBuffers(unsigned int posVBO, unsigned int velVBO, unsigned int upVBO, int numBoids);

        void setProjection(float *projectionMat) { m_projectionMatrix = projectionMat; }
        void setModelView(float* modelViewMatrix) {m_modelviewMatrix = modelViewMatrix; }

        void display();

    protected:  //methods
        void _initGL();
        void _drawBoids();
        GLuint _compileProgram(const char *vSource, const char *fSource);

    protected:  //data
        float *m_pos;
        int m_numBoids;

        float m_boidScale;
        
        float *m_projectionMatrix;
        float *m_modelviewMatrix;

        GLuint m_program;

        GLuint m_posVBO;
        GLuint m_velVBO;
        GLuint m_upVBO;

        GLuint m_meshVAO;
};

#endif  //__RENDERER__