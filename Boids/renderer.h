#ifndef __RENDERER__
#define __RENDERER__

class BoidRenderer
{
    public:
        BoidRenderer();
        ~BoidRenderer();

        void setPositions(float *pos, int numBoids);
        void setVertexBuffer(unsigned int vbo, int numBoids);
        void setColorBuffer(unsigned int vbo) { m_colorVBO = vbo; }

        void setFOV(float fov) { m_fov = fov; }
        void setWindowSize(int w, int h)
        {
            m_window_w = w;
            m_window_h = h;
        }

        void display();

    protected:  //methods
        void _drawPoints();

    protected:  //data
        float *m_pos;
        int m_numBoids;
        
        float m_pointSize;
        
        float m_fov;
        int m_window_w, m_window_h;
        
        GLuint m_posVBO;
        GLuint m_colorVBO;
};

#endif  //__RENDERER__