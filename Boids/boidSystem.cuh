extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, size_t size);
    void freeArray(void *devPtr);

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);

    void setParameters(SimParams *hostParams);
    
    void integrateSystem(float *pos, float *vel, float deltaTime, uint numBoids);

    void calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *pos,
        int numBoids);
        
    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex,
                    uint numBoids);

    void reorderDataAndFindCellStart(uint *cellStart, uint *cellEnd,
                                     float *sortedPos, float *sortedVel, float *sortedUp,
                                     uint *gridParticleHash,
                                     uint *gridParticleIndex, float *oldPos,
                                     float *oldVel, float* oldUp, uint numBoids,
                                     uint numCells);


    void interact(float *newVel, float *newUp, float *sortedPos, float *sortedVel, float *sortedUp,
                  uint *gridParticleIndex, uint *cellStart, uint *cellEnd,
                  uint numBoids, uint numCells);
}