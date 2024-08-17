#include <helper_gl.h>
#include <helper_cuda.h>

#include <cuda_gl_interop.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "boidSystem_kernel.cuh"

extern "C"
{
    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest
        // Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);
      
        if (devID < 0)
        {
          printf("No CUDA Capable devices found, exiting...\n");
          exit(EXIT_SUCCESS);
        }
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }
      
    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
    {
        checkCudaErrors(
            cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
    }

    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(
            cudaGraphicsUnregisterResource(cuda_vbo_resource));
    }

    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
    {
        void *ptr;

        checkCudaErrors(
            cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
        
        size_t num_bytes;
        checkCudaErrors(
            cudaGraphicsResourceGetMappedPointer((void**)&ptr, &num_bytes, *cuda_vbo_resource));

        return ptr;
    }

    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(
            cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

    }

    // Transfer simulation parameters to the constant CUDA memory
    void setParameters(SimParams *hostParams)
    {
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    // Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

    // Compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    // Advance the system one step using thrust
    // It is a SIMD operation, as the velocity is calculated in the previous frame
    void integrateSystem(float *pos, float *vel, float deltaTime, uint numBoids)
    {
        thrust::device_ptr<float4> d_pos4((float4 *)pos);
        thrust::device_ptr<float4> d_vel4((float4 *)vel);

        thrust::for_each
        (
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4 + numBoids, 
                                                         d_vel4 + numBoids)),
            integrate_functor(deltaTime)
        );
    }

    // Calculate hash values for each boids position in the grid
    void calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *pos, int numBoids)
    {
        uint numThreads, numBlocks;
        computeGridSize(numBoids, 256, numBlocks, numThreads);

        calcHash_kernel<<<numBlocks, numThreads>>>(gridParticleHash, gridParticleIndex,
                                                   (float4 *)pos, numBoids);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    // Sort elements by grid hashes so that each one can be reached from their grids
    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numBoids)
    {
        thrust::sort_by_key
        (
            thrust::device_ptr<uint>(dGridParticleHash),
            thrust::device_ptr<uint>(dGridParticleHash + numBoids),
            thrust::device_ptr<uint>(dGridParticleIndex)
        );
    }

    // Order data into sorted arrays and find particles bounding each cell 
    void reorderDataAndFindCellStart(uint *cellStart, uint *cellEnd,
                                     float *sortedPos, float *sortedVel, float *sortedUp,
                                     uint *gridParticleHash,
                                     uint *gridParticleIndex, float *oldPos,
                                     float *oldVel, float *oldUp, uint numBoids,
                                     uint numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(numBoids, 256, numBlocks, numThreads);

        // Set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

        uint smemSize = sizeof(uint) * (numThreads + 1);
        reorderDataAndFindCellStart_kernel<<<numBlocks, numThreads, smemSize>>>(
        cellStart, cellEnd, (float4 *)sortedPos, (float4 *)sortedVel,
        (float4 *)sortedUp, gridParticleHash, gridParticleIndex,
        (float4 *)oldPos, (float4 *)oldVel, (float4 *)oldUp,
        numBoids);
        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStart_kernel");
    }

    // Call the kernel for calculation boid interactions
    void interact(float *newVel, float *newUp, float *sortedPos, float *sortedVel, float *sortedUp,
                  uint *gridParticleIndex, uint *cellStart, uint *cellEnd,
                  uint numBoids)
    {
        uint numThreads, numBlocks;
        computeGridSize(numBoids, 64, numBlocks, numThreads);

        interact_kernel<<<numBlocks, numThreads>>>((float4 *) newVel, (float4 *) newUp,
                                                   (float4 *)sortedPos, (float4 *)sortedVel,
                                                   (float4 *)sortedUp, gridParticleIndex,
                                                   cellStart, cellEnd, numBoids);

        // Check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }    
} // extern "C"