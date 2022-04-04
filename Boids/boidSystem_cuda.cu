#include <helper_cuda.h>

#include <cuda_gl_interop.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"

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

    void integrateSystem(float *pos, float *vel, float *up, float deltaTime, uint numBoids)
    {
        thrust::device_ptr<float4> d_pos4((float4 *)pos);
        thrust::device_ptr<float4> d_vel4((float4 *)vel);
        thrust::device_ptr<float4> d_up4((float4 *)up);

        thrust::for_each
        (
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4, d_up4)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4 + numBoids, 
                                                         d_vel4 + numBoids,
                                                         d_up4 + numBoids)),
            integrate_functor(deltaTime)
        );
    }
}