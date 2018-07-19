#include <cuda.h>
#include "cu_errchk.h"
#include "cu_malloc.h"


/**
*  Allocate memory on the device.
*  @param size - [size_t] : Size to allocate in bytes.
*/
void *cu_malloc(size_t size)
{
    void *d_arr;
    gpuErrchk(cudaMalloc((void **)&d_arr, size));
    return d_arr;
}

#include <iostream>
using namespace std;

/**
*  Allocate memory on the device.
*  @param int - [n] : Batch count or number of pointers to allocate.
*/
void **cu_malloc_dblptr(void *A_dflat, unsigned long long N, int batch_size, int dtype)
{
    void **A_d;
    gpuErrchk(cudaMalloc((void**)&A_d,batch_size*sizeof(size_t)));

    switch(dtype) {
        case 0: 
        {
            float **A = (float **)malloc(batch_size*sizeof(float*));
            A[0] = static_cast<float*>(A_dflat);
            for (int i = 1; i < batch_size; i++) {
                A[i] = A[i-1]+N;
            }
            gpuErrchk(cudaMemcpy(A_d,A,batch_size*sizeof(float*),cudaMemcpyHostToDevice));
            break;
        }
        case 1: 
        {
            double **A = (double **)malloc(batch_size*sizeof(double*));
            A[0] = static_cast<double*>(A_dflat);
            for (int i = 1; i < batch_size; i++) {
                A[i] = A[i-1]+N;
            }
            gpuErrchk(cudaMemcpy(A_d,A,batch_size*sizeof(double*),cudaMemcpyHostToDevice));
            break;
        }
        case 2: 
        {
            float2 **A = (float2 **)malloc(batch_size*sizeof(float2*));
            A[0] = static_cast<float2*>(A_dflat);
            for (int i = 1; i < batch_size; i++) {
                A[i] = A[i-1]+N;
            }
            gpuErrchk(cudaMemcpy(A_d,A,batch_size*sizeof(float2*),cudaMemcpyHostToDevice));
            break;
        }
        case 3: 
        {
            double2 **A = (double2 **)malloc(batch_size*sizeof(double2*));
            A[0] = static_cast<double2*>(A_dflat);
            for (int i = 1; i < batch_size; i++) {
                A[i] = A[i-1]+N;
            }
            gpuErrchk(cudaMemcpy(A_d,A,batch_size*sizeof(double2*),cudaMemcpyHostToDevice));
            break;
        }
    }
     
    return A_d;
}


/**
*  Allocate mananged memory on the host and device. CUDA will link
*  host and device memory to the same pointer. Thus, this pointer can
*  be accessed from either. Updating values within the array on either
*  the host or device will result in an automatic update of the other.
*  Using managed memory removes the need to explicity call h2d or d2h
*  memory transfers, albeit at the cost of performance.
*  @param size - [size_t] : Size to allocate in bytes.
*/
void *cu_malloc_managed(size_t size)
{
    void *arr;
    gpuErrchk(cudaMallocManaged(&arr, size));
    return arr;
}

/**
*  Allocate memory (cudaArray) on the device.
*  @param channel - [cudaChannelFormatDesc] : cudaChannelFormatDesc object
*  @param extent - [dim3] : Dimensions of the cudaArray [x,y,z].
*  @param layered - [bool] : cudaArray treated as layered.
*/
cudaArray *cu_malloc_3d(cudaChannelFormatDesc *channel,
                        dim3 extent,
                        bool layered)
{
    cudaArray *cu_array;
    if (layered) {
        gpuErrchk(cudaMalloc3DArray(&cu_array,
                                    channel,
                                    make_cudaExtent(extent.x, extent.y, extent.z),
                                    cudaArrayLayered));
    }
    else {
        gpuErrchk(cudaMalloc3DArray(&cu_array,
                                    channel,
                                    make_cudaExtent(extent.x, extent.y, extent.z)));
    }
    return cu_array;
}



void cu_free(void *d_arr)
{
    gpuErrchk(cudaFree(d_arr));
    return;
}
