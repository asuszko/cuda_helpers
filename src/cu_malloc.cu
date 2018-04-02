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
