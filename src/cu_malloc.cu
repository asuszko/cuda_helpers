#include <cuda.h>
#include "cu_errchk.h"
#include "cu_malloc.h"


void* cu_malloc(size_t size)
{
	  void* d_arr;
    gpuErrchk(cudaMalloc((void **)&d_arr, size));
	  return d_arr;
}


void *cu_malloc_managed(size_t size)
{
    void *arr;
    gpuErrchk(cudaMallocManaged(&arr, size));
    return arr;
}


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
