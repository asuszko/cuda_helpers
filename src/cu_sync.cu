
#include <cuda.h>
#include "cu_errchk.h"
#include "cu_sync.h"


/**
*  Wait for CUDA device to finish all tasks.
*/
void cu_sync_device()
{
	  gpuErrchk(cudaDeviceSynchronize());
    return;
}


/**
*  Wait for CUDA stream to finish all tasks.
*  @param stream - [cudaStream_t*] - CUDA stream handle
*/
void cu_sync_stream(cudaStream_t *stream)
{
	  gpuErrchk(cudaStreamSynchronize(*stream));
    return;
}
