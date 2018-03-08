#include <cuda.h>
#include "cu_errchk.h"
#include "cu_sync.h"


/**
*  Block host thread until the CUDA device finishes all tasks.
*/
void cu_sync_device()
{
    gpuErrchk(cudaDeviceSynchronize());
    return;
}


/**
*  Block host thread until the CUDA stream finishes all tasks.
*  @param stream - [cudaStream_t*] - CUDA stream handle
*/
void cu_sync_stream(cudaStream_t *stream)
{
    gpuErrchk(cudaStreamSynchronize(*stream));
    return;
}
