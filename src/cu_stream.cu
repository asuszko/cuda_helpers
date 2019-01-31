#include <cuda.h>
#include "cu_errchk.h"
#include "cu_stream.h"


/**
*  Create a CUDA stream.
*  @return stream - [cudaStream_t*] - CUDA stream
*/
cudaStream_t *cu_stream_create()
{
    cudaStream_t *stream = new cudaStream_t;
    gpuErrchk(cudaStreamCreate(stream));
    return stream;
}

/**
*  Destroy a CUDA stream.
*  @param stream - [cudaStream_t*] - CUDA stream
*/
void cu_stream_destroy(cudaStream_t *stream)
{
    gpuErrchk(cudaStreamSynchronize(*stream));
    gpuErrchk(cudaStreamDestroy(*stream));
    delete[] stream;
    return;
}
