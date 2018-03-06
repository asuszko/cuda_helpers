#include <cuda.h>
#include "cu_errchk.h"
#include "cu_stream.h"


cudaStream_t *cu_stream_create()
{
    cudaStream_t *stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    gpuErrchk(cudaStreamCreate(stream));
    return stream;
}


void cu_stream_destroy(cudaStream_t *stream)
{
    gpuErrchk(cudaStreamSynchronize(*stream));
    gpuErrchk(cudaStreamDestroy(*stream));
    free(stream);
    return;
}
