#include <cuda.h>

#include "cu_conj.h"
#include "cu_errchk.h"

#define BLOCKSIZE 128
const int bs = BLOCKSIZE;

template<typename T>
__global__ void complex_conj(T *odata, const unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    #pragma unroll bs
    for(; index < N; index += stride) {
        odata[index].y *= -1.;
    }
}


void cu_conj(void *d_data,
             const unsigned long long N,
             const int dtype,
             cudaStream_t *stream)
{
    dim3 blockSize(BLOCKSIZE);
    dim3 gridSize((((N-1)/blockSize.x+1)-1)/blockSize.x+1);
    
    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype) {

        case 2:
            complex_conj<<<gridSize,blockSize,0,stream_id>>>(static_cast<float2*>(d_data), N);
            break;

        case 3:
            complex_conj<<<gridSize,blockSize,0,stream_id>>>(static_cast<double2*>(d_data), N);
            break;
    }

    return;
}
