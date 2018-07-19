#include <cuda.h>

#include "cu_pow.h"
#include "cu_errchk.h"

#define BLOCKSIZE 128


template <typename T>
__global__ void ipowR(T* __restrict__ y, const T b, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    #pragma unroll BLOCKSIZE
    for(; index < N; index += stride) {
        y[index] = pow(y[index], b);
    }
}


template <typename T, typename U>
__global__ void ipowC(T* __restrict__ y, const U b, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    #pragma unroll BLOCKSIZE
    for(; index < N; index += stride) {
        T val = y[index];
        y[index].x = val.x*val.x-val.y*val.y;
        y[index].y = (U)2*val.x*val.y;
    }
}


void cu_ipow(void *x, unsigned long long N, void *b,
             int dtype, cudaStream_t *stream)
{
    dim3 blockSize(BLOCKSIZE);
    dim3 gridSize((((N-1)/blockSize.x+1)-1)/blockSize.x+1);
    
    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype) {
        
        case 0:
        {
            ipowR<<<gridSize,blockSize,0,stream_id>>>(static_cast<float*>(x),
                                                      static_cast<float*>(b)[0], N);
            break;
        }
        case 1:
        {
            ipowR<<<gridSize,blockSize,0,stream_id>>>(static_cast<double*>(x),
                                                      static_cast<double*>(b)[0], N);
            break;
        }
        case 2:
        {
            ipowC<<<gridSize,blockSize,0,stream_id>>>(static_cast<float2*>(x),
                                                      static_cast<float*>(b)[0], N);
            break;
        }
        case 3:
        {
            ipowC<<<gridSize,blockSize,0,stream_id>>>(static_cast<double2*>(x),
                                                      static_cast<double*>(b)[0], N);
            break;
        }  
    }
    
    return;
}
