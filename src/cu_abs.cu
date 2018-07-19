#include <cuda.h>

#include "cu_abs.h"
#include "cu_errchk.h"

#define BLOCKSIZE 128


template <typename T, typename U>
__global__ void abs2(T* __restrict__ y, const U* __restrict__ x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll BLOCKSIZE
    for(; index < N; index += stride) {
        U val = x[index];
        y[index] = sqrt(val.x*val.x+val.y*val.y);
    }
}


template <typename T>
__global__ void iabs2(T* __restrict__ y, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll BLOCKSIZE
    for(; index < N; index += stride) {
        T val = y[index];
        y[index].x = sqrt(val.x*val.x+val.y*val.y);
        y[index].y = 0.; 
    }
}



void cu_abs(void *y, void *x, unsigned long long N,
            int dtype, int dtype_len,
            cudaStream_t *stream)
{
    dim3 blockSize(BLOCKSIZE);
    dim3 gridSize((((N-1)/blockSize.x+1)-1)/blockSize.x+1);
    
    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype) {
        case 2:
        {
            switch(dtype_len) {
                case 2:
                {
                    abs2<<<gridSize,blockSize,0,stream_id>>>(static_cast<float*>(y),
                                                             static_cast<const float2*>(x), N);
                    break;
                }    
            }
            break;
        }
        case 3:
        {
            switch(dtype_len) {
                case 2:
                {
                    abs2<<<gridSize,blockSize,0,stream_id>>>(static_cast<double*>(y),
                                                             static_cast<const double2*>(x), N);
                    break;
                }    
            }
            break;
        }
    }
}



void cu_iabs(void *y, unsigned long long N,
             int dtype, int dtype_len,
             cudaStream_t *stream)
{
    dim3 blockSize(BLOCKSIZE);
    dim3 gridSize((((N-1)/blockSize.x+1)-1)/blockSize.x+1);
    
    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype) {
        case 2:
        {
            switch(dtype_len) {
                case 2:
                {
                    iabs2<<<gridSize,blockSize,0,stream_id>>>(static_cast<float2*>(y), N);
                    break;
                }    
            }
            break;
        }
        case 3:
        {
            switch(dtype_len) {
                case 2:
                {
                    iabs2<<<gridSize,blockSize,0,stream_id>>>(static_cast<double2*>(y), N);
                    break;
                }    
            }
            break;
        }
    }
}