#include <cuda.h>

#include "cu_imul.h"
#include "cu_errchk.h"

#define BLOCKSIZE 128
const int bs = BLOCKSIZE;

template <typename T>
__global__ void mul1_val(T* __restrict__ y, const T x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; index < N; index += stride) {
        y[index] *= x;
    }
}


template <typename T>
__global__ void mul1_vec(T* __restrict__ y, const T* __restrict__ x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    #pragma unroll bs
    for(; index < N; index += stride) {
        y[index] *= x[index];
    }     
}


template <typename T>
__global__ void mul2_val(T* __restrict__ y, const T x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    #pragma unroll bs
    for(; index < N; index += stride) {
        T valy = y[index];
        
        y[index].x = valy.x*x.x-valy.y*x.y;
        y[index].y = valy.x*x.y+valy.y*x.x;
    }     
}


template <typename T>
__global__ void mul2_vec(T* __restrict__ y, const T* __restrict__ x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    #pragma unroll bs
    for(; index < N; index += stride) {
        T valy = y[index];
        T valx = x[index];
        
        y[index].x = valy.x*valx.x-valy.y*valx.y;
        y[index].y = valy.x*valx.y+valy.y*valx.x;
    }     
}




void cu_imul_vec(void *y, void *x, unsigned long long N,
                 int dtype, int dtype_len,
                 cudaStream_t *stream)
{
    dim3 blockSize(BLOCKSIZE);
    dim3 gridSize((((N-1)/blockSize.x+1)-1)/blockSize.x+1);
    
    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype) {
        case 0:
            switch(dtype_len) {
                case 1:
                    mul1_vec<<<gridSize,blockSize,0,stream_id>>>(static_cast<float*>(y),
                                                                 static_cast<const float*>(x), N);
                    break;
                case 2:
                    mul2_vec<<<gridSize,blockSize,0,stream_id>>>(static_cast<float2*>(y),
                                                                 static_cast<const float2*>(x), N);
                    break;
            }
            break;
        case 1:
            switch(dtype_len) {
                case 1:
                    mul1_vec<<<gridSize,blockSize,0,stream_id>>>(static_cast<double*>(y),
                                                                 static_cast<const double*>(x), N);
                    break;
                case 2:
                    mul2_vec<<<gridSize,blockSize,0,stream_id>>>(static_cast<double2*>(y),
                                                                 static_cast<const double2*>(x), N);
                    break;
            }
            break;
    }
}


void cu_imul_val(void *y, void *x, unsigned long long N,
                 int dtype, int dtype_len,
                 cudaStream_t *stream)
{
    dim3 blockSize(BLOCKSIZE);
    dim3 gridSize((((N-1)/blockSize.x+1)-1)/blockSize.x+1);
    
    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype) {
        case 0:
            switch(dtype_len) {
                case 1:
                    mul1_val<<<gridSize,blockSize,0,stream_id>>>(static_cast<float*>(y),
                                                                (static_cast<const float*>(x))[0], N);
                    break;
                case 2:
                    mul2_val<<<gridSize,blockSize,0,stream_id>>>(static_cast<float2*>(y),
                                                                (static_cast<const float2*>(x))[0], N);
                    break;
            }
            break;
        case 1:
            switch(dtype_len) {
                case 1:
                    mul1_val<<<gridSize,blockSize,0,stream_id>>>(static_cast<double*>(y),
                                                                (static_cast<const double*>(x))[0], N);
                    break;
                case 2:
                    mul2_val<<<gridSize,blockSize,0,stream_id>>>(static_cast<double2*>(y),
                                                                (static_cast<const double2*>(x))[0], N);
                    break;
            }
            break;
    }
}