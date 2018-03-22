#include <cuda.h>
#include <tuple>

#include "cu_idiv.h"
#include "cu_errchk.h"


template <typename T>
__global__ void div1_val(T *y, const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N; index += stride) {
        y[index] /= x[0];
    }
}


template <typename T>
__global__ void div1_vec(T *y, const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N; index += stride) {
        y[index] /= x[index];
    }     
}


template <typename T>
__global__ void div2_val(T *y, const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N; index += stride) {
        T valy = y[index];
        T valx = x[0];
        
        y[index].x = (valy.x*valx.x+valy.y*valx.y)/(valx.x*valx.x+valx.y*valx.y);
        y[index].y = (valy.y*valx.x-valy.x*valx.y)/(valx.x*valx.x+valx.y*valx.y);
    }     
}


template <typename T>
__global__ void div2_vec(T *y, const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N; index += stride) {
        T valy = y[index];
        T valx = x[index];
        
        y[index].x = (valy.x*valx.x+valy.y*valx.y)/(valx.x*valx.x+valx.y*valx.y);
        y[index].y = (valy.y*valx.x-valy.x*valx.y)/(valx.x*valx.x+valx.y*valx.y);
    }     
}


void cu_idiv(void *y, void *x, unsigned long long N,
             const int dtype, int dtype_len, bool vec,
             cudaStream_t *stream)
{
    dim3 blockSize(256);
    dim3 gridSize((((N-1)/blockSize.x+1)-1)/blockSize.x+1);
    
    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype) {
        case 0:
            switch(dtype_len) {
				     case 1:
				         if (vec) div1_vec<<<gridSize,blockSize,0,stream_id>>>((float*)y, (const float*)x, N);
				         else     div1_val<<<gridSize,blockSize,0,stream_id>>>((float*)y, (const float*)x, N);
				         break;
                case 2:
				         if (vec) div2_vec<<<gridSize,blockSize,0,stream_id>>>((float2*)y,(const float2*)x,N);
				         else     div2_val<<<gridSize,blockSize,0,stream_id>>>((float2*)y,(const float2*)x,N);
				         break;
        }
            break;
        case 1:
            switch(dtype_len) {
				     case 1:
				         if (vec) div1_vec<<<gridSize,blockSize,0,stream_id>>>((double*)y, (const double*)x, N);
				         else     div1_val<<<gridSize,blockSize,0,stream_id>>>((double*)y, (const double*)x, N);
				         break;
                case 2:
				         if (vec) div2_vec<<<gridSize,blockSize,0,stream_id>>>((double2*)y,(const double2*)x,N);
				         else     div2_val<<<gridSize,blockSize,0,stream_id>>>((double2*)y,(const double2*)x,N);
				         break;
            }
            break;
    }
}