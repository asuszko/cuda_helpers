#include <cuda.h>
#include <tuple>

#include "cu_isub.h"
#include "cu_errchk.h"


template <typename T>
__global__ void sub1_val(T *y, const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N; index += stride) {
        y[index] -= x[0];
    }
}


template <typename T>
__global__ void sub1_vec(T *y, const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N; index += stride) {
        y[index] -= x[index];
    }     
}


template <typename T>
__global__ void sub2_val(T *y, const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N; index += stride) {
        T valy = y[index];
        T valx = x[0];
		
        valy.x -= valx.x;
        valy.y -= valx.y;
		
        y[index] = valy;
    }     
}


template <typename T>
__global__ void sub2_vec(T *y, const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N; index += stride) {
        T valy = y[index];
        T valx = x[index];
        
        valy.x -= valx.x;
        valy.y -= valx.y;
		
        y[index] = valy;
    }     
}


template <typename T>
__global__ void sub3_val(T *y, const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N; index += stride) {
        T valy = y[index];
        T valx = x[0];
		
        valy.x -= valx.x;
        valy.y -= valx.y;
        valy.z -= valx.z;
		
        y[index] = valy;
    }     
}


template <typename T>
__global__ void sub3_vec(T *y, const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N; index += stride) {
        T valy = y[index];
        T valx = x[index];
		
        valy.x -= valx.x;
        valy.y -= valx.y;
        valy.z -= valx.z;
		
        y[index] = valy;
    }     
}


template <typename T>
__global__ void sub4_val(T *y, const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N; index += stride) {
        T valy = y[index];
        T valx = x[0];
		
        valy.x -= valx.x;
        valy.y -= valx.y;
        valy.z -= valx.z;
        valy.w -= valx.w;
		
        y[index] = valy;
    }     
}


template <typename T>
__global__ void sub4_vec(T *y, const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N/4; index += stride) {
        T valy = y[index];
        T valx = x[index];
		
        valy.x -= valx.x;
        valy.y -= valx.y;
        valy.z -= valx.z;
        valy.w -= valx.w;
		
        y[index] = valy;
    }     
}



void cu_isub(void *y, void *x, unsigned long long N,
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
				         if (vec) sub1_vec<<<gridSize,blockSize,0,stream_id>>>((float*)y, (const float*)x, N);
				         else     sub1_val<<<gridSize,blockSize,0,stream_id>>>((float*)y, (const float*)x, N);
				         break;
                case 2:
				         if (vec) sub2_vec<<<gridSize,blockSize,0,stream_id>>>((float2*)y,(const float2*)x,N);
				         else     sub2_val<<<gridSize,blockSize,0,stream_id>>>((float2*)y,(const float2*)x,N);
				         break;
                case 3:
				         if (vec) sub3_vec<<<gridSize,blockSize,0,stream_id>>>((float3*)y,(const float3*)x,N);
				         else     sub3_val<<<gridSize,blockSize,0,stream_id>>>((float3*)y,(const float3*)x,N);
				         break;
                case 4:
				         if (vec) sub4_vec<<<gridSize,blockSize,0,stream_id>>>((float4*)y,(const float4*)x,N);
				         else     sub4_val<<<gridSize,blockSize,0,stream_id>>>((float4*)y,(const float4*)x,N);
				         break;
        }
            break;
        case 1:
            switch(dtype_len) {
				     case 1:
				         if (vec) sub1_vec<<<gridSize,blockSize,0,stream_id>>>((double*)y, (const double*)x, N);
				         else     sub1_val<<<gridSize,blockSize,0,stream_id>>>((double*)y, (const double*)x, N);
				         break;
                case 2:
				         if (vec) sub2_vec<<<gridSize,blockSize,0,stream_id>>>((double2*)y,(const double2*)x,N);
				         else     sub2_val<<<gridSize,blockSize,0,stream_id>>>((double2*)y,(const double2*)x,N);
				         break;
                case 3:
				         if (vec) sub3_vec<<<gridSize,blockSize,0,stream_id>>>((double3*)y,(const double3*)x,N);
				         else     sub3_val<<<gridSize,blockSize,0,stream_id>>>((double3*)y,(const double3*)x,N);
				         break;
                case 4:
				         if (vec) sub4_vec<<<gridSize,blockSize,0,stream_id>>>((double4*)y,(const double4*)x,N);
				         else     sub4_val<<<gridSize,blockSize,0,stream_id>>>((double4*)y,(const double4*)x,N);
				         break;
            }
            break;
    }
}