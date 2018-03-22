#include <cuda.h>

#include "cu_abs.h"
#include "cu_errchk.h"


template <typename T>
__global__ void abs_val(const T *x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;
    
    for(; index < N; index += stride) {
        x[index] *= max(-x[index], x[index]);
    }
}


void cu_abs(void *x, unsigned long long N,
            const int dtype, int dtype_len,
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
				         abs_val<<<gridSize,blockSize,0,stream_id>>>((float*)x, N);
				         break;
                case 2:
				         abs_val<<<gridSize,blockSize,0,stream_id>>>((float2*)x, N);
				         break;
                case 3:
				         abs_val<<<gridSize,blockSize,0,stream_id>>>((float3*)x, N);
				         break;
                case 4:
				         abs_val<<<gridSize,blockSize,0,stream_id>>>((float4*)x, N);
				         break;
            }
            break;
        case 1:
            switch(dtype_len) {
                case 1:
				         abs_val<<<gridSize,blockSize,0,stream_id>>>((double*)x, N);
				         break;
                case 2:
				         abs_val<<<gridSize,blockSize,0,stream_id>>>((double2*)x, N);
				         break;
                case 3:
				         abs_val<<<gridSize,blockSize,0,stream_id>>>((double3*)x, N);
				         break;
                case 4:
				         abs_val<<<gridSize,blockSize,0,stream_id>>>((double4*)x, N);
				         break;
            }
            break;
    }
}