#include <cuda.h>
#include "cu_errchk.h"
#include "cu_memcpy.h"


void cu_memcpy_h2d(void *d_arr, void *h_arr, size_t size)
{
	  gpuErrchk(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));
		return;
}


void cu_memcpy_h2d_async(void *d_arr, void *h_arr, size_t size,
									       cudaStream_t *stream)
{
	  gpuErrchk(cudaMemcpyAsync(d_arr, h_arr, size, cudaMemcpyHostToDevice, *stream));
		return;
}


void cu_memcpy_d2h(void *d_arr, void *h_arr, size_t size)
{
	  gpuErrchk(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));
		return;
}


void cu_memcpy_d2h_async(void *d_arr, void *h_arr, size_t size,
										     cudaStream_t *stream)
{
	  gpuErrchk(cudaMemcpyAsync(h_arr, d_arr, size, cudaMemcpyDeviceToHost, *stream));
		return;
}


void cu_memcpy_d2d(void *d_arr_src, void *d_arr_dst, size_t size)
{
	  gpuErrchk(cudaMemcpy(d_arr_dst, d_arr_src, size, cudaMemcpyDeviceToDevice));
		return;
}


void cu_memcpy_d2d_async(void *d_arr_src, void *d_arr_dst, size_t size,
												 cudaStream_t *stream)
{
	  gpuErrchk(cudaMemcpyAsync(d_arr_dst, d_arr_src, size, cudaMemcpyDeviceToDevice, *stream));
		return;
}


void cu_memset(void *d_arr, int value, size_t size)
{
	  gpuErrchk(cudaMemset(d_arr, value, size));
    return;
}


void cu_memset_async(void *d_arr, int value, size_t size, cudaStream_t *stream)
{
	  gpuErrchk(cudaMemsetAsync(d_arr, value, size, *stream));
    return;
}


void cu_mempin(void *h_arr, size_t size)
{
    /* Check if array is already pinned. */
    cudaPointerAttributes ptr_attr;
    bool is_pinned = (cudaPointerGetAttributes(&ptr_attr, h_arr) != cudaErrorInvalidValue);

    /* If array is not already pinned, clear out the error and pin it. Else, do nothing. */
    if (!is_pinned) {
        cudaGetLastError();
        gpuErrchk(cudaHostRegister(h_arr, size, cudaHostRegisterPortable));
    }
    return;
}


void cu_memunpin(void *h_arr)
{
    /* Check if array is pinned. */
    cudaPointerAttributes ptr_attr;
    bool is_pinned = (cudaPointerGetAttributes(&ptr_attr, h_arr) != cudaErrorInvalidValue);

    if (is_pinned) {
    		gpuErrchk(cudaDeviceSynchronize());
    		gpuErrchk(cudaHostUnregister(h_arr));
    }
    else {
        cudaGetLastError();
    }
    return;
}


cudaMemcpy3DParms cu_copyparams(void* src_Array,
							                  cudaArray* dst_Array,
							                  cudaExtent extent,
							                  unsigned int element_size_bytes)
{
	  int nx = extent.width;
	  int ny = extent.height;
	  int nz = extent.depth;


		cudaMemcpy3DParms copyParams = {0};

		copyParams.srcPos   = make_cudaPos(0, 0, 0);
		copyParams.dstPos   = make_cudaPos(0, 0, 0);
		copyParams.srcPtr   = make_cudaPitchedPtr(src_Array, nx*element_size_bytes, nx, ny);
		copyParams.dstArray = dst_Array;
		copyParams.extent   = make_cudaExtent(nx, ny, nz);
		copyParams.kind     = cudaMemcpyHostToDevice;

		return copyParams;
}



// void cu_memcpy3d(void* src_Array,
//                  cudaArray* dst_Array,
//                  cudaExtent extent,
//                  unsigned int element_size_bytes)
// {
//     int nx = extent.width;
//     int ny = extent.height;
//     int nz = extent.depth;
//
//     cudaMemcpy3DParms copyParams = {0};
//
//     copyParams.srcPos   = make_cudaPos(0, 0, 0);
//   	copyParams.dstPos   = make_cudaPos(0, 0, 0);
//   	copyParams.srcPtr   = make_cudaPitchedPtr(src_Array, nx*element_size_bytes, nx, ny);
//     copyParams.dstArray = dst_Array;
//     copyParams.extent   = make_cudaExtent(nx, ny, nz);
//     copyParams.kind     = cudaMemcpyHostToDevice;
//
//     gpuErrchk(cudaMemcpy3D(&copyParams));
//
//     return;
// }
//
//
void cu_memcpy_3d_async(void *src_Array,
                        cudaArray *dst_Array,
                        dim3 extent,
                        unsigned int element_size_bytes,
                        cudaStream_t *stream)
{
    int nx = extent.x;
    int ny = extent.y;
    int nz = extent.z;

    cudaMemcpy3DParms copyParams = {0};

    copyParams.srcPos   = make_cudaPos(0, 0, 0);
  	copyParams.dstPos   = make_cudaPos(0, 0, 0);
  	copyParams.srcPtr   = make_cudaPitchedPtr(src_Array, nx*element_size_bytes, nx, ny);
    copyParams.dstArray = dst_Array;
    copyParams.extent   = make_cudaExtent(nx, ny, nz);
    copyParams.kind     = cudaMemcpyHostToDevice;

    gpuErrchk(cudaMemcpy3DAsync(&copyParams,*stream));

    return;
}
