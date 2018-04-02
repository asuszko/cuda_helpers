#include <cuda.h>
#include "cu_errchk.h"
#include "cu_memcpy.h"


/**
*  Copy a chunk of memory from the host to device.
*  @param d_arr - [void*] : Pointer to device memory.
*  @param h_arr - [void*] : Pointer to host memory.
*  @param size - [size_t] : Size of the transfer in bytes.
*/
void cu_memcpy_h2d(void *d_arr,
                   void *h_arr,
                   size_t size)
{
    gpuErrchk(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));
    return;
}

/**
*  Async Copy a chunk of memory from the host to device.
*  @param d_arr - [void*] : Pointer to device memory.
*  @param h_arr - [void*] : Pointer to host memory.
*  @param size - [size_t] : Size of the transfer in bytes.
*  @param stream - [cudaStream_t*] : CUDA stream
*/
void cu_memcpy_h2d_async(void *d_arr,
                         void *h_arr,
                         size_t size,
                         cudaStream_t *stream)
{
    gpuErrchk(cudaMemcpyAsync(d_arr, h_arr, size, cudaMemcpyHostToDevice, *stream));
    return;
}

/**
*  Copy a chunk of memory from the device to host.
*  @param d_arr - [void*] : Pointer to device memory.
*  @param h_arr - [void*] : Pointer to host memory.
*  @param size - [size_t] : Size of the transfer in bytes.
*/
void cu_memcpy_d2h(void *d_arr,
                   void *h_arr,
                   size_t size)
{
    gpuErrchk(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));
    return;
}

/**
*  Async copy a chunk of memory from the device to host.
*  @param d_arr - [void*] : Pointer to device memory.
*  @param h_arr - [void*] : Pointer to host memory.
*  @param size - [size_t] : Size of the transfer in bytes.
*  @param stream - [cudaStream_t*] : CUDA stream
*/
void cu_memcpy_d2h_async(void *d_arr,
                         void *h_arr,
                         size_t size,
                         cudaStream_t *stream)
{
    gpuErrchk(cudaMemcpyAsync(h_arr, d_arr, size, cudaMemcpyDeviceToHost, *stream));
    return;
}

/**
*  Copy a chunk of memory from the device to device. This copy can be
*  from one device to another, or to another memory space on the same
*  device.
*  @param d_src - [void*] : Pointer to device source memory.
*  @param d_dst - [void*] : Pointer to device destination memory.
*  @param size - [size_t] : Size of the transfer in bytes.
*/
void cu_memcpy_d2d(void *d_src,
                   void *d_dst,
                   size_t size)
{
    gpuErrchk(cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice));
    return;
}

/**
*  Async copy a chunk of memory from the device to device. This copy can
*  be from one device to another, or to another memory space on the same
*  device.
*  @param d_src - [void*] : Pointer to device source memory.
*  @param d_dst - [void*] : Pointer to device destination memory.
*  @param size - [size_t] : Size of the transfer in bytes.
*  @param stream - [cudaStream_t*] : CUDA stream
*/
void cu_memcpy_d2d_async(void *d_src,
                         void *d_dst,
                         size_t size,
                         cudaStream_t *stream)
{
    gpuErrchk(cudaMemcpyAsync(d_dst, d_src, size, cudaMemcpyDeviceToDevice, *stream));
    return;
}

/**
*  Set the byte value of the memory on the device.
*  @param d_arr - [void*] : Pointer to device memory.
*  @param value - [int] : Value to set.
*  @param size - [size_t] : Size in bytes to set.
*/
void cu_memset(void *d_arr,
               int value,
               size_t size)
{
    gpuErrchk(cudaMemset(d_arr, value, size));
    return;
}

/**
*  Async set the byte value of the memory on the device.
*  @param d_arr - [void*] : Pointer to device memory.
*  @param value - [int] : Value to set.
*  @param size - [size_t] : Size in bytes to set.
*  @param stream - [cudaStream_t*] : CUDA stream
*/
void cu_memset_async(void *d_arr,
                     int value,
                     size_t size,
                     cudaStream_t *stream)
{
    gpuErrchk(cudaMemsetAsync(d_arr, value, size, *stream));
    return;
}

/**
*  Pin host memory space so that it works with CUDA streams.
*  @param h_arr - [void*] : Pointer to host memory.
*  @param size - [size_t] : Size in bytes to pin.
*/
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

/**
*  Unpin host memory space.
*  @param h_arr - [void*] : Pointer to host memory.
*/
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

/**
*  Create and return the copyparms3D object that is used by a
*  subsequent cudaMemcpy3D call.
*  @param src_Array - [void*] : Pointer to host memory.
*  @param dst_Array - [cudaArray*] : Pointer to device cudaArray.
*  @param extent - [cudaExtent] : Dimensions of 3D memory [width,height,depth].
*  @param element_size_bytes [unsigned int] : Size of each element in bytes.
*/
cudaMemcpy3DParms cu_copyparams(void *src_Array,
                                cudaArray *dst_Array,
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

/**
*  Copy a host array to a 3D cudaArray.
*  @param src_Array - [void*] : Pointer to host memory.
*  @param dst_Array - [cudaArray*] : Pointer to device cudaArray.
*  @param extent - [dim3] : Dimensions of 3D memory [x,y,z].
*  @param element_size_bytes [unsigned int] : Size of each element in bytes.
*/
void cu_memcpy_3d(void *src_Array,
                  cudaArray *dst_Array,
                  dim3 extent,
                  unsigned int element_size_bytes)
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

    gpuErrchk(cudaMemcpy3D(&copyParams));

    return;
}

/**
*  Async Copy a host array to a 3D cudaArray.
*  @param src_Array - [void*] : Pointer to host memory.
*  @param dst_Array - [cudaArray*] : Pointer to device cudaArray.
*  @param extent - [dim3] : Dimensions of 3D memory [x,y,z].
*  @param element_size_bytes [unsigned int] : Size of each element in bytes.
*  @param stream - [cudaStream_t*] : CUDA stream
*/
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
