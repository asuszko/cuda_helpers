#include <cuda.h>
#include "cu_errchk.h"
#include "cu_device.h"

/**
*  Get the cudaGetDeviceProperties object.
*  @param device - [int] : Device id to query.
*  @return props - [cudaDeviceProp] : cudaDeviceProp object.
*/
cudaDeviceProp cu_device_props(int device)
{
    cudaDeviceProp props;
    gpuErrchk(cudaGetDeviceProperties(&props,device));
    return props;
}

/**
*  Get the number of CUDA devices.
*  @return count - [int] : Number of CUDA devices.
*/
int cu_device_count()
{
    int count;
    gpuErrchk(cudaGetDeviceCount(&count));
    return count;
}

/**
*  Reset the device on the current CUDA context.
*/
void cu_device_reset()
{
    gpuErrchk(cudaDeviceReset());
    return;
}

/**
* Get memory info of the device.
*/
void cu_get_mem_info(size_t *free, size_t *total)
{
    gpuErrchk(cudaMemGetInfo(free, total));
    return;	
}