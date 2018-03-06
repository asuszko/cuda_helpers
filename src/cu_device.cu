#include <cuda.h>
#include "cu_errchk.h"
#include "cu_device.h"


cudaDeviceProp cu_device_props(int device)
{
    cudaDeviceProp props;
    gpuErrchk(cudaGetDeviceProperties(&props,device));
    return props;
}


int cu_device_count()
{
    int count;
    gpuErrchk(cudaGetDeviceCount(&count));
    return count;
}


void cu_device_reset()
{
	  gpuErrchk(cudaDeviceReset());
    return;
}
