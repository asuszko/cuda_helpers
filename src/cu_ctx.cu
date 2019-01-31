#include <cassert>
#include <cuda.h>
#include "cu_errchk.h"
#include "cu_ctx.h"

#include <iostream>
using namespace std;

/**
 *  Initialize a CUDA context on a device.
 *  @param device - [int] : CUDA device.
 *  @return ctx - [CUcontext*] : New floating CUDA context handle.
 */
CUcontext *cu_context_create(int device)
{
    CUcontext *ctx = new CUcontext;
    
    int num_devices = 0;
    gpuErrchk(cudaGetDeviceCount(&num_devices));
    assert(device < num_devices && "Invalid device_id in cu_create_context");

    gpuContextErrchk(cuCtxCreate(ctx,
                                 CU_CTX_MAP_HOST,
                                 device));

    return ctx;
}

/**
 *  Push a CUDA context to the current thread stack.
 *  @param ctx - [CUcontext*] : Floating CUDA context handle.
 */
void cu_context_push(CUcontext *ctx)
{
    /* Pop the existing context if it exists, then push the new one,
    otherwise, push the context. */
    CUcontext tmp_ctx;
    gpuContextErrchk(cuCtxGetCurrent(&tmp_ctx));
    if(tmp_ctx != NULL) {
        cu_context_pop(&tmp_ctx);
    }
    gpuContextErrchk(cuCtxPushCurrent(*ctx));
    return;
}

/**
 *  Pop a CUDA context from the current thread stack.
 *  @param ctx - [CUcontext*] : Floating CUDA context handle.
 */
void cu_context_pop(CUcontext *ctx)
{
    gpuContextErrchk(cuCtxPopCurrent(ctx));
    return;
}

/**
 *  Destroy a CUDA context, implicitly freeing all resources associated with it.
 *  @param ctx - [CUcontext*] : Floating CUDA context handle.
 */
void cu_context_destroy(CUcontext *ctx)
{
    gpuContextErrchk(cuCtxDestroy(*ctx));
    delete[] ctx;
    return;
}
