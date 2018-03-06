#ifndef CU_CTX_H
#define CU_CTX_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

    CUcontext DLL_EXPORT *cu_context_create(int device=0);

    void DLL_EXPORT cu_context_push(CUcontext *ctx);

    void DLL_EXPORT cu_context_pop(CUcontext *ctx);

    void DLL_EXPORT cu_context_destroy(CUcontext *ctx);

}

#endif /* ifndef CU_CTX_H */
