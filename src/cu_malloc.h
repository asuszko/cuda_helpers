#ifndef CU_MALLOC_H
#define CU_MALLOC_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

    void DLL_EXPORT *cu_malloc(size_t size);
    
    void DLL_EXPORT **cu_malloc_dblptr(void *A_dflat, 
                                       unsigned long long N,
                                       int batch_size,
                                       int dtype);

    cudaArray DLL_EXPORT *cu_malloc_3d(cudaChannelFormatDesc *channel,
                                       dim3 extent,
                                       bool layered);

    void DLL_EXPORT *cu_malloc_managed(size_t size);
    
    void DLL_EXPORT cu_free(void *d_arr);

}

#endif /* ifndef CU_MALLOC_H */
