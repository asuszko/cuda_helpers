#ifndef CU_MEMCPY_H
#define CU_MEMCPY_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

    void DLL_EXPORT cu_memcpy_h2d(void *d_arr, void *h_arr, size_t size);

    void DLL_EXPORT cu_memcpy_h2d_async(void *d_arr, void *h_arr, size_t size,
    									                  cudaStream_t *stream);

    void DLL_EXPORT cu_memcpy_d2h(void *d_arr, void *h_arr, size_t size);

    void DLL_EXPORT cu_memcpy_d2h_async(void *d_arr, void *h_arr, size_t size,
    										                cudaStream_t *stream);

    void DLL_EXPORT cu_memcpy_d2d(void *d_arr_src, void *d_arr_dst, size_t size);

    void DLL_EXPORT cu_memcpy_d2d_async(void *d_arr_src, void *d_arr_dst, size_t size,
    												            cudaStream_t *stream);

    void DLL_EXPORT cu_memcpy_3d_async(void *src_Array,
                                       cudaArray *dst_Array,
                                       dim3 extent,
                                       unsigned int element_size_bytes,
                                       cudaStream_t *stream);

    void DLL_EXPORT cu_memset(void *d_arr, int value, size_t size);

    void DLL_EXPORT cu_memset_async(void *d_arr, int value, size_t size, cudaStream_t *stream);

    void DLL_EXPORT cu_mempin(void *h_arr, size_t size);

    void DLL_EXPORT cu_memunpin(void *h_arr);

}

#endif /* ifndef CU_MEMCPY_H */
