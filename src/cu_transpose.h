#ifndef CU_TRANSPOSE_H
#define CU_TRANSPOSE_H


#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    void DLL_EXPORT cu_transpose(void *dev_ptr, int nrows, int ncols, int dtype_id,
                                 cudaStream_t *stream=NULL);

}


#endif/* ifndef CU_TRANSPOSE_H */
