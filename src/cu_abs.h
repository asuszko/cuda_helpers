#ifndef CU_ABS_H
#define CU_ABS_H


#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    void DLL_EXPORT cu_abs(void *x, unsigned long long N,
                           const int dtype, int dtype_len,
                           cudaStream_t *stream=NULL);

}


#endif/* ifndef CU_ABS_H */
