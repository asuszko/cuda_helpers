#ifndef CU_ABS_H
#define CU_ABS_H


#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    void DLL_EXPORT cu_abs(void *y, void *x, unsigned long long N,
                           int dtype, int dtype_len,
                           cudaStream_t *stream);
                                
    void DLL_EXPORT cu_iabs(void *y, unsigned long long N,
                            int dtype, int dtype_len,
                            cudaStream_t *stream);

}


#endif/* ifndef CU_ABS_H */
