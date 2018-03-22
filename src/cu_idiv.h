#ifndef CU_IDIV_H
#define CU_IDIV_H


#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    void DLL_EXPORT cu_idiv(void *y, void *x, unsigned long long N,
                            const int dtype, int dtype_len, bool vec,
                            cudaStream_t *stream=NULL);

}


#endif/* ifndef CU_IDIV_H */
