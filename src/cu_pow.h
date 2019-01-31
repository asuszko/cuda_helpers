#ifndef CU_POW_H
#define CU_POW_H


#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    void DLL_EXPORT cu_ipow(void *x, unsigned long long N, void *b,
                            int dtype, cudaStream_t *stream=NULL);
								
}


#endif/* ifndef CU_POW_H */
