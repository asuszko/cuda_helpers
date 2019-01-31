#ifndef CU_CONJ_H
#define CU_CONJ_H


#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    void DLL_EXPORT cu_conj(void *d_data,
                            const unsigned long long N,
                            const int dtype,
                            cudaStream_t *stream=NULL);

}


#endif/* ifndef CU_CONJ_H */
