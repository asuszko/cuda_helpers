#ifndef CU_SYNC_H
#define CU_SYNC_H


#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    void DLL_EXPORT cu_sync_device();

    void DLL_EXPORT cu_sync_stream(cudaStream_t *stream);

}


#endif/* ifndef CU_SYNC_H */
