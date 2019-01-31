#ifndef CU_STREAM_H
#define CU_STREAM_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

    cudaStream_t DLL_EXPORT *cu_stream_create();

    void DLL_EXPORT cu_stream_destroy(cudaStream_t *stream);

}

#endif /* ifndef CU_STREAM_H */
