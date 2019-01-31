#ifndef CU_DEVICE_H
#define CU_DEVICE_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

    cudaDeviceProp DLL_EXPORT cu_device_props(int device);

    int DLL_EXPORT cu_device_count();

    void DLL_EXPORT cu_device_reset();
    
    void DLL_EXPORT cu_get_mem_info(size_t *free, size_t *total);

}

#endif /* ifndef CU_DEVICE_H */
