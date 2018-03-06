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

}

#endif /* ifndef CU_DEVICE_H */
