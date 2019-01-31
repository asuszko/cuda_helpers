#ifndef CU_CHANNEL_H
#define CU_CHANNEL_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

    cudaChannelFormatDesc DLL_EXPORT cu_create_channel_char(int ncomponents,
                                                            bool is_unsigned=false);

    cudaChannelFormatDesc DLL_EXPORT cu_create_channel_short(int ncomponents,
                                                             bool is_unsigned=false);

    cudaChannelFormatDesc DLL_EXPORT cu_create_channel_half();

    cudaChannelFormatDesc DLL_EXPORT cu_create_channel_float(int ncomponents);

}

#endif /* ifndef CU_CHANNEL_H */
