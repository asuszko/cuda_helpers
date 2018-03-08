#include <cuda.h>
#include "cu_errchk.h"
#include "cu_channel.h"

/**
*  Create a char cudaChannelFormatDesc object. CUDA has built-in support 
*  for vector types: multi-dimensional data with 1 to 4 components, 
*  addressed by x,y,x,w.
*  @param ncomponents - [int] : Number of components in the channel.
*  @param is_unsigned - [bool] : Unsigned flag.
*  @return channelDescData - [cudaChannelFormatDesc] : char cudaChannelFormatDesc object.
*/
cudaChannelFormatDesc cu_create_channel_char(int ncomponents,
                                             bool is_unsigned)
{
    cudaChannelFormatDesc channelDescData;
    if(is_unsigned) {
        switch(ncomponents) {
            case 1: {
                channelDescData = cudaCreateChannelDesc<unsigned char>();
                break;
            }

            case 2: {
                channelDescData = cudaCreateChannelDesc<uchar2>();
                break;
            }

            case 3: {
                channelDescData = cudaCreateChannelDesc<uchar3>();
                break;
            }

            case 4: {
                channelDescData = cudaCreateChannelDesc<uchar4>();
                break;
            }
        }
    }
    else {
        switch(ncomponents) {
            case 1: {
                channelDescData = cudaCreateChannelDesc<char>();
                break;
            }

            case 2: {
                channelDescData = cudaCreateChannelDesc<char2>();
                break;
            }

            case 3: {
                channelDescData = cudaCreateChannelDesc<char3>();
                break;
            }

            case 4: {
                channelDescData = cudaCreateChannelDesc<char4>();
                break;
            }
        }
    }
    return channelDescData;
}

/**
*  Create a short cudaChannelFormatDesc object. CUDA has built-in support 
*  for vector types: multi-dimensional data with 1 to 4 components, 
*  addressed by x,y,x,w.
*  @param ncomponents - [int] : Number of components in the channel.
*  @param is_unsigned - [bool] : Unsigned flag.
*  @return channelDescData - [cudaChannelFormatDesc] : short cudaChannelFormatDesc object.
*/
cudaChannelFormatDesc cu_create_channel_short(int ncomponents,
                                              bool is_unsigned)
{
    cudaChannelFormatDesc channelDescData;
    if(is_unsigned) {
        switch(ncomponents) {
            case 1: {
                channelDescData = cudaCreateChannelDesc<unsigned short>();
                break;
            }

            case 2: {
                channelDescData = cudaCreateChannelDesc<ushort2>();
                break;
            }

            case 3: {
                channelDescData = cudaCreateChannelDesc<ushort3>();
                break;
            }

            case 4: {
                channelDescData = cudaCreateChannelDesc<ushort4>();
                break;
            }
        }
    }
    else {
        switch(ncomponents) {
            case 1: {
                channelDescData = cudaCreateChannelDesc<short>();
                break;
            }

            case 2: {
                channelDescData = cudaCreateChannelDesc<short2>();
                break;
            }

            case 3: {
                channelDescData = cudaCreateChannelDesc<short3>();
                break;
            }

            case 4: {
                channelDescData = cudaCreateChannelDesc<short4>();
                break;
            }
        }
    }
    return channelDescData;
}

/**
*  Create a half cudaChannelFormatDesc object. CUDA has built-in support 
*  for vector types: multi-dimensional data with 1 to 4 components, 
*  addressed by x,y,x,w.
*  @return channelDescData - [cudaChannelFormatDesc] : half cudaChannelFormatDesc object.
*/
cudaChannelFormatDesc cu_create_channel_half()
{
    cudaChannelFormatDesc channelDescData;
    channelDescData = cudaCreateChannelDescHalf();
    return channelDescData;
}

/**
*  Create a float cudaChannelFormatDesc object. CUDA has built-in support 
*  for vector types: multi-dimensional data with 1 to 4 components, 
*  addressed by x,y,x,w.
*  @param ncomponents - [int] : Number of components in the channel.
*  @return channelDescData - [cudaChannelFormatDesc] : float cudaChannelFormatDesc object.
*/
cudaChannelFormatDesc cu_create_channel_float(int ncomponents)
{
    cudaChannelFormatDesc channelDescData;
    switch(ncomponents) {
        case 1: {
            channelDescData = cudaCreateChannelDesc<float>();
            break;
        }

        case 2: {
            channelDescData = cudaCreateChannelDesc<float2>();
            break;
        }

        case 3: {
            channelDescData = cudaCreateChannelDesc<float3>();
            break;
        }

        case 4: {
            channelDescData = cudaCreateChannelDesc<float4>();
            break;
        }
    }
    return channelDescData;
}