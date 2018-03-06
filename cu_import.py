# -*- coding: utf-8 -*-

__all__ = [
    "cu_context_create",
    "cu_context_destroy",
    "cu_context_pop",
    "cu_context_push",
    "cu_create_channel_char",
    "cu_create_channel_half",
    "cu_create_channel_short",
    "cu_create_channel_float",
    "cu_device_count",
    "cu_device_props",
    "cu_device_reset",
    "cu_malloc",
    "cu_malloc_3d",
    "cu_malloc_managed",
    "cu_memcpy_d2d",
    "cu_memcpy_d2d_async",
    "cu_memcpy_d2h",
    "cu_memcpy_d2h_async",
    "cu_memcpy_h2d",
    "cu_memcpy_h2d_async",
    "cu_memcpy_3d_async",
    "cu_mempin",
    "cu_memunpin",
    "cu_memset",
    "cu_memset_async",
    "cu_stream_create",
    "cu_stream_destroy",
    "cu_sync_device",
    "cu_sync_stream",    
]

import os
import sys
from numpy.ctypeslib import ndpointer
from ctypes import (c_bool,
                    c_int,
                    c_void_p,
                    c_size_t,
                    POINTER)

# User imports
from cuda_structs import channelDesc, deviceProps

# Load the shared library
sys.path.append("..")
from shared_utils.load_lib import load_lib
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib"))
cu_lib = load_lib(lib_path,"cuda")


# Define argtypes for all functions to import
argtype_defs = {

    "cu_context_create" :       [c_int],            #CUDA device id
      
    "cu_context_destroy" :      [c_void_p],         #Pointer to CUDA ctx
    
    "cu_context_pop" :          [c_void_p],         #Pointer to CUDA ctx
    
    "cu_context_push" :         [c_void_p],         #Pointer to CUDA ctx

    "cu_create_channel_char" :  [c_int,             #Number of components in channel
                                 c_bool],           #Unsigned flag                   
               
    "cu_create_channel_short" : [c_int,             #Number of components in channel
                                 c_bool],           #Unsigned flag 

    "cu_create_channel_half" :  [],                 #Not yet tested, half precision channel
                    
    "cu_create_channel_float" : [c_int,             #Number of components in channel
                                c_bool],            #Unsigned flag 
 
    "cu_device_count" :         [],

    "cu_device_props" :         [c_int],            #CUDA device id
    
    "cu_device_reset" :         [],
                            
    "cu_malloc" :               [c_size_t],         #Size in bytes
    
    "cu_malloc_3d" :            [POINTER(channelDesc), #Pointer to the CUDA channel object
                                 ndpointer("i4"),      #Int array [x,y,z]
                                 c_bool],              #Layered arrat flag
    
    "cu_malloc_managed" :       [c_size_t],         #Size in bytes
    
    "cu_memcpy_d2d" :           [c_void_p,          #Pointer to device src array
                                 c_void_p,          #Pointer to device dst array
                                 c_size_t],         #Size in bytes
    
    "cu_memcpy_d2d_async" :     [c_void_p,          #Pointer to device src array
                                 c_void_p,          #Pointer to device dst array
                                 c_size_t,          #Size in bytes
                                 c_void_p],         #Pointer to CUDA stream
    
    "cu_memcpy_d2h" :           [c_void_p,          #Pointer to device src array
                                 ndpointer(),       #Pointer to host dst array
                                 c_size_t],         #Size in bytes
    
    "cu_memcpy_d2h_async" :     [c_void_p,          #Pointer to device src array
                                 ndpointer(),       #Pointer to host dst array
                                 c_size_t,          #Size in bytes
                                 c_void_p],         #Pointer to CUDA stream
    
    "cu_memcpy_h2d" :           [c_void_p,          #Pointer to device dst array
                                 ndpointer(),       #Pointer to host src array
                                 c_size_t],         #Size in bytes

    "cu_memcpy_h2d_async" :     [c_void_p,          #Pointer to device dst array
                                 ndpointer(),       #Pointer to host src array
                                 c_size_t,          #Size in bytes
                                 c_void_p],         #Pointer to CUDA stream

    "cu_memcpy_3d_async" :      [ndpointer(),       #Pointer to host dst array
                                 c_void_p,          #Pointer to device src array
                                 ndpointer("i4"),
                                 c_int,
                                 c_void_p],         #Pointer to CUDA stream
   
    "cu_mempin" :               [c_void_p,          #Pointer to host array
                                 c_size_t],         #Size in bytes
                                 
    "cu_memunpin" :             [ndpointer()],      #Pointer to host array
   
    "cu_memset" :               [c_void_p,          #Pointer to device array
                                 c_int,             #Byte value to set
                                 c_size_t],         #Size in bytes to set

    "cu_memset_async" :         [c_void_p,          #Pointer to device array
                                 c_int,             #Byte value to set
                                 c_size_t,          #Size in bytes to set
                                 c_void_p],         #Pointer to CUDA stream
   
    "cu_stream_create" :        [],      
        
    "cu_stream_destroy" :       [c_void_p],         #Pointer to CUDA stream
    
    "cu_sync_device" :          [],

    "cu_sync_stream" :          [c_void_p],         #Pointer to CUDA stream       
    
}

restype_defs = {

    "cu_context_create" :       c_void_p,           #Pointer to CUDA stream
    "cu_create_channel_char" :  channelDesc,        #CUDA channel desc object
    "cu_create_channel_float" : channelDesc,        #CUDA channel desc object
    "cu_create_channel_half" :  channelDesc,        #CUDA channel desc object
    "cu_create_channel_short" : channelDesc,        #CUDA channel desc object
    "cu_device_count" :         c_int,              #Number of CUDA devices
    "cu_device_props":          deviceProps,        #CUDA device props object
    "cu_malloc" :               c_void_p,           #Pointer to device memory
    "cu_malloc_3d" :            c_void_p,           #Pointer to device CUDA array
    "cu_malloc_managed" :       c_void_p,           #Pointer to host+dev managed memory
    "cu_stream_create" :        c_void_p,           #Pointer to CUDA stream
}



# Import functions from shared library
for func, argtypes in argtype_defs.items():
    restype = restype_defs.get(func)
    vars().update({func: cu_lib[func]})
    vars()[func].argtypes = argtypes
    vars()[func].restype = restype
