# -*- coding: utf-8 -*-

__all__ = [
    "cu_conj",
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
    "cu_free",
    "cu_get_mem_info",
    "cu_iadd_val",
    "cu_iadd_vec",
    "cu_idiv_val",
    "cu_idiv_vec",
    "cu_imul_val",
    "cu_imul_vec",
    "cu_isub_val",
    "cu_isub_vec",
    "cu_malloc",
    "cu_malloc_3d",
    "cu_malloc_managed",
    "cu_memcpy_d2d",
    "cu_memcpy_d2d_async",
    "cu_memcpy_d2h",
    "cu_memcpy_d2h_async",
    "cu_memcpy_h2d",
    "cu_memcpy_h2d_async",
    "cu_memcpy_3d",
    "cu_memcpy_3d_async",
    "cu_mempin",
    "cu_memunpin",
    "cu_memset",
    "cu_memset_async",
    "cu_stream_create",
    "cu_stream_destroy",
    "cu_sync_device",
    "cu_sync_stream",
    "cu_transpose",
]

import os
from numpy.ctypeslib import ndpointer
from ctypes import (c_bool,
                    c_int,
                    c_void_p,
                    c_size_t,
                    POINTER)

# User imports
from cuda_structs import channelDesc, deviceProps

# Load the shared library
from shared_utils import load_lib
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib"))
cu_lib = load_lib(lib_path,"cuda")


# Define argtypes for all functions to import
argtype_defs = {
        
    "cu_conj" :                  [c_void_p,         #Vector take take conj of
                                  c_int,            #Length of vector
                                  c_int,            #Data type identifier
                                  c_void_p],        #CUDA stream

    "cu_free" :                  [c_void_p],        #Pointer to device array

    "cu_iadd_val" :              [c_void_p,         #Vector y to add onto
                                  ndpointer(),      #Scalar to add
                                  c_int,            #Length of vector y
                                  c_int,            #Data type identifier     
                                  c_int,            #Data type depth
                                  c_void_p],        #CUDA stream

    "cu_iadd_vec" :              [c_void_p,         #Vector y to add onto
                                  c_void_p,         #Vector x to add
                                  c_int,            #Length of vector y
                                  c_int,            #Data type identifier     
                                  c_int,            #Data type depth
                                  c_void_p],        #CUDA stream

    "cu_idiv_val" :              [c_void_p,         #Vector y to divide onto
                                  ndpointer(),      #Scalar to divide
                                  c_int,            #Length of vector y
                                  c_int,            #Data type identifier     
                                  c_int,            #Data type depth
                                  c_void_p],        #CUDA stream

    "cu_idiv_vec" :              [c_void_p,         #Vector y to divide onto
                                  c_void_p,         #Vector x to divide
                                  c_int,            #Length of vector y
                                  c_int,            #Data type identifier     
                                  c_int,            #Data type depth
                                  c_void_p],        #CUDA stream

    "cu_imul_val" :              [c_void_p,         #Vector y to multiply onto
                                  ndpointer(),      #Scalar to multiply
                                  c_int,            #Length of vector y
                                  c_int,            #Data type identifier     
                                  c_int,            #Data type depth
                                  c_void_p],        #CUDA stream

    "cu_imul_vec" :              [c_void_p,         #Vector y to multiply onto
                                  c_void_p,         #Vector x to multiply
                                  c_int,            #Length of vector y
                                  c_int,            #Data type identifier     
                                  c_int,            #Data type depth
                                  c_void_p],        #CUDA stream

    "cu_isub_val" :              [c_void_p,         #Vector y to subtract onto
                                  ndpointer(),      #Scalar to subtract
                                  c_int,            #Length of vector y
                                  c_int,            #Data type identifier     
                                  c_int,            #Data type depth
                                  c_void_p],        #CUDA stream

    "cu_isub_vec" :              [c_void_p,         #Vector y to subtract onto
                                  c_void_p,         #Vector x to subtract
                                  c_int,            #Length of vector y
                                  c_int,            #Data type identifier     
                                  c_int,            #Data type depth
                                  c_void_p],        #CUDA stream
        
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
    
    "cu_get_mem_info" :         [ndpointer(),       #Free memory in bytes
                                 ndpointer()],      #Total memory in bytes
                            
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

    "cu_memcpy_3d" :             [ndpointer(),      #Pointer to host dst array
                                 c_void_p,          #Pointer to device src array
                                 ndpointer("i4"),   #Dim3 extent
                                 c_int],            #Element size in bytes

    "cu_memcpy_3d_async" :      [ndpointer(),       #Pointer to host dst array
                                 c_void_p,          #Pointer to device src array
                                 ndpointer("i4"),   #Dim3 extent
                                 c_int,             #Element size in bytes
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

    "cu_transpose" :            [c_void_p,          #Pointer to device array
                                 c_int,             #Nrows
                                 c_int,             #Ncols
                                 c_int,             #Data type identifier
                                 c_void_p],         #Pointer to CUDA stream   
    
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
