# -*- coding: utf-8 -*-

from ctypes import (c_char,
                    c_int,
                    c_void_p,
                    c_size_t,
                    Structure)


#CUDA texture descriptor object
class channelDesc(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int),
                ("z", c_int),
                ("w", c_int),
                ("f", c_int)]


class cudaExtent(Structure):
    _fields_ = [('width', c_size_t),
                ('height', c_size_t),
                ('depth', c_size_t)]


class cudaPos(Structure):
    _fields_ = [('x', c_size_t),
                ('y', c_size_t),
                ('z', c_size_t)]


class cudaPitchedPtr(Structure):
    _fields_ = [('pitch', c_size_t),
                ('ptr', c_void_p),
                ('xsize', c_size_t),
                ('ysize', c_size_t)]


class cudaMemcpy3DParms(Structure):
    _fields_ = [('srcArray', c_void_p),
                ('srcPos', cudaPos),
                ('srcPtr', cudaPitchedPtr),
                ('dstArray', c_void_p),
                ('dstPos', cudaPos),
                ('dstPtr', cudaPitchedPtr),
                ('extent', cudaExtent),
                ('kind', c_int)]


# CUDA device properties object
class deviceProps(Structure):
    _fields_ = [ 
            ('name', c_char * 256),
            ('totalGlobalMem', c_size_t),
            ('sharedMemPerBlock', c_size_t),
            ('regsPerBlock', c_int),
            ('warpSize', c_int),
            ('memPitch', c_size_t),
            ('maxThreadsPerBlock', c_int),
            ('maxThreadsDim', c_int * 3), 
            ('maxGridSize', c_int * 3), 
            ('clockRate', c_int),
            ('totalConstMem', c_size_t),
            ('major', c_int),
            ('minor', c_int),
            ('textureAlignment', c_size_t),
            ('texturePitchAlignment', c_size_t),
            ('deviceOverlap', c_int),
            ('multiProcessorCount', c_int),
            ('kernelExecTimeoutEnabled', c_int),
            ('integrated', c_int),
            ('canMapHostMemory', c_int),
            ('computeMode', c_int),
            ('maxTexture1D', c_int),
            ('maxTexture1DMipmap', c_int),
            ('maxTexture1DLinear', c_int),
            ('maxTexture2D', c_int * 2), 
            ('maxTexture2DMipmap', c_int * 2), 
            ('maxTexture2DLinear', c_int * 3), 
            ('maxTexture2DGather', c_int * 2), 
            ('maxTexture3D', c_int * 3), 
            ('maxTexture3DAlt', c_int * 3), 
            ('maxTextureCubemap', c_int),
            ('maxTexture1DLayered', c_int * 2), 
            ('maxTexture2DLayered', c_int * 3), 
            ('maxTextureCubemapLayered', c_int * 2), 
            ('maxSurface1D', c_int),
            ('maxSurface2D', c_int * 2), 
            ('maxSurface3D', c_int * 3), 
            ('maxSurface1DLayered', c_int * 2), 
            ('maxSurface2DLayered', c_int * 3), 
            ('maxSurfaceCubemap', c_int),
            ('maxSurfaceCubemapLayered', c_int * 2), 
            ('surfaceAlignment', c_size_t),
            ('concurrentKernels', c_int),
            ('ECCEnabled', c_int),
            ('pciBusID', c_int),
            ('pciDeviceID', c_int),
            ('pciDomainID', c_int),
            ('tccDriver', c_int),
            ('asyncEngineCount', c_int),
            ('unifiedAddressing', c_int),
            ('memoryClockRate', c_int),
            ('memoryBusWidth', c_int),
            ('l2CacheSize', c_int),
            ('maxThreadsPerMultiProcessor', c_int),
            ('streamPrioritiesSupported', c_int),
            ('globalL1CacheSupported', c_int),
            ('localL1CacheSupported', c_int),
            ('sharedMemPerMultiprocessor', c_size_t),
            ('regsPerMultiprocessor', c_int),
            ('managedMemSupported', c_int),
            ('isMultiGpuBoard', c_int),
            ('multiGpuBoardGroupID', c_int),
            ('singleToDoublePrecisionPerfRatio', c_int),
            ('pageableMemoryAccess', c_int),
            ('concurrentManagedAccess', c_int),
            ]