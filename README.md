# cuda_helpers

Submodule of a Python framework that allows a user to **GPU accelerate their Python code**. An object is created that handles the interface between Python and the GPU. This object is part of a larger container within the [cuda_manager](https://github.com/asuszko/cuda_manager). Thus, while this repo may be used standalone, it is recommended it be downloaded with [cuda_manager](https://github.com/asuszko/cuda_manager). Doing a recursive clone on [cuda_manager](https://github.com/asuszko/cuda_manager) will also clone [cuda_helpers](https://github.com/asuszko/cuda_helpers).

## Setup

To compile the shared library, run the **setup.py** file found in the root folder from the command line, with optional argument(s) -arch, and -cc_bin if on Windows. On Windows, the NVCC compiler looks for cl.exe to compile the C/C++, which comes with Visual Studio. On Linux, it uses the built in GCC compiler. An example of a command line run to compile the code is given below:
> python setup.py -arch=sm_50 -cc_bin="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"

If you are unable to compile, you may [download precompiled libraries here](https://github.com/asuszko/cuda_manager_libs).

## Compiler Requirements

- Python 3.6.x (2.7 compatibility not yet tested) 
- The latest  version of the [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- (Windows) Visual Studio 2013 Professional, or any compatible version with your version of CUDA Toolkit. Note: You can download a trial of Professional to obtain cl.exe. Compilation via the command line will still work after the trial period has ended.

## License
 
The MIT License (MIT)

Copyright (c) 2018 (Arthur Suszko (art.suszko@gmail.com), Geoff Gunter (geoffmgunter@gmail.com) and contributors)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.