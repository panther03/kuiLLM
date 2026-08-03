"""
A collection of unverified CUDA operator implementations, or otherwise not fully integrated Kuiper kernels.
These are mainly used as a reference for benchmarking (what is the equivalent CUDA implementation for some piece of code, 
i.e. what is the overhead of Kuiper itself). 

Conventions for kernels added here:
- <kernel>.cu: The main source/implementation of the kernel. 
   Should implement a `<kernel>_launch` function that is the main entry point into that kernel.
- kernels.h: A header file with all (or most) of the forward declarations for the `_launch` functions.
   The only exception is extracted Kuiper kernels; these generally come with a header file, and it's ok to put those in.
   But make sure you rename the file and the function to fit with the convention.
- wrapper.cpp: Where the PyTorch-facing implementation of the kernel should go. Should implement a C++ function called <kernel>
    and expose <kernel> as a Python function. As with Kuiops operators, the wrapper code should generally be minimal and not 
    call PyTorch implementations of kernels (such as elementwise casts), because that defeats the purpose.
    In the module definitions at the bottom of the file, try to give a descriptive comment for the kernel.

Make sure to <kernel> is a short, but descriptive name for what it implements. Mainly we want to know what kind of optimizations
it is doing.
"""

# TODO: implement me