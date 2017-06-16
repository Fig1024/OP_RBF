# Optimized Recursive Bilateral Filter

This project is a derivative work based on this project:
https://github.com/ufoym/RecursiveBF

The main purpose of this project is to provide a more optimized implementation of the Recursive Bilateral Filter. For more information about the image filter, see the link above

Optimization is based on 3 categories: reducing memory usage, adding multithreading, adding SSE2 / AVX2 C++ intrinsics

* Memory usage: in original implementation, memory usage of RGB32 or RGBA image would be roughtly = width * height * 40 + width * 40. In optimized implemention, it is roughly = width * height * 4 + width * 80 for non-piplined version. And width * height * 12 + width * 80 for pipelined. In general, almost 10x less memery allocation

* Multithreading: original implementation is written as single threaded solution, and in a way that it not easy split into threads. Optimized solution is multithread friendly because it separates the filter into 2 stages, one for horizontal filter pass, other for vertical filter pass. Each filter pass can then be subdivided into user chosen number of threads. For horizontal filter, each thread handles its own row from original data buffer, while for vertical pass, each thread handles its own column block

* SSE2 and AVX2: original implementation is written in basic C++ and while it is possible to select SSE2 or AVX2 optimization guidelines in compiler, the generated code does not properly take advantage of that functionality. Optimized solution provides 2 separate implementations, one written almost exclusively with SSE2 intrinsics, another almost exclusively with AVX2 intrinsics, so the compiler can utilize their capabilities much more effectively.


