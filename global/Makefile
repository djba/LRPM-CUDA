CC=nvcc
CFLAGS=-O3 -c -I/home/attila/NVIDIA_GPU_Computing_SDK/C/common/inc -I/opt/alps/include -I/home/attila/alps-src/boost -I/usr/include/python2.7 -w 
LDFLAGS=-L/home/attila/NVIDIA_GPU_Computing_SDK/C/lib -L/opt/alps/lib -lcutil_x86_64 -lalps -lboost -lgsl -lgslcblas -lcurand

all:	lrpm_gpu

lrpm_gpu:	lrpm_gpu.o
	$(CC) $(LDFLAGS) lrpm_kernel.o lrpm_gpu.o -o lrpm_gpu

lrpm_gpu.o:	lrpm_gpu.cpp lrpm_kernel.o
	$(CC) $(CFLAGS) lrpm_gpu.cpp -o lrpm_gpu.o

lrpm_kernel.o:	lrpm_kernel.cu
	$(CC) -O3 -arch=sm_20 -c lrpm_kernel.cu -o lrpm_kernel.o

clean:
	rm -rf *.o lrpm_gpu
