CC=g++
CFLAGS=-O3 -ffast-math -funroll-loops -c -I/opt/alps/include -I/home/attila/alps-src/boost -I/usr/include/python2.7 -w
LDFLAGS=-L/opt/alps/lib -lalps -lboost -lgsl -lgslcblas

all:	lrpm_cpu

lrpm_cpu:	lrpm_cpu.o
	$(CC) $(LDFLAGS) lrpm_cpu.o -o lrpm_cpu

lrpm_cpu.o:	lrpm_cpu.cpp
	$(CC) $(CFLAGS) lrpm_cpu.cpp -o lrpm_cpu.o

clean:
	rm -rf *.o lrpm_cpu
